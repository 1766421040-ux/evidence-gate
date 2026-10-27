"""Fresh identities, semantic counterfactuals and immutable V3 freeze records."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import generate as generate_v1, messages_for, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_v3", ROOT / "scripts/build_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
spec = importlib.util.spec_from_file_location("audit_v3_for_tests", ROOT / "scripts/audit_v3_data.py")
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


@pytest.fixture(scope="module")
def rows():
    return v3.generate(train_groups=12, dev_groups=12, test_groups=12)


def test_new_twenty_cell_contract_passes_independent_counterexample_audit(rows):
    audit = independent.audit_v3(rows, expected_variants=v3.VARIANTS)
    assert audit["rows"] == 720
    assert audit["groups"] == 36
    assert audit["public_rule_agreements"] == audit["recovery_agreements"] == 720
    assert audit["llm_calls"] == 0 and audit["model_predictions_read"] is False
    report = v3.validate(rows)
    assert report["cells_per_group"] == 20
    assert report["variants"] == list(v3.VARIANTS)
    assert report["cross_group_entity_overlap"] == 0
    assert all(row["messages"] == messages_for(row["observation"]) for row in rows)


def test_expanding_train_cannot_change_holdout_or_reuse_split_identities(rows):
    expanded = v3.generate(train_groups=24, dev_groups=12, test_groups=12)
    assert [row for row in expanded if row["split"] != "train"] == [row for row in rows if row["split"] != "train"]
    by_split = {split: v3.entities([row for row in expanded if row["split"] == split]) for split in v3.SPLITS}
    assert not by_split["train"] & by_split["dev"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["dev"] & by_split["test"]
    assert expanded == v3.generate(train_groups=24, dev_groups=12, test_groups=12)


def test_previous_distractor_and_cross_group_entity_reuse_are_rejected(rows):
    row = next(row for row in rows if len(v3.entities(row)) > 1)
    distractor = next(iter(v3.entities(row) - {row["gold"]["product"]}))
    with pytest.raises(ValueError, match="previously seen"):
        v3.validate(rows, {distractor})
    mutated = copy.deepcopy(rows)
    another = next(row for row in rows if row["split"] == "test")["gold"]["product"]
    mutated[0]["gold"]["catalogue"][0]["text"] += f" Reference: {another}."
    with pytest.raises(ValueError, match="Entity reused"):
        v3.validate(mutated)


def test_budget_and_value_interventions_reject_collateral_changes(rows):
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["group_id"] == changed[0]["group_id"] and row["variant"] == "supported_b1":
            row["observation"]["documents"] = list(reversed(row["observation"]["documents"]))
            row["messages"] = messages_for(row["observation"])
    with pytest.raises(ValueError, match="Budget pair"):
        v3.validate(changed)
    changed = copy.deepcopy(rows)
    group_id = changed[0]["group_id"]
    for row in changed:
        if row["group_id"] == group_id and row["variant"].startswith("value_changed"):
            row["observation"]["documents"] = list(reversed(row["observation"]["documents"]))
            row["messages"] = messages_for(row["observation"])
    with pytest.raises(ValueError, match="unrelated observable"):
        v3.validate(changed)


def test_missing_or_misnamed_cells_and_label_leak_are_rejected(rows):
    with pytest.raises(ValueError, match="twenty"):
        v3.validate(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[0]["group_id"] = "v3-test-g0000"
    with pytest.raises(ValueError, match="exact split"):
        v3.validate(changed)
    changed = copy.deepcopy(rows)
    changed[0]["messages"][-1]["content"] += " expected=answer"
    with pytest.raises(ValueError, match="boundary"):
        v3.validate(changed)


def test_freeze_binds_all_old_splits_data_contract_and_independent_audit(tmp_path):
    main, old_v2 = tmp_path / "main", tmp_path / "v2"
    for source, values in ((main, generate_v1(train_groups=1, dev_groups=1, test_groups=1)),
                           (old_v2, v3.v2.generate(train_groups=3, dev_groups=3, test_groups=3))):
        for split in v3.SPLITS:
            write_jsonl(source / f"{split}.jsonl", [row for row in values if row["split"] == split])
    before = {path: path.read_bytes() for source in (main, old_v2) for path in source.iterdir()}
    data, metadata = tmp_path / "new-data", tmp_path / "metadata"
    manifest = v3.build(data, metadata, sources=(main, old_v2), train_groups=12, dev_groups=12, test_groups=12)
    protocol = json.loads((metadata / "protocol.json").read_text())
    freeze = json.loads((metadata / "freeze.json").read_text())
    old_gate = json.loads((ROOT / "experiments/v2/protocol.json").read_text())["gate"]
    assert all(protocol["gate"][key] == value for key, value in old_gate.items())
    assert protocol["gate"]["unsupported_answer_rate_max"] == 0.05
    assert len(protocol["variants"]) == protocol["cells_per_group"] == 20
    assert protocol["selection"]["no_eligible_candidate"]["test_allowed"] is False
    assert len(manifest["previously_seen_files_sha256"]) == 6
    assert manifest["models_called"] == 0 and manifest["test_model_inference_performed"] is False
    assert manifest["independent_audit_sha256"] == hashlib.sha256((metadata / "data-audit.json").read_bytes()).hexdigest()
    for split in v3.SPLITS:
        actual = hashlib.sha256((data / f"{split}.jsonl").read_bytes()).hexdigest()
        assert protocol["data_sha256"][split] == manifest["files"][f"{split}.jsonl"]["sha256"] == actual
    assert {path: path.read_bytes() for path in before} == before
    assert freeze["model_predictions_read"] is False
    assert hashlib.sha256((metadata / "protocol.json").read_bytes()).hexdigest() in freeze["files"].values()
    with pytest.raises(FileExistsError, match="Locked"):
        v3.build(data, metadata, sources=(main, old_v2))
    with pytest.raises(ValueError, match="both main and v2"):
        v3.build(tmp_path / "bad-data", tmp_path / "bad-meta", sources=(main,))
