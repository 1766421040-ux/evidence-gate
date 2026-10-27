import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import generate as generate_v1, messages_for, write_jsonl
from evidence_gate.evaluation import contract_rule


spec = importlib.util.spec_from_file_location("build_v2", Path(__file__).resolve().parents[1] / "scripts" / "build_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


@pytest.fixture
def rows():
    return v2.generate(train_groups=6, dev_groups=3, test_groups=3)


def test_new_products_reproducibility_and_rule_targets(rows):
    assert rows == v2.generate(train_groups=6, dev_groups=3, test_groups=3)
    forbidden = {r["gold"]["product"] for r in generate_v1()}
    report = v2.validate(rows, forbidden)
    assert report["groups"] == report["products"] == 12
    assert report["rows"] == report["public_rule_target_agreements"] == 168
    assert report["recovery_world_agreements"] == 168
    assert report["v1_product_overlap"] == report["group_leaks"] == 0


def test_count_source_and_injection_cannot_predict_action(rows):
    audit = v2.validate(rows)["shortcut_audit"]
    for features in audit.values():
        for key in ("document_count", "source_counts", "injection_count"):
            assert features[key]["mutual_information_bits"] == 0
            assert features[key]["best_lookup_action_accuracy"] == pytest.approx(5 / 14)
        dates = features["source_date_signature"]
        assert len(dates["table"]) == 3
        assert all(set(counts) == {"answer", "search", "clarify", "abstain"} for counts in dates["table"].values())
        assert dates["best_lookup_action_accuracy"] == pytest.approx(5 / 14)
    # The diagnosed v1 count heuristic now always answers, losing all boundary cases.
    assert sum(r["target"]["action"] == "answer" for r in rows) / len(rows) == pytest.approx(4 / 14)


def test_counterfactuals_keep_unintended_fields_fixed(rows):
    for group_id in {r["group_id"] for r in rows}:
        g = {r["variant"]: r for r in rows if r["group_id"] == group_id}
        supported, changed, reordered = (g[x] for x in ("supported", "value_changed", "reordered"))
        assert supported["target"]["value"] != changed["target"]["value"]
        assert supported["target"] == reordered["target"] == g["distractor"]["target"]
        assert supported["observation"]["documents"] == list(reversed(reordered["observation"]["documents"]))
        left, right = (copy.deepcopy(g[x]["observation"]) for x in ("missing_search", "missing_stop"))
        assert left.pop("budget") == 1 and right.pop("budget") == 0
        assert left == right
        assert {len(r["observation"]["documents"]) for r in g.values()} == {3}
        assert changed["gold"]["recovery_target"]["value"] == changed["target"]["value"]
        assert supported["gold"]["recovery_target"]["value"] == supported["target"]["value"]


def test_search_recovery_and_public_boundary(rows):
    from evidence_gate.agent import run_episode
    for row in rows:
        assert row["messages"] == messages_for(row["observation"])
        assert "recovery_target" not in row["messages"][-1]["content"]
        if row["target"]["action"] == "search":
            episode = run_episode(row["observation"], row["gold"]["catalogue"], contract_rule)
            assert episode["searches"] == 1
            assert episode["decision"] == row["gold"]["recovery_target"]


def test_invalid_targets_catalogue_and_overlap_rejected(rows):
    bad = copy.deepcopy(rows)
    bad[0]["target"]["value"] = "999 days"
    with pytest.raises(ValueError, match="disagrees"):
        v2.validate(bad)
    bad = copy.deepcopy(rows)
    bad[0]["gold"]["catalogue"][0]["source"] = "community"
    with pytest.raises(ValueError, match="snapshot is inconsistent"):
        v2.validate(bad)
    with pytest.raises(ValueError, match="overlaps"):
        v2.validate(rows, {rows[0]["gold"]["product"]})


def test_locked_manifest_preserves_v1_and_binds_files(tmp_path):
    source, output = tmp_path / "main", tmp_path / "v2"
    original = generate_v1(train_groups=1, dev_groups=1, test_groups=1)
    for split in ("train", "dev", "test"):
        write_jsonl(source / f"{split}.jsonl", [r for r in original if r["split"] == split])
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    manifest = v2.build(output, source, train_groups=3, dev_groups=3, test_groups=3)
    assert manifest["model_predictions_read"] is False and manifest["models_called"] == 0
    assert "development uses dev only" in manifest["test_policy"]
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for filename, entry in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == entry["sha256"]
    with pytest.raises(FileExistsError, match="locked"):
        v2.build(output, source, train_groups=3, dev_groups=3, test_groups=3)
    with pytest.raises(FileExistsError, match="locked"):
        v2.build(source, source, train_groups=3, dev_groups=3, test_groups=3)
