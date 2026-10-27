"""V3.1 adds only declared semantic cells and never alters prior freezes."""
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import canonical, generate as generate_v1, messages_for, write_jsonl
from evidence_gate.evaluation import contract_rule, score


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v31_builder_tests", ROOT / "scripts/build_v31.py")
v31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v31)


@pytest.fixture(scope="module")
def rows():
    return v31.generate(train_groups=12, dev_groups=12, test_groups=12)


def groups(rows):
    result = defaultdict(dict)
    for row in rows:
        result[row["group_id"]][row["variant"]] = row
    return result


def test_twenty_four_cells_and_both_pure_conflict_interventions(rows):
    assert len(rows) == 36 * 24
    assert len(v31.VARIANTS) == 24
    for members in groups(rows).values():
        assert set(members) == set(v31.VARIANTS)
        for semantic in ("irrelevant_topic", "consensus"):
            zero, one = (copy.deepcopy(members[f"{semantic}_b{budget}"]["observation"]) for budget in (0, 1))
            assert zero.pop("budget") == 0 and one.pop("budget") == 1 and zero == one
            for budget in (0, 1):
                source, added = members[f"conflict_b{budget}"], members[f"{semantic}_b{budget}"]
                assert added["target"]["action"] == "answer"
                assert added["messages"] == messages_for(added["observation"])
                assert contract_rule(added["observation"]) == added["target"]
                before = {doc["id"]: doc for doc in source["observation"]["documents"]}
                after = {doc["id"]: doc for doc in added["observation"]["documents"]}
                changed_ids = [identifier for identifier in before if before[identifier] != after[identifier]]
                assert len(changed_ids) == 1
                identifier = changed_ids[0]
                a, b = dict(before[identifier]), dict(after[identifier])
                old_text, new_text = a.pop("text"), b.pop("text")
                assert a == b and a["source"] == "official"
                old_fact, new_fact = v31.FACT.fullmatch(old_text).groups(), v31.FACT.fullmatch(new_text).groups()
                assert old_fact[:2] == new_fact[:2]
                if semantic == "irrelevant_topic":
                    assert old_fact[2] != new_fact[2] and old_fact[3] == new_fact[3]
                    assert len(added["target"]["citations"]) == 1
                else:
                    assert old_fact[2] == new_fact[2] and old_fact[3] != new_fact[3]
                    assert len(added["target"]["citations"]) == 2
                    incomplete = copy.deepcopy(added["target"])
                    incomplete["citations"] = incomplete["citations"][:1]
                    assert score(added, canonical(incomplete))["metrics"]["success"] is False


def test_uniform_wrong_topic_missing_is_balanced_and_still_insufficient(rows):
    for split in v31.SPLITS:
        missing = [row for row in rows if row["split"] == split and row["variant"] == "missing_b0"]
        states = Counter()
        for row in missing:
            predicates = {v31.FACT.fullmatch(doc["text"]).group(3) for doc in row["observation"]["documents"] if doc["source"] == "official"}
            states[len(predicates)] += 1
            assert contract_rule(row["observation"])["action"] == "abstain"
        assert states == {1: 6, 2: 6}
        offsets = {row["design"]["irrelevant_topic_offset"] for row in rows if row["split"] == split}
        assert offsets == {1, 2}


def test_new_seed_domains_preserve_holdout_under_train_expansion(rows):
    expanded = v31.generate(train_groups=24, dev_groups=12, test_groups=12)
    assert [row for row in expanded if row["split"] != "train"] == [row for row in rows if row["split"] != "train"]
    old = v31.v3.generate(train_groups=12, dev_groups=12, test_groups=12)
    assert not v31.v3.entities(rows) & v31.v3.entities(old)
    assert all(row["group_id"].startswith(f"v31-{row['split']}-") for row in rows)
    assert expanded == v31.generate(train_groups=24, dev_groups=12, test_groups=12)


def test_all_new_recovery_targets_bind_separate_visible_world_ids(rows):
    for row in rows:
        if not row["variant"].startswith(("irrelevant_topic", "consensus")):
            continue
        visible = {doc["id"] for doc in row["observation"]["documents"] if doc["source"] == "official"}
        recovered = {**row["observation"], "documents": row["gold"]["catalogue"]}
        assert contract_rule(recovered) == row["gold"]["recovery_target"]
        assert not visible & set(row["gold"]["recovery_target"]["citations"])
        assert row["gold"]["recovery_target"]["value"] == row["target"]["value"]
        assert len(row["gold"]["recovery_target"]["citations"]) == len(row["target"]["citations"])


def test_build_freezes_new_version_without_mutating_older_sources(tmp_path):
    sources = [tmp_path / name for name in ("main", "v2", "v3")]
    values = [generate_v1(train_groups=1, dev_groups=1, test_groups=1),
              v31.v3.v2.generate(train_groups=3, dev_groups=3, test_groups=3),
              v31.v3.generate(train_groups=12, dev_groups=12, test_groups=12)]
    for directory, old_rows in zip(sources, values):
        for split in v31.SPLITS:
            write_jsonl(directory / f"{split}.jsonl", [row for row in old_rows if row["split"] == split])
    snapshots = {path: path.read_bytes() for directory in sources for path in directory.iterdir()}
    output, metadata = tmp_path / "v31", tmp_path / "metadata"
    result = v31.build(output, metadata, sources, train_groups=12, dev_groups=12, test_groups=12)
    protocol = json.loads((metadata / "protocol.json").read_text())
    original_protocol = json.loads((ROOT / "experiments/v3/protocol.json").read_text())
    assert protocol["gate"] == original_protocol["gate"]
    assert protocol["answer_risk_eligibility"] == original_protocol["answer_risk_eligibility"]
    assert protocol["selection"] == original_protocol["selection"]
    assert protocol["inference"] == original_protocol["inference"]
    assert protocol["cells_per_group"] == 24
    assert result["model_predictions_read"] is False and result["models_called"] == 0
    assert len(result["previously_seen_files_sha256"]) == 9
    for split in v31.SPLITS:
        actual = hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
        assert actual == protocol["data_sha256"][split] == result["files"][f"{split}.jsonl"]["sha256"]
    assert {path: path.read_bytes() for path in snapshots} == snapshots
    supersession = json.loads((metadata / "supersession.json").read_text())
    assert supersession["old_artifacts_modified"] is False
    with pytest.raises(FileExistsError, match="already frozen"):
        v31.build(output, metadata, sources)
    with pytest.raises(ValueError, match="main, v2 and v3"):
        v31.build(tmp_path / "bad", tmp_path / "bad-metadata", sources[:2])
