"""CPU-only V3.1 counterexample and tamper checks, without model outputs."""
import copy
import importlib.util
from pathlib import Path

import pytest

from evidence_gate.data import messages_for


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


auditor = module("audit_v31_data")
builder = module("build_v31")


@pytest.fixture(scope="module")
def fresh_rows():
    return builder.generate(train_groups=12, dev_groups=12, test_groups=12)


def test_exact_interventions_both_citations_and_known_rules_are_rejected(fresh_rows):
    before = copy.deepcopy(fresh_rows)
    report = auditor.audit_v31(fresh_rows, expected_variants=builder.VARIANTS)
    assert report["rows"] == report["public_rule_agreements"] == report["recovery_agreements"] == 864
    assert report["single_factor_interventions"] == {"topic_filter": 72, "conflict_agreement": 72}
    assert report["cells_per_group"] == 24
    assert report["model_predictions_read"] is False and report["llm_calls"] == 0
    assert set(report["strict_insufficient_rule_probes"]) == {"train", "dev"}
    for split in ("train", "dev", "test"):
        assert report["missing_topic_subtype_counts"][split] == {"mixed_topic": 6, "uniform_wrong_topic": 6}
    for probes in report["strict_insufficient_rule_probes"].values():
        original = probes["v3_topic_and_count_omitter"]
        patched = probes["patched_uniform_topic_omitter"]
        assert not original["passes_all_seven"] and not patched["passes_all_seven"]
        assert original["summary"]["success_rate"] == 19 / 24
        assert original["summary"]["answerable_success"] == 2 / 3
        assert patched["summary"]["success_rate"] == 23 / 24
        assert patched["summary"]["answerable_success"] == 1
        assert patched["summary"]["over_answer_rate"] == 1 / 12
        assert patched["summary"]["unsupported_answer_rate"] == 1 / 13
        assert {check["gate"] for check in patched["checks"] if not check["passed"]} == {
            "over_answer_rate_max", "unsupported_answer_rate_max"}
    assert fresh_rows == before


def test_insufficient_original_rule_really_passes_previous_v3_train_dev():
    # This establishes that the diagnostic tests the known omission rather than
    # an arbitrary weak predictor. No historical or fresh model output is read.
    previous = builder.v3.generate(train_groups=12, dev_groups=12, test_groups=12)
    for split in ("train", "dev"):
        subset = [row for row in previous if row["split"] == split]
        for patched in (False, True):
            report = auditor._strict_probe(subset, patched)
            assert report["passes_all_seven"]
            assert report["summary"]["success_rate"] == 1


@pytest.mark.parametrize("mutation,expected", [
    ("unrelated_topic_change", "change only the second official predicate"),
    ("unrelated_consensus_change", "change only the second official value"),
    ("missing_citation", "Public contract"),
    ("seen_entity", "previously seen"),
    ("unbalanced_missing", "subtypes must be balanced"),
    ("row_identity", "row identity"),
    ("variant_matrix", "twenty-four variants"),
])
def test_tampering_is_rejected(fresh_rows, mutation, expected):
    rows, seen, variants = copy.deepcopy(fresh_rows), [], list(builder.VARIANTS)
    if mutation.startswith("unrelated_"):
        semantic = "irrelevant_topic" if mutation == "unrelated_topic_change" else "consensus"
        group = rows[0]["group_id"]
        for row in rows:
            if row["group_id"] == group and row["variant"].startswith(semantic + "_"):
                # Labels remain correct, budget pairing remains intact, and
                # the shared injection still occurs once. The pure-intervention
                # check, rather than an unrelated format check, must reject it.
                community = next(doc for doc in row["observation"]["documents"] if doc["source"] == "community")
                community["text"] += " Extra unrelated words."
                row["messages"] = messages_for(row["observation"])
    elif mutation == "missing_citation":
        row = next(row for row in rows if row["variant"] == "consensus_b0")
        row["target"]["citations"].pop()
    elif mutation == "seen_entity":
        seen = [{"catalogue_only_old_entity": rows[0]["gold"]["product"]}]
    elif mutation == "unbalanced_missing":
        chosen = next(row for row in rows if row["variant"] == "missing_b0" and row["design"]["missing_uniform_topic_factor"])
        primary_id = next(row["target"]["citations"][0] for row in rows
                          if row["group_id"] == chosen["group_id"] and row["variant"] == "supported_b0")
        question_topic = next(p for _, p, topic in auditor.TOPICS if topic in chosen["observation"]["question"])
        for row in rows:
            if row["group_id"] == chosen["group_id"] and row["variant"].startswith("missing_"):
                secondary = next(doc for doc in row["observation"]["documents"] if doc["source"] == "official" and doc["id"] != primary_id)
                previous_topic = auditor._fact(secondary)["predicate"]
                secondary["text"] = secondary["text"].replace(previous_topic, question_topic)
                row["messages"] = messages_for(row["observation"])
    elif mutation == "row_identity":
        rows[0]["id"] += "-wrong"
    elif mutation == "variant_matrix":
        variants[-1] = "different_semantic_b1"
    with pytest.raises(ValueError, match=expected):
        auditor.audit_v31(rows, seen, variants)


def test_fresh_domain_is_disjoint_from_all_three_older_versions(fresh_rows):
    seen = [row for version in ("main", "v2", "v3") for split in ("train", "dev", "test")
            for row in auditor.read_jsonl(ROOT / "data" / version / f"{split}.jsonl")]
    report = auditor.audit_v31(fresh_rows, seen, builder.VARIANTS)
    assert report["previously_seen_entities"] == len(auditor.v3.entities(seen))
    assert report["previously_seen_entity_overlap"] == 0
