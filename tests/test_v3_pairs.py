"""V3 interventions must pair identical budgets and preserve V2 summaries."""
import copy
import json
from pathlib import Path

from evidence_gate.data import canonical, generate_group
from evidence_gate.evaluation import score, summarize


def v3_cases():
    original = {row["variant"]: row for row in generate_group(6, "dev", 19)}
    mapping = {"supported": "supported", "value_changed": "value_changed",
               "distractor": "distractor", "reordered": "reordered", "ambiguous": "ambiguous"}
    cases = []
    for semantic in (*mapping, "missing", "expired", "future", "conflict", "untrusted"):
        for budget in (0, 1):
            source = mapping.get(semantic, semantic + ("_search" if budget else "_stop"))
            source = source if source in original else ("future_search" if semantic == "future" else "untrusted_stop")
            row = copy.deepcopy(original[source])
            row["variant"] = f"{semantic}_b{budget}"
            row["id"] = f"{row['group_id']}:{row['variant']}"
            row["observation"]["budget"] = budget
            if semantic not in mapping:
                row["target"] = {"action": "search" if budget else "abstain", "value": "", "citations": []}
            cases.append(score(row, canonical(row["target"])))
    return cases


def test_v3_pairs_count_both_budgets_and_fail_matching_pair_only():
    cases = v3_cases()
    pairs = summarize(cases)["pairs"]
    assert pairs["value_change"] == {"correct": 2, "total": 2, "accuracy": 1}
    assert pairs["evidence_removal"]["total"] == 2
    assert pairs["budget_exhaustion"]["total"] == 1
    next(case for case in cases if case["variant"] == "expired_b0")["metrics"]["success"] = False
    pairs = summarize(cases)["pairs"]
    assert pairs["expiry_boundary"] == {"correct": 1, "total": 2, "accuracy": .5}
    assert pairs["value_change"]["accuracy"] == 1
    next(case for case in cases if case["variant"] == "supported_b1")["metrics"]["success"] = False
    pairs = summarize(cases)["pairs"]
    assert pairs["answer_budget_invariance"]["accuracy"] == 0
    assert pairs["expiry_boundary"]["accuracy"] == 0


def test_published_v2_raw_summary_remains_exact():
    path = Path(__file__).resolve().parents[1] / "experiments/v2/selected-v2.json"
    evaluation = json.loads(path.read_text())
    assert summarize(evaluation["cases"]) == evaluation["summary"]
