"""Fake-output scope-pair tests; never load a model or saved model prediction."""
import copy
import importlib.util
from pathlib import Path

import pytest

from evidence_gate.data import canonical, decision, digest
from evidence_gate.evaluation import score, summarize


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


reporter = module("score_scope_pairs")
builder = module("build_v31")


@pytest.fixture(scope="module")
def fixture():
    rows = [row for row in builder.generate(train_groups=12, dev_groups=12, test_groups=12) if row["split"] == "dev"]
    cases = [score(row, canonical(row["target"])) for row in rows]
    evaluation = {"name": "fake-perfect-output", "dataset_sha256": digest(rows), "cases": cases, "summary": summarize(cases)}
    protocol = {"variants": list(builder.VARIANTS), "dev_groups": 12, "test_groups": 12}
    return rows, evaluation, protocol


def test_both_budgets_and_strict_citations_are_counted(fixture):
    rows, evaluation, protocol = copy.deepcopy(fixture)
    result = reporter.scope_pairs(rows, evaluation, protocol)
    assert result["primary_summary_unchanged"] and not result["used_for_selection"]
    assert result["model_calls"] == 0
    for pair in result["pairs"].values():
        assert pair["correct"] == pair["total"] == 24
        assert pair["by_budget"]["0"]["total"] == pair["by_budget"]["1"]["total"] == 12
    index = next(i for i, row in enumerate(rows) if row["variant"] == "consensus_b1")
    prediction = copy.deepcopy(rows[index]["target"])
    prediction["citations"].pop()
    evaluation["cases"][index] = score(rows[index], canonical(prediction))
    evaluation["summary"] = summarize(evaluation["cases"])
    result = reporter.scope_pairs(rows, evaluation, protocol)
    assert result["pairs"]["topic_filter"]["accuracy"] == 1
    agreement = result["pairs"]["conflict_agreement"]
    assert agreement["correct"] == 23 and agreement["action_pair_accuracy"] == 1
    assert agreement["by_budget"]["0"]["accuracy"] == 1
    assert agreement["by_budget"]["1"]["correct"] == 11


@pytest.mark.parametrize("mutation,match", [
    ("raw", "Stored score"), ("identity", "identities"), ("expected", "altered expected"),
    ("summary", "main summary"), ("incomplete", "Incomplete evaluation"),
    ("dataset", "does not match"), ("variants", "twenty-four-cell"),
])
def test_tampered_evidence_is_rejected(fixture, mutation, match):
    rows, evaluation, protocol = copy.deepcopy(fixture)
    if mutation == "raw":
        evaluation["cases"][0]["raw"] = canonical(decision("abstain"))
    elif mutation == "identity":
        evaluation["cases"][0]["id"] = evaluation["cases"][1]["id"]
    elif mutation == "expected":
        evaluation["cases"][0]["expected"] = decision("abstain")
    elif mutation == "summary":
        evaluation["summary"]["success_rate"] = 0
    elif mutation == "incomplete":
        evaluation["cases"].pop()
    elif mutation == "dataset":
        evaluation["dataset_sha256"] = "0" * 64
    elif mutation == "variants":
        protocol["variants"].pop()
    with pytest.raises(ValueError, match=match):
        reporter.scope_pairs(rows, evaluation, protocol)
