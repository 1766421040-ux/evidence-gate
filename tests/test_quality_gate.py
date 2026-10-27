"""Truthfulness and exit semantics for preregistered test-quality gates."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import canonical, digest, generate, write_jsonl
from evidence_gate.evaluation import score, summarize


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quality_gate", ROOT / "scripts/quality_gate.py")
quality_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality_gate)


@pytest.fixture
def files(tmp_path):
    rows = [row for row in generate(seed=42, train_groups=1, dev_groups=1, test_groups=2) if row["split"] == "test"]
    cases = [score(row, canonical(row["target"]), 1.0) for row in rows]
    evaluation = {"name": "fixture-oracle", "kind": "rule", "dataset_sha256": digest(rows),
                  "cases": cases, "summary": summarize(cases)}
    protocol = json.loads((ROOT / "experiments/v2/protocol.json").read_text())
    protocol["test_groups"] = 2  # Small CPU-only fixture; original protocol is untouched.
    paths = {"data": tmp_path / "test.jsonl", "evaluation": tmp_path / "evaluation.json", "protocol": tmp_path / "protocol.json"}
    write_jsonl(paths["data"], rows)
    paths["evaluation"].write_text(json.dumps(evaluation))
    paths["protocol"].write_text(json.dumps(protocol))
    return paths, rows, evaluation, protocol


def run(paths):
    return quality_gate.audit_gate(paths["evaluation"], paths["data"], paths["protocol"])


def test_correct_complete_predictions_pass_without_changing_inputs(files):
    paths, _, _, protocol = files
    original = {name: path.read_bytes() for name, path in paths.items()}
    report = run(paths)
    assert report["passed"] and report["status"] == "pass"
    assert report["integrity"]["raw_outputs_rescored"]
    assert {check["gate"]: check["threshold"] for check in report["checks"]} == protocol["gate"]
    assert all(check["direction"] in (">=", "<=") for check in report["checks"])
    assert all(path.read_bytes() == original[name] for name, path in paths.items())


def test_zero_parseable_answers_fail_risk_eligibility_and_are_saved(files, tmp_path):
    paths, rows, evaluation, _ = files
    evaluation["cases"] = [score(row, "not valid JSON", 1.0) for row in rows]
    evaluation["summary"] = summarize(evaluation["cases"])
    paths["evaluation"].write_text(json.dumps(evaluation))
    arguments = ["--evaluation", str(paths["evaluation"]), "--data", str(paths["data"]), "--protocol", str(paths["protocol"])]
    output = tmp_path / "failed.json"
    assert quality_gate.main([*arguments, "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "fail" and report["integrity"]["verified"]
    risk = next(check for check in report["checks"] if check["metric"] == "over_answer_rate")
    assert risk["actual"] == 0.0  # Preserve the honest numerical metric.
    assert not risk["eligible"] and not risk["passed"]
    assert report["recomputed_summary"]["unsupported_answer_rate"] is None
    required = tmp_path / "required.json"
    assert quality_gate.main([*arguments, "--require-pass", "--output", str(required)]) == 2
    assert json.loads(required.read_text()) == report


@pytest.mark.parametrize("mutation", ["metrics", "prediction", "summary", "duplicate", "observation", "hash"])
def test_altered_or_incomplete_evidence_never_receives_a_quality_verdict(files, mutation):
    paths, _, evaluation, _ = files
    if mutation == "metrics":
        evaluation["cases"][0]["metrics"]["success"] = 1  # bool/int equality must not hide changes.
    elif mutation == "prediction":
        evaluation["cases"][0]["prediction"]["value"] = "invented"
    elif mutation == "summary":
        evaluation["summary"]["success_rate"] = 0.0
    elif mutation == "duplicate":
        evaluation["cases"][1] = copy.deepcopy(evaluation["cases"][0])
    elif mutation == "observation":
        evaluation["cases"][0]["observation"]["budget"] += 1
    else:
        evaluation["dataset_sha256"] = "0" * 64
    paths["evaluation"].write_text(json.dumps(evaluation))
    with pytest.raises(ValueError):
        run(paths)


@pytest.mark.parametrize("change", ["dev", "missing_variant", "missing_group"])
def test_only_the_complete_protocol_test_split_is_accepted(files, change):
    paths, rows, evaluation, _ = files
    if change == "dev":
        for row in rows:
            row["split"] = "dev"
    elif change == "missing_variant":
        rows.pop()
    else:
        group = rows[-1]["group_id"]
        rows = [row for row in rows if row["group_id"] != group]
    write_jsonl(paths["data"], rows)
    evaluation["dataset_sha256"] = digest(rows)
    evaluation["cases"] = [score(row, canonical(row["target"]), 1.0) for row in rows]
    evaluation["summary"] = summarize(evaluation["cases"])
    paths["evaluation"].write_text(json.dumps(evaluation))
    with pytest.raises(ValueError):
        run(paths)


def test_unknown_protocol_gate_is_not_silently_ignored(files):
    paths, _, _, protocol = files
    protocol["gate"]["invented_success_min"] = 0.1
    paths["protocol"].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Unknown protocol gate"):
        run(paths)
