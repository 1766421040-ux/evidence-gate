"""CPU-only V3 stage boundaries and strict selection/gate integrity checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import canonical, digest, generate_group, messages_for, write_jsonl
from evidence_gate.evaluation import score, summarize

spec = importlib.util.spec_from_file_location("selection_v3", Path(__file__).resolve().parents[1] / "scripts" / "selection_v3.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)

SEMANTICS = ("supported", "value_changed", "distractor", "reordered", "missing", "expired", "future", "conflict", "ambiguous", "untrusted")
VARIANTS = [f"{semantic}_b{budget}" for semantic in SEMANTICS for budget in (0, 1)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def rows_for(split):
    result = []
    for number in range(2):
        originals = {row["variant"]: row for row in generate_group(number + (100 if split == "dev" else 200), split, 42)}
        group = f"custom-{split}-{number}"
        for semantic in SEMANTICS:
            for budget in (0, 1):
                name = semantic
                if semantic in ("missing", "expired", "conflict", "untrusted"):
                    name += "_search" if budget else "_stop"
                elif semantic == "future":
                    name = "future_search"
                row = deepcopy(originals[name])
                row.update(id=f"{group}:{semantic}_b{budget}", group_id=group, variant=f"{semantic}_b{budget}")
                row["observation"]["budget"] = budget
                row["messages"] = messages_for(row["observation"])
                if semantic in ("missing", "expired", "future", "conflict", "untrusted"):
                    row["target"] = {"action": "search" if budget else "abstain", "value": "", "citations": []}
                result.append(row)
    return result


@pytest.fixture
def experiment(tmp_path):
    data = {}
    rows = {}
    for split in ("dev", "test"):
        rows[split] = rows_for(split)
        data[split] = tmp_path / f"{split}.jsonl"
        write_jsonl(data[split], rows[split])
    protocol = {
        "format": "evidence-gate-v3-protocol-v1", "variants": VARIANTS,
        "dev_groups": 2, "test_groups": 2,
        "data_sha256": {split: selection._sha(path) for split, path in data.items()},
        "gate": {name: item[2] for name, item in selection.GATES.items()},
        "answer_risk_eligibility": {"minimum_emitted_answers_fraction_of_answerable": .8, "rounding": "ceil", "undefined_metrics_pass": False},
        "selection": {"ordering": selection.ORDERING},
        "inference": {"decoding": "greedy free generation", "max_new_tokens": 72, "dtype": "bfloat16", "batch_size": 4, "device": "mps", "output_repair": False},
    }
    protocol_path = tmp_path / "protocol.json"
    write(protocol_path, protocol)
    candidates, adapters, provenance = [], {}, {}
    for index in range(2):
        candidate = {"id": f"candidate-{index}", "checkpoint_step": 200 * (index + 1), "base_snapshot_sha256": "a" * 64,
                     "training_data_sha256": "b" * 64, "training_config_sha256": str(index + 1) * 64}
        candidates.append(candidate)
        adapter = tmp_path / candidate["id"]
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(f"digest-only-test-weights-{index}".encode())
        write(adapter / "adapter_config.json", {"r": 8, "alpha": 16})
        hashes = {name: selection._sha(adapter / name) for name in ("adapter_model.safetensors", "adapter_config.json")}
        write(adapter / "checkpoint.json", {"format": "evidence-gate-checkpoint-v1", "step": candidate["checkpoint_step"],
                                             "binding": {"model_sha256": candidate["base_snapshot_sha256"], "data_sha256": candidate["training_data_sha256"], "config_sha256": candidate["training_config_sha256"]}, "files": hashes})
        provenance[candidate["id"]] = {**{key: candidate[key] for key in selection.BINDINGS}, "checkpoint_step": candidate["checkpoint_step"],
                                       "adapter_sha256": hashes["adapter_model.safetensors"], "adapter_config_sha256": hashes["adapter_config.json"],
                                       "checkpoint_sha256": selection._sha(adapter / "checkpoint.json"), "adapter_training_binding_verified": True}
        adapters[candidate["id"]] = adapter
    registry_path = tmp_path / "registry.json"
    write(registry_path, {"format": "evidence-gate-v3-candidate-registry-v1", "candidates": candidates})

    def evaluation(candidate_id, split="dev", change=None, suffix=""):
        cases = []
        for row in rows[split]:
            target = deepcopy(row["target"])
            raw = change(row, target) if change else canonical(target)
            cases.append(score(row, raw, 1.25))
        result = {"name": candidate_id, "kind": "model", "dataset_sha256": digest(rows[split]), "cases": cases, "summary": summarize(cases),
                  "model_provenance": provenance[candidate_id], "inference": {**protocol["inference"], "adapter_used": True}}
        path = tmp_path / f"{candidate_id}-{split}{suffix}.json"
        write(path, result)
        return path
    evaluations = {candidate["id"]: evaluation(candidate["id"]) for candidate in candidates}
    return {"root": tmp_path, "rows": rows, "data": data, "protocol": protocol_path, "registry": registry_path,
            "evaluations": evaluations, "adapters": adapters, "evaluate": evaluation}


def choose(e):
    return selection.select_candidates(e["evaluations"], e["data"]["dev"], e["protocol"], e["registry"])


def save_choice(e):
    path = e["root"] / "selection.json"
    write(path, choose(e))
    return path


def test_explicit_twenty_variants_and_exact_tie_use_registration_order(experiment):
    result = choose(experiment)
    assert result["status"] == "selected" and result["test_allowed"] is True
    assert result["selected"]["id"] == "candidate-0"
    assert result["eligible_candidate_ids"] == ["candidate-0", "candidate-1"]
    assert all(len(candidate["audit"]["checks"]) == 7 for candidate in result["candidate_audits"])
    assert result["candidate_audits"][0]["audit"]["answer_risk_eligibility"] == {"required_emitted_answers": 13, "actual_emitted_answers": 16}


def test_registered_protocol_binding_is_checked_before_dev_scoring(experiment):
    e = experiment
    registry = json.loads(e["registry"].read_text())
    registry["protocol_sha256"] = selection._sha(e["protocol"])
    write(e["registry"], registry)
    assert choose(e)["status"] == "selected"
    registry["protocol_sha256"] = "f" * 64
    write(e["registry"], registry)
    e["data"]["dev"].unlink()  # The mismatch must fail before opening model evidence.
    with pytest.raises(ValueError, match="bound to a different protocol"):
        choose(e)


def test_higher_average_cannot_bypass_answerable_gate(experiment):
    e = experiment
    wrong_ids = {row["id"] for row in e["rows"]["dev"] if row["target"]["action"] == "answer" and row["group_id"] == "custom-dev-0" and row["variant"].startswith(("supported", "value_changed"))}
    e["evaluations"]["candidate-0"] = e["evaluate"]("candidate-0", change=lambda row, target: canonical({"action": "abstain", "value": "", "citations": []} if row["id"] in wrong_ids else target))
    result = choose(e)
    rejected = result["candidate_audits"][0]["audit"]
    assert rejected["recomputed_summary"]["success_rate"] == .9
    assert rejected["recomputed_summary"]["answerable_success"] == .75
    assert result["selected"]["id"] == "candidate-1"


@pytest.mark.parametrize("wrong_field", ["value", "citations"])
def test_unsupported_answer_gate_is_distinct_from_over_answer(experiment, wrong_field):
    e = experiment
    chosen_id = next(row["id"] for row in e["rows"]["dev"] if row["target"]["action"] == "answer")
    def change(row, target):
        if row["id"] == chosen_id:
            target[wrong_field] = "999 days" if wrong_field == "value" else ["invisible-doc"]
        return canonical(target)
    e["evaluations"]["candidate-0"] = e["evaluate"]("candidate-0", change=change)
    result = choose(e)
    checks = {check["metric"]: check for check in result["candidate_audits"][0]["audit"]["checks"]}
    assert checks["over_answer_rate"]["passed"] is True
    assert checks["unsupported_answer_rate"]["actual"] == 1 / 16
    assert checks["unsupported_answer_rate"]["passed"] is False
    assert result["selected"]["id"] == "candidate-1"


@pytest.mark.parametrize("raw", ['{"action":"abstain","value":"","citations":[]}', "invalid JSON"])
def test_no_answer_never_claims_safety_or_authorizes_test(experiment, raw):
    e = experiment
    for name in e["evaluations"]:
        e["evaluations"][name] = e["evaluate"](name, change=lambda row, target: raw)
    result = choose(e)
    assert result["status"] == "no_eligible_candidate" and result["selected"] is None and result["test_allowed"] is False
    assert all(not check["eligible"] for check in result["candidate_audits"][0]["audit"]["checks"] if check["metric"] in ("over_answer_rate", "unsupported_answer_rate"))
    path = save_choice(e)
    # Nonexistent test files would raise FileNotFoundError if touched first.
    with pytest.raises(ValueError, match="Test is forbidden"):
        selection.audit_test("unopened-result", "unopened-test", "unopened-protocol", path)
    with pytest.raises(ValueError, match="Test is forbidden"):
        selection.require_test_authorization(path, "unopened-protocol", "unopened-registry", "unopened-adapter", "unopened-dev", {})


@pytest.mark.parametrize("change,match", [
    ("summary", "Stored summary"), ("case_score", "Stored case score"), ("raw", "Stored case score"),
    ("observation", "altered observation"), ("expected", "altered expected"), ("duplicate", "duplicated"),
    ("rule", "Only original model"), ("dtype", "inference contract"), ("binding", "preregistered"),
])
def test_altered_evaluation_evidence_is_not_a_quality_failure(experiment, change, match):
    e = experiment
    path = e["evaluations"]["candidate-0"]
    value = json.loads(path.read_text())
    if change == "summary": value["summary"]["success_rate"] = .1
    elif change == "case_score": value["cases"][0]["metrics"]["success"] = False
    elif change == "raw": value["cases"][0]["raw"] = "changed"
    elif change == "observation": value["cases"][0]["observation"]["budget"] = 99
    elif change == "expected": value["cases"][0]["expected"]["value"] = "999 days"
    elif change == "duplicate": value["cases"][-1] = value["cases"][0]
    elif change == "rule": value["kind"] = "rule"
    elif change == "dtype": value["inference"]["dtype"] = "float32"
    elif change == "binding": value["model_provenance"]["training_config_sha256"] = "f" * 64
    write(path, value)
    with pytest.raises(ValueError, match=match):
        choose(e)


def test_incomplete_frozen_variants_and_lowered_thresholds_rejected(experiment):
    e = experiment
    rows = e["rows"]["dev"][:-1]
    write_jsonl(e["data"]["dev"], rows)
    protocol = json.loads(e["protocol"].read_text())
    protocol["data_sha256"]["dev"] = selection._sha(e["data"]["dev"])
    write(e["protocol"], protocol)
    with pytest.raises(ValueError, match="protocol variants"):
        choose(e)
    protocol["gate"]["answerable_success_min"] = .5
    write(e["protocol"], protocol)
    with pytest.raises(ValueError, match="cannot lower"):
        choose(e)


def test_authorization_and_actual_test_reaudit_match_selected_artifact(experiment):
    e = experiment
    path = save_choice(e)
    chosen = selection.require_test_authorization(path, e["protocol"], e["registry"], e["adapters"]["candidate-0"], e["data"]["dev"], e["evaluations"])
    assert chosen == json.loads(path.read_text())["selected"]
    test_eval = e["evaluate"]("candidate-0", split="test")
    result = selection.audit_test(test_eval, e["data"]["test"], e["protocol"], path)
    assert result["status"] == "pass" and result["confirmation_status"] == "confirmed_pass"
    assert result["integrity"]["test_rows"] == 40
    other = e["evaluate"]("candidate-1", split="test")
    with pytest.raises(ValueError, match="preregistered|not the selected"):
        selection.audit_test(other, e["data"]["test"], e["protocol"], path)


@pytest.mark.parametrize("filename", ["adapter_model.safetensors", "adapter_config.json", "checkpoint.json"])
def test_changed_artifact_cannot_cross_test_boundary(experiment, filename):
    e = experiment
    path = save_choice(e)
    adapter = e["adapters"]["candidate-0"]
    changed = adapter / filename
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum changed"):
        selection.require_test_authorization(path, e["protocol"], e["registry"], adapter, e["data"]["dev"], e["evaluations"])


@pytest.mark.parametrize("source", ["protocol", "registry", "raw_dev"])
def test_changed_protocol_registry_or_raw_dev_invalidates_saved_selection(experiment, source):
    e = experiment
    saved = save_choice(e)
    path = e[source] if source != "raw_dev" else e["evaluations"]["candidate-0"]
    value = json.loads(path.read_text())
    value["new_note"] = "modified after selection"
    write(path, value)
    with pytest.raises(ValueError, match="Saved selection differs"):
        selection.require_test_authorization(saved, e["protocol"], e["registry"], e["adapters"]["candidate-0"], e["data"]["dev"], e["evaluations"])


def test_final_failure_is_preserved(experiment):
    e = experiment
    saved = save_choice(e)
    path = e["evaluate"]("candidate-0", split="test", change=lambda row, target: "invalid")
    result = selection.audit_test(path, e["data"]["test"], e["protocol"], saved)
    assert result["status"] == "fail" and result["confirmation_status"] == "confirmed_fail"
    assert result["passed"] is False and result["integrity"]["verified"] is True
