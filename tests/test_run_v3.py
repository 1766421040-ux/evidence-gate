"""Confirmation cannot launch model inference before dev and identity checks."""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_gate.training import _json_bytes, normalize_config, sha256_file


spec = importlib.util.spec_from_file_location("run_v3", Path(__file__).resolve().parents[1] / "scripts/run_v3.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ATTEMPT_DIRECTORY", tmp_path / "local-attempts")
    paths = {}
    for name in ("selection", "registry", "dev_data", "test_data"):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text("{}\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"data_sha256": {"test": sha256_file(paths["test_data"])},
                                   "generator_code_sha256": {"src/evidence_gate/evaluation.py": sha256_file(runner.REPOSITORY / "src/evidence_gate/evaluation.py")},
                                   "inference": {"device": "cpu", "dtype": "float32", "batch_size": 4, "max_new_tokens": 72}}))
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "dev-eval.json").write_text("{}\n")
    calls = []

    def run_cli(*arguments):
        calls.append(arguments)
        output = Path(arguments[arguments.index("--output") + 1])
        output.write_text("{}\n")

    module = SimpleNamespace(
        require_test_authorization=lambda *args: {"id": "candidate", "step": 200},
        audit_test=lambda *args: {"status": "fail", "passed": False},
    )
    monkeypatch.setattr(runner, "selection_module", lambda: module)
    monkeypatch.setattr(runner, "verify_freeze", lambda path: None)
    monkeypatch.setattr(runner, "selected_training_record", lambda *args: {"name": "training", "history": []})
    monkeypatch.setattr(runner, "run_cli", run_cli)
    kwargs = {**paths, "protocol": protocol, "model": model, "adapter": tmp_path / "checkpoint",
              "training_run": tmp_path / "training", "evaluations": {"candidate": tmp_path / "dev-eval.json"},
              "output": tmp_path / "confirmation"}
    return kwargs, calls, module


def test_no_eligible_dev_candidate_never_touches_test_or_runs_model(confirmation):
    kwargs, calls, module = confirmation
    kwargs["test_data"].unlink()  # Even reading test would now fail.
    def refuse(*args):
        raise ValueError("No eligible candidate")
    module.require_test_authorization = refuse
    with pytest.raises(ValueError, match="No eligible"):
        runner.confirm(**kwargs)
    assert not calls
    assert not kwargs["output"].exists()


def test_changed_test_bytes_block_before_model_call(confirmation):
    kwargs, calls, _ = confirmation
    kwargs["test_data"].write_text("changed")
    with pytest.raises(ValueError, match="Test bytes differ"):
        runner.confirm(**kwargs)
    assert not calls
    assert not kwargs["output"].exists()


def test_existing_confirmation_is_preserved(confirmation):
    kwargs, calls, _ = confirmation
    kwargs["output"].mkdir()
    existing = kwargs["output"] / "evidence.json"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.confirm(**kwargs)
    assert not calls
    assert existing.read_text() == "preserve"


def test_changed_scoring_code_hash_blocks_test(confirmation):
    kwargs, calls, _ = confirmation
    protocol = json.loads(kwargs["protocol"].read_text())
    protocol["generator_code_sha256"]["src/evidence_gate/evaluation.py"] = "0" * 64
    kwargs["protocol"].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Frozen source changed"):
        runner.confirm(**kwargs)
    assert not calls
    assert not kwargs["output"].exists()


def test_honest_failed_confirmation_is_saved_with_references(confirmation):
    kwargs, calls, _ = confirmation
    result = runner.confirm(**kwargs, reference_adapter="frozen-v2")
    assert result["completed"] and result["quality_gate_status"] == "fail"
    assert result["test_used_for_selection"] is False
    assert [call[call.index("--name") + 1] for call in calls if call[0] == "evaluate"] == ["selected-v3", "contract", "base", "v2-evidence"]
    assert (kwargs["output"] / "confirmation-start.json").exists()
    assert json.loads((kwargs["output"] / "quality-gate.json").read_text())["passed"] is False


def test_new_output_directory_cannot_repeat_same_holdout(confirmation):
    kwargs, calls, _ = confirmation
    runner.confirm(**kwargs)
    first_calls = len(calls)
    kwargs["output"] = kwargs["output"].with_name("another-confirmation")
    with pytest.raises(FileExistsError, match="already has a confirmation"):
        runner.confirm(**kwargs)
    assert len(calls) == first_calls
    assert not kwargs["output"].exists()


def test_mid_run_protocol_change_prevents_completed_receipt(confirmation):
    kwargs, calls, module = confirmation
    def audit(*args):
        kwargs["protocol"].write_text("{}\n")
        return {"status": "pass", "passed": True}
    module.audit_test = audit
    with pytest.raises(ValueError, match="Source changed.*protocol"):
        runner.confirm(**kwargs)
    assert calls
    assert not (kwargs["output"] / "confirmation.json").exists()


def test_mid_run_dev_generation_change_prevents_completed_receipt(confirmation):
    kwargs, calls, module = confirmation
    def audit(*args):
        Path(kwargs["evaluations"]["candidate"]).write_text("changed")
        return {"status": "pass", "passed": True}
    module.audit_test = audit
    with pytest.raises(ValueError, match="Dev evaluation changed"):
        runner.confirm(**kwargs)
    assert calls
    assert not (kwargs["output"] / "confirmation.json").exists()


@pytest.mark.parametrize("values", [["missing"], ["=path"], ["id="], ["id=a", "id=b"]])
def test_candidate_mapping_rejects_ambiguous_input(values):
    with pytest.raises(ValueError):
        runner.parse_evaluations(values)


def test_original_freeze_detects_coordinated_protocol_and_data_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPOSITORY", tmp_path)
    protocol = tmp_path / "protocol.json"
    data = tmp_path / "data.jsonl"
    data.write_text("original")
    protocol.write_text(json.dumps({"data_sha256": sha256_file(data)}))
    freeze = {"format": "evidence-gate-v3-freeze-v1", "files": {
        "protocol.json": sha256_file(protocol), "data.jsonl": sha256_file(data)}}
    (tmp_path / "freeze.json").write_text(json.dumps(freeze))
    runner.verify_freeze(protocol)
    data.write_text("changed")
    protocol.write_text(json.dumps({"data_sha256": sha256_file(data)}))
    with pytest.raises(ValueError, match="Original freeze mismatch"):
        runner.verify_freeze(protocol)


@pytest.fixture
def selected_run(tmp_path):
    root = tmp_path / "training"
    adapter = root / "checkpoints/step-000001"
    adapter.mkdir(parents=True)
    config = normalize_config({"max_steps": 2, "save_steps": 1})
    binding = {"data_sha256": "a" * 64, "model_sha256": "b" * 64,
               "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest()}
    history = [{"step": 1, "loss": .2, "elapsed_s": 1.0}, {"step": 2, "loss": .1, "elapsed_s": 2.0}]
    for name in ("adapter_model.safetensors", "adapter_config.json", "training_state.safetensors"):
        (adapter / name).write_text(name)
    checkpoint = {"format": "evidence-gate-checkpoint-v1", "step": 1, "binding": binding,
                  "history": history[:1], "files": {path.name: sha256_file(path) for path in adapter.iterdir()}}
    (adapter / "checkpoint.json").write_text(json.dumps(checkpoint))
    (root / "run_manifest.json").write_text(json.dumps({"model_name": config["model_name"], "config": config, "binding": binding}))
    (root / "summary.json").write_text(json.dumps({"binding": binding, "optimizer_steps": 2,
        "first_loss": .2, "last_loss": .1, "elapsed_s": 2.0, "trainable_parameters": 10, "total_parameters": 100}))
    (root / "metrics.json").write_text(json.dumps(history))
    selected = {"id": "round-step1", "step": 1,
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
                "checkpoint_sha256": sha256_file(adapter / "checkpoint.json")}
    return root, adapter, selected


def test_selected_training_history_reports_prefix_and_full_cost_separately(selected_run):
    result = runner.selected_training_record(*selected_run)
    assert result["metadata"]["optimizer_steps"] == 1
    assert result["metadata"]["completed_training_steps"] == 2
    assert result["metadata"]["elapsed_s"] == 1
    assert result["metadata"]["total_training_elapsed_s"] == 2
    assert len(result["history"]) == 1


def test_selected_training_budget_cannot_be_changed_in_manifest_and_summary(selected_run):
    root, _, _ = selected_run
    for name in ("run_manifest.json", "summary.json"):
        path = root / name
        value = json.loads(path.read_text())
        if name == "run_manifest.json":
            value["config"]["max_steps"] = 3
        else:
            value["optimizer_steps"] = 3
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="configuration differs"):
        runner.selected_training_record(*selected_run)


def test_selected_training_requires_full_actual_history(selected_run):
    root, _, _ = selected_run
    path = root / "metrics.json"
    path.write_text(json.dumps(json.loads(path.read_text())[:1]))
    with pytest.raises(ValueError, match="Completed history"):
        runner.selected_training_record(*selected_run)
