"""Mocked orchestration only: no real training, inference or release evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_gate.data import generate_group, write_jsonl
from evidence_gate.training import _json_bytes, _model_manifest, normalize_config, sha256_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_v31_experiment", ROOT / "scripts/run_v31_experiment.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(workflow, "REPOSITORY", root)
    monkeypatch.setattr(workflow, "_runtime_check", lambda: None)
    model = root / "local-model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"Digest-only mock fixture; not a model")
    config = normalize_config({"max_steps": 2400, "save_steps": 600, "device": "mps", "dtype": "bfloat16"})
    write(root / "configs/v31.json", config)
    write(root / "src/evidence_gate/training.py", {"fixture": "not executable training code"})
    folder = root / "experiments/v31"
    data = root / "data/v31"
    for number, split in enumerate(("train", "dev")):
        write_jsonl(data / f"{split}.jsonl", generate_group(number, split, 42))
    # Byte hashes may be checked before eligibility; parsing this sentinel as
    # a dataset would fail. No test data is needed for mocked orchestration.
    (data / "test.jsonl").write_text("TEST CONTENT MUST NOT BE PARSED BY THIS WRAPPER")
    write(root / "experiments/v2/protocol.json", {"fixture": "historical protocol"})
    protocol = {"data_sha256": {split: sha256_file(data / f"{split}.jsonl") for split in ("train", "dev", "test")},
                "inference": workflow.INFERENCE, "generator_code_sha256": {},
                "baseline_protocol_sha256": sha256_file(root / "experiments/v2/protocol.json")}
    write(folder / "protocol.json", protocol)
    binding = {"base_snapshot_sha256": _model_manifest(model)["sha256"], "training_data_sha256": protocol["data_sha256"]["train"],
               "training_config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest()}
    candidates = [{"id": f"reproduction-r8-step{step}", "checkpoint_step": step, **binding} for step in workflow.STEPS]
    historical = {"config": config, "protocol_sha256": sha256_file(folder / "protocol.json"), "source_config_sha256": sha256_file(root / "configs/v31.json"),
                  "training_code_sha256": sha256_file(root / "src/evidence_gate/training.py"), "candidates": candidates}
    write(folder / "registry.json", historical)
    reference = root / "artifacts/evidence-v2-lora"
    reference.mkdir(parents=True)
    (reference / "adapter_model.safetensors").write_bytes(b"Mock reference adapter, not model weights")
    write(reference / "adapter_config.json", {"r": 8})
    ref_binding = {"model_sha256": binding["base_snapshot_sha256"], "data_sha256": "d" * 64, "config_sha256": "c" * 64}
    write(reference / "checkpoint.json", {"format": "evidence-gate-checkpoint-v1", "step": 600, "binding": ref_binding,
                                          "files": {name: sha256_file(reference / name) for name in ("adapter_model.safetensors", "adapter_config.json")}})
    write(root / "experiments/v2/selection.json", {"selected_step": 600,
          **{key: sha256_file(reference / name) for key, name in (("selected_adapter_sha256", "adapter_model.safetensors"), ("selected_adapter_config_sha256", "adapter_config.json"), ("selected_checkpoint_sha256", "checkpoint.json"))},
          "training_binding": {"training_data_sha256": ref_binding["data_sha256"], "training_config_sha256": ref_binding["config_sha256"], "adapter_config_sha256": sha256_file(reference / "adapter_config.json")}})
    calls = []
    state = {"selection": {"status": "no_eligible_candidate", "selected": None, "test_allowed": False}, "confirm_status": "pass"}

    def register_round(**kwargs):
        calls.append(("register", kwargs))
        return deepcopy(historical)

    def cli(*args):
        calls.append(("cli", args))
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        output = Path(args[args.index("--output") + 1])
        if args[0] == "train":
            final = output / "checkpoints/step-002400"
            final.mkdir(parents=True)
            for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint.json"):
                (final / name).write_bytes(b"mock helper input, not real model output")
        else:
            write(output, {"mock_subprocess": True, "name": args[args.index("--name") + 1]})

    def complete(*args):
        calls.append(("completed_training", args))
        return {"metadata": {"completed_training_steps": 2400, "binding": {"model_sha256": binding["base_snapshot_sha256"], "data_sha256": binding["training_data_sha256"], "config_sha256": binding["training_config_sha256"]}}}

    def select(*args):
        calls.append(("select", args))
        return deepcopy(state["selection"])

    def confirm(**kwargs):
        calls.append(("confirm", kwargs))
        return {"quality_gate_status": state["confirm_status"]}

    runner = SimpleNamespace(ATTEMPT_DIRECTORY=root / ".cache/confirmation-attempts", verify_freeze=lambda *args: calls.append(("freeze", args)),
                             verify_frozen_code=lambda *args: calls.append(("code", args)), register_round=register_round, write_json=write,
                             run_cli=cli, selected_training_record=complete, selection_module=lambda: SimpleNamespace(select_candidates=select), confirm=confirm)
    monkeypatch.setattr(workflow, "_runner", lambda: runner)
    return {"root": root, "model": model, "output": root / "runs/new-experiment", "runner": runner, "calls": calls, "state": state, "protocol": protocol}


def execute(e):
    return workflow.run_experiment(e["model"], e["output"])


def test_no_eligible_candidate_retains_four_dev_files_and_never_calls_test(experiment, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "original")
    result = execute(experiment)
    assert result["status"] == "no_eligible_candidate" and result["test_inference_performed"] is False
    assert result["historical_protocol_reproduction"] and result["new_independent_test"] is False
    assert not [call for call in experiment["calls"] if call[0] == "confirm"]
    cli = [call[1] for call in experiment["calls"] if call[0] == "cli"]
    assert [arguments[0] for arguments in cli] == ["train", "evaluate", "evaluate", "evaluate", "evaluate"]
    assert all(arguments[arguments.index("--data") + 1].name == "dev.jsonl" for arguments in cli[1:])
    assert len(list(experiment["output"].glob("dev-step-*.json"))) == 4
    assert os.environ["HF_HUB_OFFLINE"] == "original"
    stages = [name for name, _ in experiment["calls"]]
    assert stages.index("completed_training") < stages.index("select")


@pytest.mark.parametrize("gate,expected", [("pass", "confirmed_pass"), ("fail", "confirmed_fail")])
def test_eligible_selection_delegates_exact_fixed_confirmation(experiment, gate, expected):
    experiment["state"]["selection"] = {"status": "selected", "selected": {"id": "reproduction-r8-step1200", "step": 1200}, "test_allowed": True}
    experiment["state"]["confirm_status"] = gate
    result = execute(experiment)
    assert result["status"] == expected and result["test_inference_performed"] is True
    calls = [args for name, args in experiment["calls"] if name == "confirm"]
    assert len(calls) == 1
    assert calls[0]["adapter"].name == "step-001200"
    assert calls[0]["reference_adapter"] == experiment["root"] / "artifacts/evidence-v2-lora"
    assert calls[0]["test_data"] == experiment["root"] / "data/v31/test.jsonl"
    assert calls[0]["output"] == experiment["output"] / "confirmation"


def test_existing_holdout_ledger_refuses_before_registration_or_training(experiment):
    ledger = experiment["runner"].ATTEMPT_DIRECTORY / f"{experiment['protocol']['data_sha256']['test']}.json"
    write(ledger, {"attempt": "already consumed"})
    with pytest.raises(FileExistsError, match="before training"):
        execute(experiment)
    assert not experiment["output"].exists() and not experiment["calls"]


@pytest.mark.parametrize("kind", ["directory", "file", "model_child", "published_child"])
def test_output_path_cannot_overwrite_or_pollute_inputs(experiment, kind):
    if kind == "directory": experiment["output"].mkdir(parents=True)
    elif kind == "file": write(experiment["output"], {"preserve": True})
    elif kind == "model_child": experiment["output"] = experiment["model"] / "new-output"
    else: experiment["output"] = experiment["root"] / "experiments/forbidden"
    with pytest.raises((ValueError, FileExistsError)):
        execute(experiment)
    assert not experiment["calls"]


@pytest.mark.parametrize("source", ["config", "base", "code", "reference"])
def test_changed_fixed_input_is_rejected_before_any_training(experiment, source):
    paths = {"config": "configs/v31.json", "base": "local-model/model.safetensors", "code": "src/evidence_gate/training.py", "reference": "artifacts/evidence-v2-lora/adapter_model.safetensors"}
    path = experiment["root"] / paths[source]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        execute(experiment)
    assert not [call for call in experiment["calls"] if call[0] in {"register", "cli", "confirm"}]
    assert not experiment["output"].exists()


def test_freeze_failure_is_a_preflight_failure(experiment):
    def reject(*args): raise ValueError("Original freeze mismatch")
    experiment["runner"].verify_freeze = reject
    with pytest.raises(ValueError, match="freeze"):
        execute(experiment)
    assert not experiment["output"].exists()


def test_interrupted_training_preserves_failure_and_never_selects(experiment):
    def fail(*args): raise RuntimeError("mock training failure")
    experiment["runner"].run_cli = fail
    with pytest.raises(RuntimeError, match="mock training failure"):
        execute(experiment)
    failure = json.loads((experiment["output"] / "failure.json").read_text())
    assert failure["stage"] == "training" and failure["status"] == "failed"
    assert not [call for call in experiment["calls"] if call[0] in {"select", "confirm"}]
    assert not (experiment["output"] / "outcome.json").exists()


def test_incomplete_training_cannot_proceed_to_dev(experiment):
    def fail(*args): raise ValueError("Completed history does not cover training budget")
    experiment["runner"].selected_training_record = fail
    with pytest.raises(ValueError, match="Completed history"):
        execute(experiment)
    assert [args[0] for name, args in experiment["calls"] if name == "cli"] == ["train"]
    assert json.loads((experiment["output"] / "failure.json").read_text())["stage"] == "training_verification"


def test_inconsistent_selector_state_never_authorizes_test(experiment):
    experiment["state"]["selection"] = {"status": "no_eligible_candidate", "selected": None, "test_allowed": True}
    with pytest.raises(ValueError, match="test is forbidden"):
        execute(experiment)
    assert not [call for call in experiment["calls"] if call[0] == "confirm"]


def test_reference_change_during_training_stops_before_dev_or_test(experiment):
    original = experiment["runner"].run_cli
    def change_after_training(*args):
        original(*args)
        path = experiment["root"] / "experiments/v2/selection.json"
        value = json.loads(path.read_text())
        value["note"] = "different historical reference"
        path.write_text(json.dumps(value))
    experiment["runner"].run_cli = change_after_training
    with pytest.raises(ValueError, match="V2 reference changed during reproduction"):
        execute(experiment)
    assert [args[0] for name, args in experiment["calls"] if name == "cli"] == ["train"]
    assert not [call for call in experiment["calls"] if call[0] in {"select", "confirm"}]
