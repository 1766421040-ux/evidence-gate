"""Orchestration boundaries, with every subprocess and tokenizer mocked.

These tests never import transformers/torch, launch a child process, or train a
model. Dataset generation, scoring and selection correctness have their own
tests; here the contract is when test inference is permitted to begin.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_v2", ROOT / "scripts/run_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def create_fake_training(output):
    binding = {"model_sha256": "test-base", "data_sha256": "test-data", "config_sha256": "test-config"}
    for step in (200, 400, 600):
        checkpoint = output / "checkpoints" / f"step-{step:06d}"
        checkpoint.mkdir(parents=True)
        for filename in ("adapter_model.safetensors", "adapter_config.json", "training_state.safetensors"):
            (checkpoint / filename).write_bytes(f"test-only fixture {step} {filename}".encode())
        save(checkpoint / "checkpoint.json", {
            "format": "evidence-gate-checkpoint-v1", "step": step, "binding": binding,
            "files": {filename: runner.sha256_file(checkpoint / filename) for filename in
                      ("adapter_model.safetensors", "adapter_config.json", "training_state.safetensors")},
            "history": [{"step": index, "loss": 1 / index, "elapsed_s": float(index)} for index in range(1, step + 1)],
        })
    save(output / "summary.json", {"optimizer_steps": 600, "binding": binding,
                                  "elapsed_s": 600.0, "trainable_parameters": 1, "total_parameters": 2})
    save(output / "run_manifest.json", {"model_name": "test-only fixture", "binding": binding,
                                       "config": {"max_steps": 600, "device": "cpu", "dtype": "bfloat16", "seed": 42}})


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"

    def fake_build(output, source, **kwargs):
        output.mkdir()
        for split in ("train", "dev", "test"):
            (output / f"{split}.jsonl").write_text("fixture; not used by a model\n")
        save(output / "manifest.json", {"generator": "mocked-only-for-orchestration"})

    monkeypatch.setattr(runner, "build_dataset", fake_build)
    monkeypatch.setattr(runner, "tokenizer_preflight", lambda *args: {
        "test_read": False, "max_length_before": 512, "max_length_selected": 512,
    })
    return model, output


class FakeProcesses:
    def __init__(self, output, failure=None, altered_selection=False):
        self.output = output
        self.failure = failure
        self.altered_selection = altered_selection
        self.events = []

    def __call__(self, command, *, check, cwd):
        assert check is True
        assert cwd == ROOT
        command = list(map(str, command))

        def arg(flag):
            return command[command.index(flag) + 1]

        if command[1:3] == ["-m", "evidence_gate.cli"]:
            operation = command[3]
            split = Path(arg("--data")).stem if "--data" in command else None
        else:
            operation = Path(command[1]).stem
            split = None
        event = (operation, split)
        self.events.append(event)
        if self.failure == operation:
            raise subprocess.CalledProcessError(1, command)
        destination = Path(arg("--output"))
        if operation == "train":
            create_fake_training(destination)
        elif operation == "evaluate":
            if split == "test":
                # The gate is a completed file plus verified local hashes, not
                # simply having invoked selection at some earlier point.
                assert (self.output / "selection.json").is_file()
                assert (self.output / "selected-training.json").is_file()
                assert ("select_checkpoint", None) in self.events
            save(destination, {"mock_evaluation": True})
        elif operation == "select_checkpoint":
            assert Path(arg("--data")).name == "dev.jsonl"
            checkpoint = self.output / "training/checkpoints/step-000200"
            save(destination, {
                "selection_split": "dev", "test_used_for_selection": False, "selected_step": 200,
                "selected_adapter_sha256": "changed" if self.altered_selection else runner.sha256_file(checkpoint / "adapter_model.safetensors"),
                "selected_adapter_config_sha256": runner.sha256_file(checkpoint / "adapter_config.json"),
                "selected_checkpoint_sha256": runner.sha256_file(checkpoint / "checkpoint.json"),
            })
        elif operation == "quality_gate":
            assert "--require-pass" not in command
            save(destination, {"status": "fail", "passed": False})
        elif operation == "report":
            save(destination, {"mock_report": True})
        else:
            raise AssertionError(f"Unexpected subprocess: {command}")
        return subprocess.CompletedProcess(command, 0)


def test_test_inference_waits_for_selection_and_history_matches_selected_weights(isolated_runner, monkeypatch):
    model, output = isolated_runner
    fake = FakeProcesses(output)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    result = runner.reproduce(model, output, ROOT / "configs/v2.json", device="cpu")
    selected_at = fake.events.index(("select_checkpoint", None))
    assert all(index > selected_at for index, event in enumerate(fake.events) if event == ("evaluate", "test"))
    assert fake.events.count(("evaluate", "dev")) == 3
    assert fake.events.count(("evaluate", "test")) == 3
    assert (output / "protocol.json").read_bytes() == runner.PROTOCOL.read_bytes()
    training = json.loads((output / "selected-training.json").read_text())
    assert len(training["history"]) == training["metadata"]["selected_step"] == 200
    assert training["history"][-1]["loss"] == training["metadata"]["last_loss"] == 1 / 200
    assert training["metadata"]["total_training_budget_steps"] == 600
    assert training["metadata"]["elapsed_s"] == 200.0
    assert training["metadata"]["total_training_elapsed_s"] == 600.0
    # Quality failure is not a process failure: it still produces the report,
    # then stops instead of starting a further model/data tuning round.
    assert result["completed"] and result["quality_gate_status"] == "fail"
    assert (output / "report.json").exists()
    assert fake.events.count(("train", "train")) == 1


@pytest.mark.parametrize("failure", ["train", "select_checkpoint", "quality_gate"])
def test_failed_stage_stops_without_later_inference_or_report(isolated_runner, monkeypatch, failure):
    model, output = isolated_runner
    fake = FakeProcesses(output, failure=failure)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(subprocess.CalledProcessError):
        runner.reproduce(model, output, ROOT / "configs/v2.json", device="cpu")
    assert fake.events[-1][0] == failure
    assert not (output / "report.json").exists()
    if failure in ("train", "select_checkpoint"):
        assert ("evaluate", "test") not in fake.events


def test_selection_with_wrong_weight_hash_never_unlocks_test(isolated_runner, monkeypatch):
    model, output = isolated_runner
    fake = FakeProcesses(output, altered_selection=True)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(ValueError, match="hashes"):
        runner.reproduce(model, output, ROOT / "configs/v2.json", device="cpu")
    assert ("evaluate", "test") not in fake.events
    assert not (output / "selected-training.json").exists()


def test_existing_output_is_untouched_and_launches_nothing(isolated_runner, monkeypatch):
    model, output = isolated_runner
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user-owned unfinished run")
    fake = FakeProcesses(output)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(FileExistsError, match="Output already exists"):
        runner.reproduce(model, output, ROOT / "configs/v2.json", device="cpu")
    assert sentinel.read_text() == "user-owned unfinished run"
    assert fake.events == []
