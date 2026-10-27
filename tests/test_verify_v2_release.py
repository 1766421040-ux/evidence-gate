"""Published V2 selection, gate and report checks use only synthetic CPU fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from evidence_gate.data import canonical, digest, generate_group, write_jsonl
from evidence_gate.evaluation import score, summarize


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_artifacts", ROOT / "scripts/verify_artifacts.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def release(tmp_path):
    # Copy executable audit code, never predictions, weights or result files.
    (tmp_path / "scripts").mkdir()
    for name in ("select_checkpoint", "quality_gate"):
        shutil.copyfile(ROOT / f"scripts/{name}.py", tmp_path / f"scripts/{name}.py")
    folder = tmp_path / "experiments/v2"
    adapter = tmp_path / "artifacts/evidence-v2-lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"CPU-only digest fixture; not model tensors")
    save(adapter / "adapter_config.json", {"r": 8})
    weights_sha, config_sha = (verifier.sha(adapter / name)
                               for name in ("adapter_model.safetensors", "adapter_config.json"))
    binding = {"model_sha256": "fixture-base", "data_sha256": "fixture-train", "config_sha256": "fixture-config"}
    history = [{"step": step, "loss": 1 / step, "elapsed_s": float(step)} for step in range(1, 201)]
    checkpoint = {"format": "evidence-gate-checkpoint-v1", "step": 200, "binding": binding,
                  "files": {"adapter_model.safetensors": weights_sha, "adapter_config.json": config_sha},
                  "history": history}
    save(adapter / "checkpoint.json", checkpoint)
    checkpoint_sha = verifier.sha(adapter / "checkpoint.json")
    dev, test = generate_group(11, "dev", 51), generate_group(12, "test", 51)
    (tmp_path / "data/v2").mkdir(parents=True)
    for split, rows in (("dev", dev), ("test", test)):
        write_jsonl(tmp_path / f"data/v2/{split}.jsonl", rows)
    inference = {"decoding": "greedy free generation", "max_new_tokens": 72,
                 "dtype": "bfloat16", "device": "mps", "batch_size": 4, "output_repair": False}
    provenance = {"base_snapshot_sha256": binding["model_sha256"],
                  "training_data_sha256": binding["data_sha256"],
                  "training_config_sha256": binding["config_sha256"],
                  "adapter_config_sha256": config_sha, "adapter_sha256": weights_sha,
                  "checkpoint_sha256": checkpoint_sha, "checkpoint_step": 200,
                  "adapter_training_binding_verified": True}

    def evaluation(rows, name, correct=True):
        cases = [score(row, canonical(row["target"]) if correct else "invalid", 1.0) for row in rows]
        return {"name": name, "kind": "model", "dataset_sha256": digest(rows),
                "model_provenance": copy.deepcopy(provenance), "inference": inference,
                "cases": cases, "summary": summarize(cases)}

    dev_files = []
    for step in (200, 400, 600):
        result = evaluation(dev, f"step-{step}", correct=step != 600)
        result["model_provenance"]["checkpoint_step"] = step
        if step != 200:
            result["model_provenance"].update(adapter_sha256=f"fixture-weights-{step}",
                                               checkpoint_sha256=f"fixture-checkpoint-{step}")
        path = folder / f"dev-{step}.json"
        save(path, result)
        dev_files.append(path)
    selection = verifier.load_script(tmp_path, "select_checkpoint").select_checkpoint(
        tmp_path / "data/v2/dev.jsonl", dev_files)
    save(folder / "selection.json", selection)
    protocol = read(ROOT / "experiments/v2/protocol.json")
    protocol["test_groups"] = 1
    save(folder / "protocol.json", protocol)
    save(folder / "selected-v2.json", evaluation(test, "selected-v2"))
    metadata = {"optimizer_steps": 200, "selected_step": 200,
                "total_training_budget_steps": 600, "completed_training_steps": 600,
                "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"], "elapsed_s": 200.0,
                "adapter_sha256": weights_sha, "adapter_config_sha256": config_sha,
                "checkpoint_sha256": checkpoint_sha, "binding": binding,
                "selection_split": "dev", "test_used_for_selection": False}
    training = {"name": "selected-v2-step-200", "metadata": metadata, "history": history}
    save(folder / "selected-training.json", training)
    save(tmp_path / "experiments/training/evidence-v2/summary.json", metadata)
    save(tmp_path / "experiments/training/evidence-v2/metrics.json", history)
    refresh_gate_and_report(tmp_path)
    return tmp_path


def refresh_gate_and_report(root):
    """Keep internally valid test evidence while testing a cross-file mismatch."""
    folder = root / "experiments/v2"
    gate = verifier.load_script(root, "quality_gate").audit_gate(
        folder / "selected-v2.json", root / "data/v2/test.jsonl", folder / "protocol.json")
    save(folder / "quality-gate.json", gate)
    save(folder / "report.json", {"methods": [read(folder / "selected-v2.json")],
                                  "training": [read(folder / "selected-training.json")]})


def test_unfinished_release_is_explicitly_pending_and_does_not_read_outputs(tmp_path):
    assert verifier.verify_v2_release(tmp_path) == {"status": "pending", "audit_manifests_present": []}
    path = tmp_path / "experiments/v2/selection.json"
    path.parent.mkdir(parents=True)
    path.write_text("intentionally incomplete while a run is in progress")
    assert verifier.verify_v2_release(tmp_path) == {
        "status": "pending", "audit_manifests_present": ["selection.json"]}


def test_complete_release_recomputes_selection_gate_and_selected_history(release):
    before = {path: path.read_bytes() for path in release.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    assert verifier.verify_v2_release(release) == {
        "status": "verified", "dev_candidates_recomputed": 3, "quality_gate_recomputed": True,
        "quality_gate_status": "pass", "selected_step": 200}
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("change", ["selection", "dev_raw", "gate", "report", "training", "weights", "config", "checkpoint"])
def test_modified_release_evidence_is_rejected(release, change):
    folder = release / "experiments/v2"
    if change == "selection":
        path = folder / "selection.json"
        value = read(path)
        value["candidates"][0]["macro_strict_success"] = 0.9
        save(path, value)
    elif change == "dev_raw":
        path = folder / "dev-200.json"
        value = read(path)
        value["cases"][0]["raw"] = "invalid"
        save(path, value)
        # Even updating its file checksum cannot hide changed dev scoring.
        selection = read(folder / "selection.json")
        selection["candidates"][0]["evaluation_file_sha256"] = verifier.sha(path)
        save(folder / "selection.json", selection)
    elif change == "gate":
        path = folder / "quality-gate.json"
        value = read(path)
        value["passed"] = 1  # JSON type changes must not compare as True.
        save(path, value)
    elif change == "report":
        path = folder / "report.json"
        value = read(path)
        value["methods"][0]["cases"][0]["raw"] = "not the published generation"
        save(path, value)
    elif change == "training":
        training = read(folder / "selected-training.json")
        training["metadata"]["selected_step"] = 600
        save(folder / "selected-training.json", training)
        refresh_gate_and_report(release)
    else:
        name = {"weights": "adapter_model.safetensors", "config": "adapter_config.json", "checkpoint": "checkpoint.json"}[change]
        path = release / "artifacts/evidence-v2-lora" / name
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AssertionError):
        verifier.verify_v2_release(release)


@pytest.mark.parametrize("field", ["adapter_sha256", "adapter_config_sha256", "checkpoint_sha256", "checkpoint_step"])
def test_self_consistent_test_result_must_still_match_dev_selected_identity(release, field):
    path = release / "experiments/v2/selected-v2.json"
    selected = read(path)
    selected["model_provenance"][field] = 600 if field == "checkpoint_step" else "different"
    save(path, selected)
    refresh_gate_and_report(release)
    with pytest.raises(AssertionError, match="differs from dev selection"):
        verifier.verify_v2_release(release)


def test_failed_quality_gate_is_valid_publishable_evidence(release):
    path = release / "experiments/v2/selected-v2.json"
    selected = read(path)
    rows = [json.loads(line) for line in (release / "data/v2/test.jsonl").read_text().splitlines()]
    selected["cases"] = [score(row, "invalid", 1.0) for row in rows]
    selected["summary"] = summarize(selected["cases"])
    save(path, selected)
    refresh_gate_and_report(release)
    assert verifier.verify_v2_release(release)["quality_gate_status"] == "fail"


def test_completed_manifests_do_not_hide_missing_release_files(release):
    (release / "experiments/v2/report.json").unlink()
    with pytest.raises(FileNotFoundError):
        verifier.verify_v2_release(release)
