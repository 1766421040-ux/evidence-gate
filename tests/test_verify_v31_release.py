"""Fabricated CPU digest fixtures exercise audits; these are NOT model results."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from evidence_gate.data import canonical, digest, generate_group, write_jsonl
from evidence_gate.evaluation import score, summarize
from evidence_gate.training import _json_bytes, normalize_config, sha256_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_v31_release", ROOT / "scripts/verify_v31_release.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def refresh_public_manifest(root, folder, adapter=None):
    result = {name: sha256_file(root / folder / name) for name in verify.PUBLIC_RECORDS}
    if adapter:
        result.update({f"adapter/{name}": sha256_file(root / adapter / name) for name in verify.ADAPTER_FILES})
    else:
        result[verify.FULL_CHECKPOINT] = sha256_file(root / folder / verify.FULL_CHECKPOINT)
    save(root / folder / "artifact_manifest.json", result)


@pytest.fixture
def release(tmp_path):
    # Only executable audit code is copied. Every prediction and "weight" below
    # is explicitly manufactured in tmp_path and is never release evidence.
    for relative in ("scripts/selection_v3.py", "scripts/run_v3.py", "src/evidence_gate/training.py", "src/evidence_gate/data.py", "src/evidence_gate/evaluation.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    selector = verify._load_script(tmp_path, "selection_v3")
    config = normalize_config({"max_steps": 2400, "save_steps": 600, "warmup_steps": 60, "device": "cpu", "dtype": "float32"})
    save(tmp_path / "configs/v31.json", config)
    rows, data_paths = {}, {}
    for number, split in enumerate(("train", "dev", "test")):
        rows[split] = generate_group(number + 700, split, 42)
        data_paths[split] = tmp_path / f"data/v31/{split}.jsonl"
        write_jsonl(data_paths[split], rows[split])
    folder, confirmation = tmp_path / verify.FOLDER, tmp_path / verify.FOLDER / "confirmation"
    model_files = {"model.safetensors": "b" * 64, "tokenizer.json": "c" * 64}
    binding = {"data_sha256": sha256_file(data_paths["train"]), "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(), "model_sha256": hashlib.sha256(_json_bytes(model_files)).hexdigest()}
    inference = {"decoding": "greedy free generation", "max_new_tokens": 72, "dtype": "float32", "batch_size": 4, "device": "cpu", "output_repair": False}
    save(tmp_path / "experiments/v2/protocol.json", {"fixture": "Historical reference protocol; not real evidence"})
    protocol = {"format": "evidence-gate-v31-frozen-holdout-protocol-v1", "variants": [row["variant"] for row in rows["dev"]], "train_groups": 1, "dev_groups": 1, "test_groups": 1,
                "gate": {key: value[2] for key, value in selector.GATES.items()},
                "answer_risk_eligibility": {"minimum_emitted_answers_fraction_of_answerable": .8, "rounding": "ceil", "undefined_metrics_pass": False},
                "selection": {"ordering": selector.ORDERING}, "inference": inference,
                "baseline_protocol_sha256": sha256_file(tmp_path / "experiments/v2/protocol.json"),
                "data_sha256": {split: sha256_file(path) for split, path in data_paths.items()},
                "generator_code_sha256": {path: sha256_file(tmp_path / path) for path in ("src/evidence_gate/data.py", "src/evidence_gate/evaluation.py")}}
    save(folder / "protocol.json", protocol)
    freeze_paths = [verify.FOLDER / "protocol.json", *(Path(f"data/v31/{split}.jsonl") for split in rows), *(Path(path) for path in protocol["generator_code_sha256"])]
    save(folder / "freeze.json", {"format": "evidence-gate-v31-freeze-v1", "files": {path.as_posix(): sha256_file(tmp_path / path) for path in freeze_paths}})
    candidates = [{"id": f"fixture-step{step}", "checkpoint_step": step, "base_snapshot_sha256": binding["model_sha256"], "training_data_sha256": binding["data_sha256"], "training_config_sha256": binding["config_sha256"]} for step in verify.STEPS]
    registry = {"format": "evidence-gate-v3-candidate-registry-v1", "registered_at": "2000-01-01T00:00:00Z", "protocol_sha256": sha256_file(folder / "protocol.json"), "config": config,
                "source_config_sha256": sha256_file(tmp_path / "configs/v31.json"), "training_code_sha256": sha256_file(tmp_path / "src/evidence_gate/training.py"), "candidates": candidates}
    save(folder / "registry.json", registry)
    history = [{"step": step, "loss": 1 / step, "elapsed_s": float(step)} for step in range(1, 2401)]
    adapter_config = {"r": config["lora_rank"], "lora_alpha": config["lora_alpha"], "lora_dropout": config["lora_dropout"], "base_model_name_or_path": config["model_name"], "target_modules": config["target_modules"], "peft_type": "LORA", "task_type": "CAUSAL_LM"}

    def adapter(path, step, adapter_binding):
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(b"CPU digest fixture, NOT valid model weights")
        save(path / "adapter_config.json", adapter_config)
        files = {name: sha256_file(path / name) for name in verify.ADAPTER_FILES[:2]}
        save(path / "checkpoint.json", {"format": "evidence-gate-checkpoint-v1", "binding": adapter_binding, "step": step,
                                        "history": history[:step], "files": {**files, "training_state.safetensors": "0" * 64}})
        return {"model_name": config["model_name"], "base_snapshot_sha256": adapter_binding["model_sha256"], "base_files": model_files,
                "training_data_sha256": adapter_binding["data_sha256"], "training_config_sha256": adapter_binding["config_sha256"], "checkpoint_step": step,
                "adapter_training_binding_verified": True, **{key: sha256_file(path / name) for name, key in zip(verify.ADAPTER_FILES, ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256"))}}

    provenance = adapter(tmp_path / verify.ADAPTER, 600, binding)
    (tmp_path / verify.ADAPTER / "LICENSE").write_text("Fixture only; no licensed model represented")
    (tmp_path / verify.ADAPTER / "README.md").write_text("Fixture only; no training or quality claim")
    v2 = adapter(tmp_path / "artifacts/evidence-v2-lora", 600, {**binding, "data_sha256": "2" * 64, "config_sha256": "3" * 64})
    save(tmp_path / "experiments/v2/selection.json", {
        "format": "evidence-gate-dev-selection-v1", "selection_split": "dev", "test_used_for_selection": False,
        "selected_step": 600, "selected_adapter_sha256": v2["adapter_sha256"], "selected_adapter_config_sha256": v2["adapter_config_sha256"], "selected_checkpoint_sha256": v2["checkpoint_sha256"],
        "training_binding": {key: v2[key] for key in ("base_snapshot_sha256", "training_data_sha256", "training_config_sha256", "adapter_config_sha256")},
    })

    def evaluation(members, name, prov, kind="model"):
        cases = [score(row, canonical(row["target"]), 1.0) for row in members]
        value = {"name": name, "kind": kind, "dataset_sha256": digest(members), "cases": cases, "summary": summarize(cases)}
        if kind == "model": value.update(model_provenance=deepcopy(prov), inference={**inference, "adapter_used": name != "base"})
        return value

    evaluations = {}
    for candidate in candidates:
        step = candidate["checkpoint_step"]
        prov = deepcopy(provenance)
        if step != 600:
            prov.update(checkpoint_step=step, adapter_sha256=f"{step:064x}", checkpoint_sha256=f"{step + 1:064x}")
        if step == 2400:
            final_checkpoint = tmp_path / verify.FULL_TRAINING / verify.FULL_CHECKPOINT
            save(final_checkpoint, {"format": "evidence-gate-checkpoint-v1", "binding": binding, "step": step, "history": history,
                                    "files": {"adapter_model.safetensors": prov["adapter_sha256"], "adapter_config.json": prov["adapter_config_sha256"], "training_state.safetensors": "0" * 64}})
            prov["checkpoint_sha256"] = sha256_file(final_checkpoint)
        path = folder / f"dev-step-{step:06d}.json"
        save(path, evaluation(rows["dev"], candidate["id"], prov))
        evaluations[candidate["id"]] = path
    selection = selector.select_candidates(evaluations, data_paths["dev"], folder / "protocol.json", folder / "registry.json")
    save(folder / "selection.json", selection)
    chosen = selection["selected"]
    selected_eval = evaluation(rows["test"], "selected-v3", provenance)
    save(confirmation / "selected-v3.json", selected_eval)
    save(confirmation / "contract.json", evaluation(rows["test"], "contract", None, "rule"))
    base_prov = {"model_name": config["model_name"], "base_snapshot_sha256": binding["model_sha256"], "base_files": model_files, "adapter_sha256": None, "adapter_config_sha256": None}
    save(confirmation / "base.json", evaluation(rows["test"], "base", base_prov))
    save(confirmation / "v2-evidence.json", evaluation(rows["test"], "v2-evidence", v2))
    full_manifest = {"format": "evidence-gate-training-v1", "model_name": config["model_name"], "binding": binding, "config": config, "model_files": model_files}
    full_summary = {"binding": binding, "optimizer_steps": 2400, "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"], "elapsed_s": history[-1]["elapsed_s"], "trainable_parameters": 123, "total_parameters": 456,
                    "adapter_sha256": read(folder / "dev-step-002400.json")["model_provenance"]["adapter_sha256"]}
    metadata = {"format": "evidence-gate-selected-training-summary-v1", "model_name": config["model_name"], "device": config["device"], "dtype": config["dtype"], "seed": config["seed"],
                "optimizer_steps": 600, "selected_step": 600, "completed_training_steps": 2400, "total_training_budget_steps": 2400,
                "first_loss": history[0]["loss"], "last_loss": history[599]["loss"], "elapsed_s": history[599]["elapsed_s"], "total_training_elapsed_s": history[-1]["elapsed_s"],
                "trainable_parameters": 123, "total_parameters": 456, "binding": binding, **{key: chosen[key] for key in ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256")},
                "selection_split": "dev", "test_used_for_selection": False, "training_is_quality_evidence": False}
    training = {"name": "selected-v3-fixture-step600", "history": history[:600], "metadata": metadata}
    save(confirmation / "selected-training.json", training)
    for directory, summary, metrics in ((verify.TRAINING, metadata, history[:600]), (verify.FULL_TRAINING, full_summary, history)):
        save(tmp_path / directory / "summary.json", summary)
        save(tmp_path / directory / "metrics.json", metrics)
        save(tmp_path / directory / "run_manifest.json", full_manifest)
        refresh_public_manifest(tmp_path, directory, verify.ADAPTER if directory == verify.TRAINING else None)
    snapshots = {"selection": sha256_file(folder / "selection.json"), "protocol": sha256_file(folder / "protocol.json"), "registry": sha256_file(folder / "registry.json"), "test_data": sha256_file(data_paths["test"]), "dev_data": sha256_file(data_paths["dev"])}
    receipt = {"selected": chosen, "source_sha256": snapshots, "dev_evaluation_sha256": {name: sha256_file(path) for name, path in evaluations.items()}, "test_used_for_selection": False}
    save(confirmation / "confirmation-start.json", {**receipt, "format": "evidence-gate-v3-confirmation-start-v1", "dev_revalidated": True})
    save(confirmation / "confirmation.json", {**receipt, "format": "evidence-gate-v3-confirmation-v1", "completed": True, "quality_gate_status": "pass", "report": "report.json", "quality_gate": "quality-gate.json"})
    refresh_results(tmp_path)
    return tmp_path


def refresh_results(root):
    selector = verify._load_script(root, "selection_v3")
    folder, confirmation = root / verify.FOLDER, root / verify.FOLDER / "confirmation"
    gate = selector.audit_test(confirmation / "selected-v3.json", root / "data/v31/test.jsonl", folder / "protocol.json", folder / "selection.json")
    save(confirmation / "quality-gate.json", gate)
    complete = read(confirmation / "confirmation.json")
    complete["quality_gate_status"] = gate["status"]
    save(confirmation / "confirmation.json", complete)
    methods = [read(confirmation / name) for name in ("selected-v3.json", "contract.json", "base.json", "v2-evidence.json")]
    save(confirmation / "report.json", {"provenance": {"dataset": methods[0]["dataset_sha256"]}, "methods": methods, "training": [read(confirmation / "selected-training.json")]})


def test_missing_results_are_pending_never_passing(tmp_path):
    result = verify.verify_release(tmp_path)
    assert result["status"] == "pending" and result["publication_allowed"] is False
    assert result["missing"]
    assert verify.main(["--root", str(tmp_path)]) == 2


def test_public_only_complete_fixture_is_verified_without_private_models(release):
    assert not (release / "runs").exists() and not (release / ".cache").exists()
    assert not list(release.rglob("training_state.safetensors"))
    result = verify.verify_release(release)
    assert result["status"] == "verified" and result["publication_allowed"] is True
    assert result["selected_step"] == 600 and result["completed_training_steps"] == 2400
    assert result["dev_candidates_recomputed"] == 4
    assert result["model_execution_performed"] is False


@pytest.mark.parametrize("change", ["selection", "dev_raw", "gate", "report", "training_prefix", "full_budget", "full_config", "weights", "checkpoint", "base_files", "receipt", "freeze", "registry_config"])
def test_changed_public_release_is_rejected(release, change):
    folder, confirmation = release / verify.FOLDER, release / verify.FOLDER / "confirmation"
    if change == "selection":
        path = folder / "selection.json"; value = read(path); value["selected"]["step"] = 2400
    elif change == "dev_raw":
        path = folder / "dev-step-000600.json"; value = read(path); value["cases"][0]["raw"] = "invalid"
    elif change == "gate":
        path = confirmation / "quality-gate.json"; value = read(path); value["passed"] = 1
    elif change == "report":
        path = confirmation / "report.json"; value = read(path); value["methods"][1]["cases"][0]["raw"] = "invalid"
    elif change == "training_prefix":
        path = release / verify.TRAINING / "metrics.json"; value = read(path); value[-1]["loss"] += 1
    elif change == "full_budget":
        path = release / verify.FULL_TRAINING / "metrics.json"; value = read(path)[:-1]
    elif change == "full_config":
        path = release / verify.FULL_TRAINING / "run_manifest.json"; value = read(path); value["config"]["max_steps"] = 600
    elif change in {"weights", "checkpoint"}:
        path = release / verify.ADAPTER / ("adapter_model.safetensors" if change == "weights" else "checkpoint.json")
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError): verify.verify_release(release)
        return
    elif change == "base_files":
        path = confirmation / "base.json"; value = read(path); value["model_provenance"]["base_files"]["model.safetensors"] = "0" * 64
    elif change == "receipt":
        path = confirmation / "confirmation-start.json"; value = read(path); value["dev_revalidated"] = False
    elif change == "freeze":
        path = folder / "protocol.json"; value = read(path); value["note"] = "changed"
    else:
        path = folder / "registry.json"; value = read(path); value["config"]["learning_rate"] *= 2
    save(path, value)
    with pytest.raises(ValueError):
        verify.verify_release(release)


def test_a_complete_receipt_does_not_turn_missing_files_into_pending(release):
    (release / verify.FOLDER / "confirmation/report.json").unlink()
    with pytest.raises(ValueError, match="missing public release"):
        verify.verify_release(release)


@pytest.mark.parametrize("filename,location", [("confirmation-start.json", "experiments/another-attempt"), ("confirmation.json", "experiments/another-attempt"), ("confirmation-start.json", "reports/another-attempt")])
def test_second_published_attempt_on_same_holdout_is_rejected(release, filename, location):
    save(release / location / filename, read(release / verify.FOLDER / "confirmation" / filename))
    with pytest.raises(ValueError, match="More than one published"):
        verify.verify_release(release)


@pytest.mark.parametrize("outside", [False, True])
def test_public_release_never_resolves_private_or_external_symlink(release, outside):
    public = release / verify.ADAPTER / "adapter_model.safetensors"
    hidden = (release.parent / f"outside-{release.name}" if outside else release / "runs/private") / "weights"
    hidden.parent.mkdir(parents=True, exist_ok=True)
    public.rename(hidden)
    public.symlink_to(hidden)
    with pytest.raises(ValueError, match="escapes|symlinks"):
        verify.verify_release(release)


def test_consistent_adapter_replacement_cannot_impersonate_historical_v2(release):
    confirmation = release / verify.FOLDER / "confirmation"
    for filename in verify.ADAPTER_FILES:
        shutil.copyfile(release / verify.ADAPTER / filename, release / "artifacts/evidence-v2-lora" / filename)
    replacement = read(confirmation / "selected-v3.json")
    replacement["name"] = "v2-evidence"
    save(confirmation / "v2-evidence.json", replacement)
    refresh_results(release)
    with pytest.raises(ValueError, match="historical published selection"):
        verify.verify_release(release)


def test_self_consistent_full_log_rewrite_cannot_change_final_checkpoint_history(release):
    # Keep length, elapsed time, boundary losses and the public manifest valid;
    # only the original final checkpoint can detect this interior-history edit.
    path = release / verify.FULL_TRAINING / "metrics.json"
    history = read(path)
    history[1000]["loss"] += 1
    save(path, history)
    refresh_public_manifest(release, verify.FULL_TRAINING)
    with pytest.raises(ValueError, match="final evaluated checkpoint"):
        verify.verify_release(release)


def test_an_honest_failed_confirmation_never_authorizes_quality_publication(release):
    confirmation = release / verify.FOLDER / "confirmation"
    path = confirmation / "selected-v3.json"
    method = read(path)
    rows = [json.loads(line) for line in (release / "data/v31/test.jsonl").read_text().splitlines()]
    method["cases"] = [score(row, "invalid", 1.0) for row in rows]
    method["summary"] = summarize(method["cases"])
    save(path, method)
    refresh_results(release)
    result = verify.verify_release(release)
    assert result["integrity_verified"] is True and result["quality_gate_status"] == "fail"
    assert result["status"] == "failed_quality" and result["publication_allowed"] is False
