"""Independently audit the public V3.1 release using CPU and published files.

No model, private runs directory, optimizer tensors, network or ML dependency is
required. Missing release evidence is pending (CLI exit 2), never a pass. A
complete but failed confirmation remains auditable and cannot authorize a
quality-qualified publication. This verifies recorded evidence, not undisclosed
experiments or a new execution of the model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from evidence_gate.data import audit, canonical, digest, read_jsonl
from evidence_gate.evaluation import score, summarize
from evidence_gate.training import _json_bytes, normalize_config, sha256_file

STEPS = (600, 1200, 1800, 2400)
FOLDER = Path("experiments/v31")
ADAPTER = Path("artifacts/evidence-v31-lora")
TRAINING = Path("experiments/training/evidence-v31")
FULL_TRAINING = Path("experiments/training/evidence-v31-full")
PUBLIC_RECORDS = ("summary.json", "metrics.json", "run_manifest.json")
FULL_CHECKPOINT = "checkpoint-step-002400.json"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "checkpoint.json")
INFERENCE = ("decoding", "max_new_tokens", "dtype", "batch_size", "device", "output_repair")
PRIVATE_DIRECTORIES = {".git", ".cache", "runs", ".venv", "node_modules", "__pycache__"}


def _public_file(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Release evidence must be inside the public repository") from error
    if any(part in PRIVATE_DIRECTORIES or part == ".." for part in relative.parts):
        raise ValueError("Release evidence cannot depend on private run/cache directories")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Release evidence escapes the public repository")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Release evidence must be actual public files, not symlinks")
    return path


def _receipt_paths(root, name):
    # Public evidence can live outside experiments/. Private runtime directories
    # and Git internals are deliberately outside the published audit scope.
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [folder for folder in folders if folder not in PRIVATE_DIRECTORIES]
        if name in files:
            yield _public_file(root, Path(directory) / name)


def _load_script(root, name):
    spec = importlib.util.spec_from_file_location(f"v31_release_{name}", root / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same(actual, expected, message):
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def _required():
    result = [Path(f"scripts/{name}.py") for name in ("selection_v3", "run_v3")]
    result += [Path("src/evidence_gate/training.py"), Path("configs/v31.json")]
    result += [Path(f"data/v31/{split}.jsonl") for split in ("train", "dev", "test")]
    result += [FOLDER / name for name in ("protocol.json", "freeze.json", "registry.json", "selection.json")]
    result += [FOLDER / f"dev-step-{step:06d}.json" for step in STEPS]
    result += [ADAPTER / name for name in (*ADAPTER_FILES, "LICENSE", "README.md")]
    result += [directory / name for directory in (TRAINING, FULL_TRAINING) for name in (*PUBLIC_RECORDS, "artifact_manifest.json")]
    result.append(FULL_TRAINING / FULL_CHECKPOINT)
    result += [FOLDER / "confirmation" / name for name in ("selected-v3.json", "quality-gate.json", "report.json", "selected-training.json", "confirmation-start.json", "confirmation.json", "base.json", "v2-evidence.json", "contract.json")]
    result += [Path("artifacts/evidence-v2-lora") / name for name in ADAPTER_FILES]
    result += [Path("experiments/v2") / name for name in ("selection.json", "protocol.json")]
    return result


def _history(history, steps):
    _same([item.get("step") for item in history], list(range(1, steps + 1)), "Training history is incomplete or has altered step types")
    previous = 0.0
    for item in history:
        for key in ("loss", "elapsed_s"):
            value = item.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Training losses and elapsed times must be finite and nonnegative")
        if item["elapsed_s"] < previous:
            raise ValueError("Training elapsed times move backwards")
        previous = item["elapsed_s"]


def _public_manifest(root, folder, adapter, read):
    recorded = read(root / folder / "artifact_manifest.json")
    expected = {name: sha256_file(root / folder / name) for name in PUBLIC_RECORDS}
    if adapter:
        expected.update({f"adapter/{name}": sha256_file(root / adapter / name) for name in ADAPTER_FILES})
    else:
        expected[FULL_CHECKPOINT] = sha256_file(root / folder / FULL_CHECKPOINT)
    _same(recorded, expected, "Public training artifact manifest differs from the published records")


def _raw_method(rows, method):
    if method.get("dataset_sha256") != digest(rows):
        raise ValueError("Confirmation method uses a different dataset")
    lookup = {row["id"]: row for row in rows}
    cases = method.get("cases", [])
    ids = [case.get("id") for case in cases]
    if len(ids) != len(rows) or len(set(ids)) != len(rows) or set(ids) != set(lookup):
        raise ValueError("Confirmation method must cover every test row exactly once")
    rebuilt = []
    for case in cases:
        latency = case.get("latency_ms")
        if not isinstance(case.get("raw"), str) or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Raw generation and finite nonnegative latency are required")
        actual = score(lookup[case["id"]], case["raw"], latency)
        for key in actual:
            _same(case.get(key), actual[key], f"Confirmation raw score or source identity changed: {key}")
        rebuilt.append(actual)
    _same(method.get("summary"), summarize(rebuilt), "Confirmation summary differs from raw scoring")


def _adapter_binding(root, folder, provenance, read):
    for name, key in zip(ADAPTER_FILES, ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256")):
        _same(provenance.get(key), sha256_file(root / folder / name), "Published adapter differs from evaluation identity")
    checkpoint = read(root / folder / "checkpoint.json")
    expected = {"model_sha256": provenance["base_snapshot_sha256"], "data_sha256": provenance["training_data_sha256"], "config_sha256": provenance["training_config_sha256"]}
    _same(checkpoint.get("binding"), expected, "Published checkpoint binding differs from evaluation")
    _same(checkpoint.get("step"), provenance.get("checkpoint_step"), "Published checkpoint step differs from evaluation")
    if checkpoint.get("format") != "evidence-gate-checkpoint-v1" or provenance.get("adapter_training_binding_verified") is not True:
        raise ValueError("Checkpoint format or training provenance is unverified")
    for name in ADAPTER_FILES[:2]:
        _same(checkpoint["files"].get(name), sha256_file(root / folder / name), "Checkpoint's published file manifest changed")
    return checkpoint


def verify_release(root):
    root = Path(root).resolve()
    missing = [path.as_posix() for path in _required() if not _public_file(root, path).is_file()]
    if missing:
        if (root / FOLDER / "confirmation/confirmation.json").is_file():
            raise ValueError(f"Completed confirmation has missing public release files: {missing}")
        return {"status": "pending", "publication_allowed": False, "missing": missing}
    selector = _load_script(root, "selection_v3")
    runner = _load_script(root, "run_v3")
    runner.REPOSITORY = root
    read = lambda path: selector._json(_public_file(root, path))[0]
    folder, confirmation = root / FOLDER, root / FOLDER / "confirmation"
    protocol_path, registry_path = folder / "protocol.json", folder / "registry.json"
    protocol, registry = read(protocol_path), read(registry_path)
    for relative in read(folder / "freeze.json")["files"]:
        _public_file(root, relative)
    for relative in protocol["generator_code_sha256"]:
        _public_file(root, relative)
    runner.verify_freeze(protocol_path)
    runner.verify_frozen_code(protocol)
    _same(protocol.get("baseline_protocol_sha256"), sha256_file(root / "experiments/v2/protocol.json"), "Historical V2 baseline protocol differs from the frozen reference")
    _same(registry.get("protocol_sha256"), sha256_file(protocol_path), "Registry is not bound to the frozen protocol")
    _same(registry.get("training_code_sha256"), sha256_file(root / "src/evidence_gate/training.py"), "Registered training code changed")
    _same(registry.get("source_config_sha256"), sha256_file(root / "configs/v31.json"), "Registered source configuration changed")
    settings = normalize_config(root / "configs/v31.json")
    _same(registry.get("config"), settings, "Registry configuration differs from normalized source configuration")
    _same(settings["max_steps"], 2400, "Release requires the registered complete 2400-step budget")
    _same([candidate["checkpoint_step"] for candidate in registry["candidates"]], list(STEPS), "Registry must preserve all four candidates and their registration order")
    paths = {split: root / f"data/v31/{split}.jsonl" for split in ("train", "dev", "test")}
    for split, path in paths.items():
        _same(sha256_file(path), protocol["data_sha256"][split], "Released data differs from the frozen protocol")
    evaluations = {candidate["id"]: folder / f"dev-step-{candidate['checkpoint_step']:06d}.json" for candidate in registry["candidates"]}
    chosen = selector.require_test_authorization(folder / "selection.json", protocol_path, registry_path, root / ADAPTER, paths["dev"], evaluations)
    gate = selector.audit_test(confirmation / "selected-v3.json", paths["test"], protocol_path, folder / "selection.json")
    _same(read(confirmation / "quality-gate.json"), gate, "Stored final quality gate differs from complete raw-output re-audit")

    full_manifest = read(root / FULL_TRAINING / "run_manifest.json")
    full_summary = read(root / FULL_TRAINING / "summary.json")
    full_history = read(root / FULL_TRAINING / "metrics.json")
    _same(full_manifest.get("config"), settings, "Full training manifest differs from registered configuration")
    expected_binding = {"data_sha256": sha256_file(paths["train"]), "config_sha256": hashlib.sha256(_json_bytes(settings)).hexdigest(), "model_sha256": hashlib.sha256(_json_bytes(full_manifest["model_files"])).hexdigest()}
    for binding in (full_manifest.get("binding"), full_summary.get("binding")):
        _same(binding, expected_binding, "Full training binding differs from data/config/base manifest")
    for candidate in registry["candidates"]:
        for key, value in (("base_snapshot_sha256", expected_binding["model_sha256"]), ("training_data_sha256", expected_binding["data_sha256"]), ("training_config_sha256", expected_binding["config_sha256"])):
            _same(candidate.get(key), value, "Registered candidate differs from completed training binding")
    _same(full_summary.get("optimizer_steps"), settings["max_steps"], "Full training budget was not completed")
    _history(full_history, settings["max_steps"])
    for field, actual in (("first_loss", full_history[0]["loss"]), ("last_loss", full_history[-1]["loss"]), ("elapsed_s", full_history[-1]["elapsed_s"])):
        _same(full_summary.get(field), actual, "Full training summary differs from the complete history")
    _public_manifest(root, FULL_TRAINING, None, read)

    selected = read(confirmation / "selected-v3.json")
    checkpoint = _adapter_binding(root, ADAPTER, selected["model_provenance"], read)
    adapter_config = read(root / ADAPTER / "adapter_config.json")
    for public, setting in (("r", "lora_rank"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout"), ("base_model_name_or_path", "model_name")):
        _same(adapter_config.get(public), settings[setting], "Published adapter configuration differs from registered training")
    _same(sorted(adapter_config.get("target_modules", [])), sorted(settings["target_modules"]), "Published adapter target modules differ from training")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Published adapter is not the registered causal-LM LoRA")
    last_candidate = read(folder / f"dev-step-{STEPS[-1]:06d}.json")
    _same(full_summary.get("adapter_sha256"), last_candidate["model_provenance"]["adapter_sha256"], "Completed training summary differs from the final-step evaluated adapter")
    final_checkpoint_path = root / FULL_TRAINING / FULL_CHECKPOINT
    _same(sha256_file(final_checkpoint_path), last_candidate["model_provenance"]["checkpoint_sha256"], "Full-budget checkpoint differs from the final dev evaluation identity")
    final_checkpoint = read(final_checkpoint_path)
    if final_checkpoint.get("format") != "evidence-gate-checkpoint-v1":
        raise ValueError("Full-budget checkpoint format is invalid")
    _same(final_checkpoint.get("step"), 2400, "Full-budget checkpoint is not the final registered step")
    _same(final_checkpoint.get("binding"), expected_binding, "Full-budget checkpoint uses a different training binding")
    _same(final_checkpoint.get("history"), full_history, "Complete training history differs from the final evaluated checkpoint")
    for name, key in (("adapter_model.safetensors", "adapter_sha256"), ("adapter_config.json", "adapter_config_sha256")):
        _same(final_checkpoint["files"].get(name), last_candidate["model_provenance"].get(key), "Full-budget checkpoint files differ from final-step evaluation")
    step = chosen["step"]
    prefix = full_history[:step]
    _history(checkpoint["history"], step)
    _same(checkpoint["history"], prefix, "Selected checkpoint does not contain the actual full-run prefix")
    training = read(confirmation / "selected-training.json")
    _same(training.get("history"), prefix, "Report training history is not the selected prefix")
    metadata = training["metadata"]
    expected_metadata = {
        "format": "evidence-gate-selected-training-summary-v1", "model_name": full_manifest["model_name"],
        "device": settings["device"], "dtype": settings["dtype"], "seed": settings["seed"],
        "optimizer_steps": step, "selected_step": step, "completed_training_steps": settings["max_steps"], "total_training_budget_steps": settings["max_steps"],
        "first_loss": prefix[0]["loss"], "last_loss": prefix[-1]["loss"], "elapsed_s": prefix[-1]["elapsed_s"], "total_training_elapsed_s": full_summary["elapsed_s"],
        "trainable_parameters": full_summary["trainable_parameters"], "total_parameters": full_summary["total_parameters"],
        "binding": expected_binding, **{key: chosen[key] for key in ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256")},
        "selection_split": "dev", "test_used_for_selection": False, "training_is_quality_evidence": False,
    }
    for key, value in expected_metadata.items():
        _same(metadata.get(key), value, f"Selected training metadata differs from public evidence: {key}")
    _same(read(root / TRAINING / "summary.json"), metadata, "Public selected summary differs from confirmation training")
    _same(read(root / TRAINING / "metrics.json"), prefix, "Public selected metrics differ from the selected prefix")
    _same(read(root / TRAINING / "run_manifest.json"), full_manifest, "Selected manifest must preserve the actual full training configuration")
    _public_manifest(root, TRAINING, ADAPTER, read)
    for name in ("README.md", "LICENSE"):
        if not (root / ADAPTER / name).read_text(encoding="utf-8").strip():
            raise ValueError("Published adapter requires a nonempty model card and license")

    # Read test contents only after a complete public release and dev selection
    # have passed. No GPU/model process is ever created by this verifier.
    rows = {split: read_jsonl(path) for split, path in paths.items()}
    if any(any(row.get("split") != split for row in members) for split, members in rows.items()):
        raise ValueError("Released dataset splits were mixed")
    audit(rows["train"] + rows["dev"] + rows["test"])
    methods = [selected, read(confirmation / "contract.json"), read(confirmation / "base.json"), read(confirmation / "v2-evidence.json")]
    _same([method.get("name") for method in methods], ["selected-v3", "contract", "base", "v2-evidence"], "Confirmation reference methods changed")
    for method in methods:
        _raw_method(rows["test"], method)
    if methods[1].get("kind") != "rule" or any(method.get("kind") != "model" for method in (methods[0], methods[2], methods[3])):
        raise ValueError("Model and rule kinds are not the declared fixed references")
    for method in [*methods[::2], methods[3], *(read(path) for path in evaluations.values())]:
        _same({key: method.get("inference", {}).get(key) for key in INFERENCE}, protocol["inference"], "Model inference differs from frozen controls")
        provenance = method["model_provenance"]
        _same(provenance.get("base_snapshot_sha256"), expected_binding["model_sha256"], "Model base differs from the completed training snapshot")
        _same(provenance.get("base_files"), full_manifest["model_files"], "Evaluation base-file manifest differs from completed training")
    base = methods[2]
    if base["inference"].get("adapter_used") is not False or any(base["model_provenance"].get(key) is not None for key in ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256")):
        raise ValueError("Base reference unexpectedly uses an adapter")
    if methods[3]["inference"].get("adapter_used") is not True:
        raise ValueError("V2 reference must use its fixed released adapter")
    _adapter_binding(root, Path("artifacts/evidence-v2-lora"), methods[3]["model_provenance"], read)
    historical_v2 = read(root / "experiments/v2/selection.json")
    if historical_v2.get("format") != "evidence-gate-dev-selection-v1" or historical_v2.get("selection_split") != "dev" or historical_v2.get("test_used_for_selection") is not False:
        raise ValueError("Historical V2 reference lacks its original dev selection")
    expected_reference = {"adapter_sha256": historical_v2["selected_adapter_sha256"], "adapter_config_sha256": historical_v2["selected_adapter_config_sha256"],
                          "checkpoint_sha256": historical_v2["selected_checkpoint_sha256"], "checkpoint_step": historical_v2["selected_step"], **historical_v2["training_binding"]}
    for key, value in expected_reference.items():
        _same(methods[3]["model_provenance"].get(key), value, "V2 reference differs from its historical published selection")
    report = read(confirmation / "report.json")
    _same(report.get("methods"), methods, "Report methods differ from the four complete raw evaluation files")
    _same(report.get("training"), [training], "Report must contain exactly the selected training record")
    _same(report.get("provenance", {}).get("dataset"), digest(rows["test"]), "Report dataset identity changed")

    snapshots = {"selection": sha256_file(folder / "selection.json"), "protocol": sha256_file(protocol_path), "registry": sha256_file(registry_path), "test_data": sha256_file(paths["test"]), "dev_data": sha256_file(paths["dev"])}
    dev_snapshots = {name: sha256_file(path) for name, path in evaluations.items()}
    receipts = {name: read(confirmation / name) for name in ("confirmation-start.json", "confirmation.json")}
    for name, receipt in receipts.items():
        _same(receipt.get("selected"), chosen, "Confirmation receipt names a different selected model")
        _same(receipt.get("source_sha256"), snapshots, "Confirmation receipt source hashes changed")
        _same(receipt.get("dev_evaluation_sha256"), dev_snapshots, "Confirmation receipt dev evidence changed")
        if receipt.get("test_used_for_selection") is not False:
            raise ValueError("Confirmation receipt does not declare dev-only selection")
    start, complete = receipts["confirmation-start.json"], receipts["confirmation.json"]
    if start.get("format") != "evidence-gate-v3-confirmation-start-v1" or start.get("dev_revalidated") is not True:
        raise ValueError("Original confirmation start receipt is invalid")
    if complete.get("format") != "evidence-gate-v3-confirmation-v1" or complete.get("completed") is not True:
        raise ValueError("Final confirmation receipt is incomplete")
    _same(complete.get("quality_gate_status"), gate["status"], "Confirmation receipt misstates the actual quality gate")
    _same((complete.get("report"), complete.get("quality_gate")), ("report.json", "quality-gate.json"), "Confirmation receipt has unexpected report paths")
    # Scope is all *published* attempts for this holdout hash, including an
    # incomplete second start receipt. Hidden/private history is not provable.
    matches = {"confirmation-start.json": [], "confirmation.json": []}
    for name in matches:
        for path in _receipt_paths(root, name):
            receipt = read(path)
            if receipt.get("source_sha256", {}).get("test_data") == snapshots["test_data"]:
                matches[name].append(path.resolve())
        _same([path.as_posix() for path in matches[name]], [(confirmation / name).resolve().as_posix()], "More than one published main confirmation attempt uses this holdout")
    passed = gate["status"] == "pass" and gate["passed"] is True
    return {"status": "verified" if passed else "failed_quality", "publication_allowed": passed,
            "integrity_verified": True, "dev_candidates_recomputed": len(evaluations), "quality_gate_recomputed": True,
            "quality_gate_status": gate["status"], "selected_step": step, "completed_training_steps": 2400,
            "selected_history_prefix_verified": True, "published_confirmation_attempts": 1,
            "model_execution_performed": False, "optimizer_state_required": False,
            "scope": "Published raw outputs, identities and records only; does not prove unrecorded experiments absent or rerun model inference."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    try:
        result = verify_release(args.root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "failed", "publication_allowed": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "verified" else (2 if result["status"] == "pending" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
