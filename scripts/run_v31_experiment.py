"""Reproduce the fixed V3.1 local experiment, including its dev-only test gate.

Usage (with ML dependencies installed and an existing pinned local snapshot):
  PYTHONPATH=src python scripts/run_v31_experiment.py --model PATH --output runs/NEW

This runs 2400 MPS training steps, four frozen dev evaluations and, only if an
eligible candidate exists, run_v3's one-attempt confirmation with fixed
contract/base/V2 references. A historical protocol replay is not a new,
independent test. No downloads, publication, threshold overrides, ledger bypass,
resume or output overwriting are offered. An existing local confirmation ledger
refuses the workflow before training; published historical results remain seen
even when reproducing them on a fresh machine.

Exit 0: completed confirmation passed; 2: no eligible dev candidate; 1: failure
or failed final confirmation. Evidence in the fresh output directory is retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from evidence_gate.data import audit, canonical, read_jsonl
from evidence_gate.hf_policy import _check_adapter_metadata
from evidence_gate.training import _json_bytes, _model_manifest, normalize_config, sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
STEPS = (600, 1200, 1800, 2400)
INFERENCE = {"dtype": "bfloat16", "device": "mps", "batch_size": 4,
             "decoding": "greedy free generation", "max_new_tokens": 72, "output_repair": False}


def _runner():
    spec = importlib.util.spec_from_file_location("v31_fixed_runner", REPOSITORY / "scripts/run_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same(actual, expected, message):
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_new_output(output, model):
    lexical = Path(output).expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError("Output must be a completely new directory; existing evidence is never overwritten")
    path = lexical.resolve()
    # Parent symlinks create ambiguous provenance and can redirect writes into
    # the model or published evidence after an otherwise harmless-looking path.
    if lexical != path:
        raise ValueError("Use an output path without symlinked or relative parent traversal")
    protected = [model, *(REPOSITORY / name for name in ("data", "experiments", "artifacts", "configs", "src", "scripts", ".git", ".cache"))]
    if any(path == parent or path.is_relative_to(parent.resolve()) for parent in protected):
        raise ValueError("Output cannot be inside the model snapshot or frozen/public source directories")
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise PermissionError("Output parent must be an existing writable directory")
    return path


def _assert_unused_holdout(runner, protocol):
    ledger = runner.ATTEMPT_DIRECTORY / f"{protocol['data_sha256']['test']}.json"
    if ledger.exists() or ledger.is_symlink():
        raise FileExistsError("This holdout already has a local confirmation attempt; refusing before training. Audit saved evidence instead")
    ancestor = ledger.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise PermissionError("The once-per-holdout ledger directory is not writable")


def _runtime_check():
    for name in ("torch", "transformers", "peft", "safetensors"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Install the local training dependency before starting: {name}")
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("The frozen experiment requires MPS; no silent CPU or cloud fallback")


def preflight(model, output, runner):
    """Validate all fixed inputs before creating an output or model subprocess.

    Test bytes are hashed by freeze validation but are not parsed or evaluated.
    """
    model = Path(model).expanduser().resolve()
    if not model.is_dir():
        raise ValueError("Model must be an existing downloaded local snapshot")
    output = _assert_new_output(output, model)
    folder = REPOSITORY / "experiments/v31"
    protocol_path, config_path = folder / "protocol.json", REPOSITORY / "configs/v31.json"
    historical_path = folder / "registry.json"
    protocol, historical = _read(protocol_path), _read(historical_path)
    _assert_unused_holdout(runner, protocol)
    runner.verify_freeze(protocol_path)
    runner.verify_frozen_code(protocol)
    _same(historical.get("protocol_sha256"), sha256_file(protocol_path), "Historical registry protocol changed")
    _same(historical.get("source_config_sha256"), sha256_file(config_path), "Frozen source configuration changed")
    _same(historical.get("training_code_sha256"), sha256_file(REPOSITORY / "src/evidence_gate/training.py"), "Registered training code changed")
    settings = normalize_config(config_path)
    _same(settings, historical.get("config"), "Source configuration differs from historical registration")
    _same({key: settings[key] for key in ("max_steps", "save_steps", "device", "dtype")},
          {"max_steps": 2400, "save_steps": 600, "device": "mps", "dtype": "bfloat16"}, "This entrypoint only reproduces the fixed 2400-step MPS experiment")
    _same(protocol.get("inference"), INFERENCE, "Frozen V3.1 inference controls changed")
    _same([candidate.get("checkpoint_step") for candidate in historical["candidates"]], list(STEPS), "Historical candidate steps/order changed")
    paths = {split: REPOSITORY / f"data/v31/{split}.jsonl" for split in ("train", "dev", "test")}
    for split, path in paths.items():
        _same(sha256_file(path), protocol["data_sha256"][split], "Data differs from its frozen byte identity")
    rows = {split: read_jsonl(paths[split]) for split in ("train", "dev")}
    if any(not members or any(row.get("split") != split for row in members) for split, members in rows.items()):
        raise ValueError("Only the fixed nonempty train/dev splits may enter development")
    audit(rows["train"] + rows["dev"])
    model_manifest = _model_manifest(model)
    binding = {"base_snapshot_sha256": model_manifest["sha256"], "training_data_sha256": sha256_file(paths["train"]),
               "training_config_sha256": hashlib.sha256(_json_bytes(settings)).hexdigest()}
    for candidate in historical["candidates"]:
        _same({key: candidate.get(key) for key in binding}, binding, "Local base/data/config differs from the historical experiment")
    reference = REPOSITORY / "artifacts/evidence-v2-lora"
    original_v2 = _read(REPOSITORY / "experiments/v2/selection.json")
    _same(protocol.get("baseline_protocol_sha256"), sha256_file(REPOSITORY / "experiments/v2/protocol.json"), "Historical V2 reference protocol changed")
    for filename, key in (("adapter_model.safetensors", "selected_adapter_sha256"), ("adapter_config.json", "selected_adapter_config_sha256"), ("checkpoint.json", "selected_checkpoint_sha256")):
        _same(sha256_file(reference / filename), original_v2[key], "Fixed V2 reference adapter changed")
    reference_provenance = _check_adapter_metadata(reference, model_manifest["sha256"], original_v2["selected_adapter_sha256"])
    if reference_provenance.get("adapter_training_binding_verified") is not True:
        raise ValueError("The fixed V2 reference training binding is unverified")
    for key in ("training_data_sha256", "training_config_sha256", "adapter_config_sha256"):
        _same(reference_provenance.get(key), original_v2["training_binding"][key], "V2 reference differs from historical training")
    _same(reference_provenance.get("checkpoint_step"), original_v2["selected_step"], "V2 reference step changed")
    _runtime_check()
    reference_sources = [REPOSITORY / "experiments/v2/selection.json", REPOSITORY / "experiments/v2/protocol.json",
                         *(reference / name for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint.json"))]
    return {"model": model, "output": output, "config": config_path, "protocol": protocol_path,
            "protocol_record": protocol, "historical_registry": historical_path, "paths": paths, "reference": reference,
            "reference_sha256": {path.relative_to(REPOSITORY).as_posix(): sha256_file(path) for path in reference_sources},
            "binding": binding, "source_sha256": {"protocol": sha256_file(protocol_path), "historical_registry": sha256_file(historical_path), "config": sha256_file(config_path), "training_code": historical["training_code_sha256"]}}


def _recheck(plan, runner):
    runner.verify_freeze(plan["protocol"])
    runner.verify_frozen_code(plan["protocol_record"])
    for key, path in (("protocol", plan["protocol"]), ("historical_registry", plan["historical_registry"]), ("config", plan["config"]), ("training_code", REPOSITORY / "src/evidence_gate/training.py")):
        _same(sha256_file(path), plan["source_sha256"][key], "Fixed source changed during reproduction")
    for relative, expected in plan["reference_sha256"].items():
        _same(sha256_file(REPOSITORY / relative), expected, "Fixed V2 reference changed during reproduction")


@contextmanager
def _offline_environment():
    updates = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_experiment(model, output):
    runner = _runner()
    plan = preflight(model, output, runner)
    # register_round is read-only until its returned record is written below.
    registered = runner.register_round(config=plan["config"], train_data=plan["paths"]["train"], model=plan["model"],
                                       protocol=plan["protocol"], round_id="reproduction-r8", steps=list(STEPS))
    for candidate in registered["candidates"]:
        _same({key: candidate.get(key) for key in plan["binding"]}, plan["binding"], "Model/data/config changed between preflight and registration")
    _assert_unused_holdout(runner, plan["protocol_record"])
    _recheck(plan, runner)
    output = plan["output"]
    output.mkdir(parents=True, exist_ok=False)
    registry_path, selection_path = output / "registry.json", output / "selection.json"
    runner.write_json(registry_path, registered)
    common = {"format": "evidence-gate-v31-reproduction-v1", "historical_protocol_reproduction": True, "new_independent_test": False,
              "interpretation": "Reproducing this published protocol does not create an unseen independent holdout or a new generalization claim."}
    runner.write_json(output / "reproduction-start.json", {**common, "started_at": datetime.now(timezone.utc).isoformat(), "binding": plan["binding"], "source_sha256": plan["source_sha256"], "reference_sha256": plan["reference_sha256"], "registered_steps": list(STEPS)})
    training = output / "training"
    stage = "training"
    try:
        with _offline_environment():
            runner.run_cli("train", "--data", plan["paths"]["train"], "--model", plan["model"], "--config", plan["config"], "--output", training)
            _recheck(plan, runner)
            stage = "training_verification"
            final_adapter = training / "checkpoints" / f"step-{STEPS[-1]:06d}"
            final_identity = {"id": registered["candidates"][-1]["id"], "step": STEPS[-1],
                              **{key: sha256_file(final_adapter / filename) for key, filename in (("adapter_sha256", "adapter_model.safetensors"), ("adapter_config_sha256", "adapter_config.json"), ("checkpoint_sha256", "checkpoint.json"))}}
            completed_training = runner.selected_training_record(training, final_adapter, final_identity)
            expected_binding = {"model_sha256": plan["binding"]["base_snapshot_sha256"], "data_sha256": plan["binding"]["training_data_sha256"], "config_sha256": plan["binding"]["training_config_sha256"]}
            _same(completed_training["metadata"]["binding"], expected_binding, "Completed training differs from preflight registration")
            _same(completed_training["metadata"]["completed_training_steps"], STEPS[-1], "The complete training budget is required before dev selection")
            stage = "dev_evaluation"
            evaluations = {}
            for candidate in registered["candidates"]:
                step = candidate["checkpoint_step"]
                path = output / f"dev-step-{step:06d}.json"
                runner.run_cli("evaluate", "--data", plan["paths"]["dev"], "--policy", "hf", "--name", candidate["id"],
                               "--model", plan["model"], "--adapter", training / "checkpoints" / f"step-{step:06d}",
                               "--device", INFERENCE["device"], "--dtype", INFERENCE["dtype"], "--batch-size", str(INFERENCE["batch_size"]),
                               "--max-new-tokens", str(INFERENCE["max_new_tokens"]), "--output", path)
                evaluations[candidate["id"]] = path
            stage = "dev_selection"
            selected = runner.selection_module().select_candidates(evaluations, plan["paths"]["dev"], plan["protocol"], registry_path)
            runner.write_json(selection_path, selected)
            _recheck(plan, runner)
            if selected.get("status") == "no_eligible_candidate" and selected.get("selected") is None and selected.get("test_allowed") is False:
                result = {**common, "status": "no_eligible_candidate", "selected": None, "test_inference_performed": False, "test_allowed": False, "raw_dev_files": [path.name for path in evaluations.values()]}
            else:
                if selected.get("status") != "selected" or selected.get("test_allowed") is not True or not isinstance(selected.get("selected"), dict) or selected["selected"].get("step") not in STEPS:
                    raise ValueError("Unexpected dev selection state; test is forbidden")
                stage = "confirmation"
                confirmed = runner.confirm(model=plan["model"], adapter=training / "checkpoints" / f"step-{selected['selected']['step']:06d}",
                                           training_run=training, selection=selection_path, protocol=plan["protocol"], registry=registry_path,
                                           dev_data=plan["paths"]["dev"], test_data=plan["paths"]["test"], evaluations=evaluations,
                                           output=output / "confirmation", reference_adapter=plan["reference"])
                _recheck(plan, runner)
                result = {**common, "status": "confirmed_pass" if confirmed["quality_gate_status"] == "pass" else "confirmed_fail",
                          "selected": selected["selected"], "test_inference_performed": True, "quality_gate_status": confirmed["quality_gate_status"], "confirmation": "confirmation/confirmation.json"}
        runner.write_json(output / "outcome.json", result)
        return result
    except (Exception, KeyboardInterrupt) as error:
        runner.write_json(output / "failure.json", {**common, "status": "failed", "stage": stage, "error_type": type(error).__name__, "error": str(error),
                                                    "note": "Retain partial evidence and the existing holdout ledger. This entrypoint never resumes or overwrites."})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing pinned local base snapshot; no Hub download")
    parser.add_argument("--output", required=True, help="Completely new output directory, e.g. runs/v31-reproduction")
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.model, args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "confirmed_pass" else (2 if result["status"] == "no_eligible_candidate" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
