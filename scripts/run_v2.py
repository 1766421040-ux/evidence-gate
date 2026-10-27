"""Reproduce the independent v2 follow-up in a new directory, without cloud APIs.

From the repository root, after installing .[train] and downloading the pinned
base snapshot:

    python scripts/run_v2.py --model .cache/base-model --device mps --output runs/my-v2
    evidence-gate serve --report runs/my-v2/report.json

This command really trains when invoked. It regenerates the locked v2 data,
trains once for 600 steps, evaluates 200/400/600 on dev, selects by the saved dev
rule, then evaluates the selected checkpoint, contract rule and base on test.
Quality-gate failure remains a saved result; no third tuning round is started.
An existing output directory is always refused, including interrupted runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from evidence_gate.data import read_jsonl
from evidence_gate.training import normalize_config, sha256_file, target_text


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
PROTOCOL = REPOSITORY / "experiments/v2/protocol.json"
STEPS = (200, 400, 600)


def build_dataset(output, source, **kwargs):
    # Load this sibling by file path so both script execution and import-based
    # CPU tests work without changing sys.path or importing any ML framework.
    spec = importlib.util.spec_from_file_location("evidence_gate_build_v2", SCRIPTS / "build_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(output=output, source=source, **kwargs)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tokenizer_preflight(model, data, initial_length):
    """Tokenize train/dev only. No model weights are loaded; test is untouched."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    splits = {}
    for split in ("train", "dev"):
        path = Path(data) / f"{split}.jsonl"
        rows = read_jsonl(path)
        if any(row["split"] != split for row in rows):
            raise ValueError(f"Unexpected split during {split} tokenizer preflight")
        lengths = [len(tokenizer.apply_chat_template(
            row["messages"] + [{"role": "assistant", "content": target_text(row["target"])}],
            tokenize=True, add_generation_prompt=False,
        )) for row in rows]
        splits[split] = {"rows": len(rows), "file_sha256": sha256_file(path),
                         "max_sequence_tokens": max(lengths),
                         "mean_sequence_tokens": sum(lengths) / len(lengths)}
    longest = max(record["max_sequence_tokens"] for record in splits.values())
    selected_length = initial_length if longest <= initial_length else ((longest + 63) // 64) * 64
    return {"format": "evidence-gate-v2-tokenizer-preflight-v1", "splits": splits,
            "test_read": False, "rule": "Only increase from 512 to next multiple of 64 when required; no truncation",
            "max_length_before": initial_length, "max_length_selected": selected_length}


def run_command(*command):
    subprocess.run([sys.executable, *map(str, command)], check=True, cwd=REPOSITORY)


def run_cli(*arguments):
    run_command("-m", "evidence_gate.cli", *arguments)


def selected_training_record(training, selection, total_budget):
    """Bind the selected weights to their own checkpoint history, not step 600."""
    if selection.get("selection_split") != "dev" or selection.get("test_used_for_selection") is not False:
        raise ValueError("Test evaluation requires a completed, dev-only selection manifest")
    step = selection.get("selected_step")
    if type(step) is not int or step not in STEPS:
        raise ValueError("Selection is outside the predeclared checkpoints")
    checkpoint = training / "checkpoints" / f"step-{step:06d}"
    checkpoint_file = checkpoint / "checkpoint.json"
    metadata = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    summary = json.loads((training / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((training / "run_manifest.json").read_text(encoding="utf-8"))
    if summary.get("optimizer_steps") != total_budget or manifest.get("config", {}).get("max_steps") != total_budget:
        raise ValueError("Training did not complete the predeclared 600-step budget")
    if metadata.get("format") != "evidence-gate-checkpoint-v1" or metadata.get("step") != step:
        raise ValueError("Selected checkpoint metadata has the wrong step or format")
    if metadata.get("binding") != manifest.get("binding") or metadata.get("binding") != summary.get("binding"):
        raise ValueError("Selected checkpoint is not bound to this training run")
    actual_hashes = {
        "selected_adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        "selected_adapter_config_sha256": sha256_file(checkpoint / "adapter_config.json"),
        "selected_checkpoint_sha256": sha256_file(checkpoint_file),
    }
    if any(selection.get(key) != value for key, value in actual_hashes.items()):
        raise ValueError("Selected artifact hashes do not match the completed dev selection")
    files = metadata.get("files", {})
    for name in ("adapter_model.safetensors", "adapter_config.json", "training_state.safetensors"):
        if files.get(name) != sha256_file(checkpoint / name):
            raise ValueError(f"Selected checkpoint file checksum mismatch: {name}")
    history = metadata.get("history", [])
    if not isinstance(history, list) or [entry.get("step") for entry in history] != list(range(1, step + 1)):
        raise ValueError("Selected checkpoint must contain its complete own training history")
    record = {
        "name": f"selected-v2-step-{step}", "history": history,
        "metadata": {
            "format": "evidence-gate-selected-training-summary-v1",
            "model_name": manifest["model_name"], "device": manifest["config"]["device"],
            "dtype": manifest["config"]["dtype"], "seed": manifest["config"]["seed"],
            "optimizer_steps": step, "selected_step": step,
            "total_training_budget_steps": total_budget,
            "completed_training_steps": summary["optimizer_steps"],
            "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
            "elapsed_s": history[-1]["elapsed_s"],
            "total_training_elapsed_s": summary["elapsed_s"],
            "trainable_parameters": summary["trainable_parameters"],
            "total_parameters": summary["total_parameters"],
            "binding": metadata["binding"],
            "adapter_sha256": actual_hashes["selected_adapter_sha256"],
            "adapter_config_sha256": actual_hashes["selected_adapter_config_sha256"],
            "checkpoint_sha256": actual_hashes["selected_checkpoint_sha256"],
            "selection_split": "dev", "test_used_for_selection": False,
            "training_is_quality_evidence": False,
            "history_scope": "Only the selected checkpoint prefix is plotted. Total completed training budget is recorded separately; later-step losses are not attributed to these weights.",
        },
    }
    return checkpoint, record


def reproduce(model, output, config_path, device="mps"):
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError("Output already exists; choose a new run directory, even after an interrupted run")
    model = Path(model).expanduser().resolve()
    if not model.is_dir():
        raise ValueError("Use an existing local base-model snapshot; this runner never downloads models")
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("checkpoints") != list(STEPS):
        raise ValueError("The locked protocol must predeclare steps 200, 400 and 600")
    config = normalize_config(config_path)
    if config["max_steps"] != 600 or config["save_steps"] != 200 or config["max_length"] != 512:
        raise ValueError("v2 reproduction requires 600 steps, save_steps=200 and initial max_length=512")
    config["device"] = device
    root.mkdir(parents=True, exist_ok=False)
    # Copy the protocol byte-for-byte, before any inference. Thresholds and
    # selection rules are never derived from generated dev or test outcomes.
    protocol_copy = root / "protocol.json"
    with protocol_copy.open("xb") as handle:
        handle.write(protocol_bytes)
    data = root / "data"
    build_dataset(output=data, source=REPOSITORY / "data/main", seed=protocol["data_seed"],
                  train_groups=protocol["train_groups"], dev_groups=protocol["dev_groups"], test_groups=protocol["test_groups"])
    preflight = tokenizer_preflight(model, data, config["max_length"])
    config["max_length"] = preflight["max_length_selected"]
    config_file = root / "config.json"
    write_json(config_file, config)
    preflight["config_file_sha256"] = sha256_file(config_file)
    write_json(root / "tokenizer-preflight.json", preflight)
    write_json(root / "run-plan.json", {
        "format": "evidence-gate-v2-run-plan-v1", "protocol_sha256": sha256_file(protocol_copy),
        "config_file_sha256": sha256_file(config_file), "source_config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(data / "manifest.json"),
        "training_budget_steps": 600, "dev_checkpoint_steps": list(STEPS),
        "test_inference_allowed_after": "Completed dev selection and verified selected checkpoint hashes",
        "quality_failure_policy": "Save the failure, finish the report, and stop without another training round",
    })
    training = root / "training"
    evaluations = root / "evaluations"
    evaluations.mkdir()
    run_cli("train", "--data", data / "train.jsonl", "--model", model,
            "--config", config_file, "--output", training)

    # All model evaluations use the same decoding settings, including base.
    def evaluate(split, name, adapter=None, policy="hf"):
        destination = evaluations / f"{name}-{split}.json"
        command = ["evaluate", "--data", data / f"{split}.jsonl", "--policy", policy,
                   "--name", name, "--output", destination]
        if policy == "hf":
            command.extend(["--model", model, "--device", device, "--dtype", config["dtype"],
                            "--batch-size", "4", "--max-new-tokens", "72"])
            if adapter is not None:
                command.extend(["--adapter", adapter])
        run_cli(*command)
        return destination

    dev_files = [evaluate("dev", f"step-{step}", training / "checkpoints" / f"step-{step:06d}") for step in STEPS]
    selection_file = root / "selection.json"
    run_command(SCRIPTS / "select_checkpoint.py", *dev_files, "--data", data / "dev.jsonl", "--output", selection_file)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    selected, training_record = selected_training_record(training, selection, config["max_steps"])
    selected_training_file = root / "selected-training.json"
    write_json(selected_training_file, training_record)

    # No test inference occurs above this boundary. Selection is already saved
    # and its hashes verified; neither the base nor selected model is re-tuned.
    selected_test = evaluate("test", "selected-v2", selected)
    contract_test = evaluate("test", "contract", policy="contract")
    base_test = evaluate("test", "base")
    quality_file = root / "quality-gate.json"
    run_command(SCRIPTS / "quality_gate.py", "--evaluation", selected_test, "--data", data / "test.jsonl",
                "--protocol", protocol_copy, "--output", quality_file)
    quality = json.loads(quality_file.read_text(encoding="utf-8"))
    run_cli("report", selected_test, contract_test, base_test,
            "--training", selected_training_file, "--output", root / "report.json")
    result = {"format": "evidence-gate-v2-reproduction-v1", "completed": True,
              "selected_step": selection["selected_step"],
              "selected_adapter": selected.relative_to(root).as_posix(),
              "selected_adapter_sha256": selection["selected_adapter_sha256"],
              "total_training_budget_steps": config["max_steps"],
              "quality_gate_status": quality["status"], "report": "report.json",
              "selection": "selection.json", "quality_gate": "quality-gate.json",
              "selected_training": "selected-training.json"}
    write_json(root / "reproduction.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Existing pinned local snapshot directory")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--output", default="runs/my-v2")
    parser.add_argument("--config", default=str(REPOSITORY / "configs/v2.json"))
    args = parser.parse_args(argv)
    try:
        result = reproduce(args.model, args.output, args.config, args.device)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
