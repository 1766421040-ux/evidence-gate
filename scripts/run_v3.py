"""Dev selection and a guarded, local-only V3 confirmation run.

Training is explicit: use evidence-gate train with the registered configuration.
This runner never downloads models, launches a cloud job, or tunes from test.
All candidate dev files must already exist. A failed dev selection refuses test
before an output directory or any model process is created.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from evidence_gate.training import _json_bytes, _model_manifest, normalize_config, sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
ATTEMPT_DIRECTORY = REPOSITORY / ".cache/confirmation-attempts"


def selection_module():
    spec = importlib.util.spec_from_file_location("evidence_gate_selection_v3", REPOSITORY / "scripts/selection_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def parse_evaluations(values):
    result = {}
    for value in values:
        candidate, separator, path = value.partition("=")
        if not separator or not candidate or not path or candidate in result:
            raise ValueError("Each --evaluation must be a unique candidate_id=path")
        result[candidate] = path
    return result


def register_round(*, config, train_data, model, protocol, round_id, steps):
    """Bind a fresh-data training round before weights or dev outputs exist."""
    from evidence_gate.data import audit, read_jsonl

    settings = normalize_config(config)
    protocol_record = json.loads(Path(protocol).read_text())
    verify_freeze(protocol)
    verify_frozen_code(protocol_record)
    data_sha = sha256_file(train_data)
    if data_sha != protocol_record["data_sha256"]["train"]:
        raise ValueError("This registration entrypoint requires the frozen V3 train file")
    rows = read_jsonl(train_data)
    audit(rows)
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Only nonempty split=train data may be registered")
    if not round_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in round_id):
        raise ValueError("Use a lowercase alphanumeric round id")
    if not steps or len(set(steps)) != len(steps) or any(type(step) is not int or step < 1 or step > settings["max_steps"] or step % settings["save_steps"] for step in steps):
        raise ValueError("Candidate steps must be unique saved checkpoints within the training budget")
    binding = {"base_snapshot_sha256": _model_manifest(Path(model))["sha256"],
               "training_data_sha256": data_sha,
               "training_config_sha256": hashlib.sha256(_json_bytes(settings)).hexdigest()}
    return {"format": "evidence-gate-v3-candidate-registry-v1",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(protocol), "config": settings,
            "source_config_sha256": sha256_file(config),
            "training_code_sha256": sha256_file(REPOSITORY / "src/evidence_gate/training.py"),
            "candidates": [{"id": f"{round_id}-step{step}", "checkpoint_step": step, **binding} for step in steps],
            "selection_scope": "This entire declared round; every registered dev evaluation is required. No test data is read by registration."}


def run_cli(*arguments):
    subprocess.run([sys.executable, "-m", "evidence_gate.cli", *map(str, arguments)], check=True, cwd=REPOSITORY)


def verify_frozen_code(protocol):
    """A frozen scoring contract cannot drift after model development."""
    recorded = protocol.get("generator_code_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Protocol must freeze generator and scoring source hashes")
    for relative, expected in recorded.items():
        path = (REPOSITORY / relative).resolve()
        if not path.is_relative_to(REPOSITORY) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen source changed or missing: {relative}")


def verify_freeze(protocol_path):
    """Check the original builder receipt, not just a protocol's own claims."""
    protocol_path = Path(protocol_path).resolve()
    freeze = json.loads(protocol_path.with_name("freeze.json").read_text())
    if freeze.get("format") not in {"evidence-gate-v3-freeze-v1", "evidence-gate-v31-freeze-v1"} or not isinstance(freeze.get("files"), dict):
        raise ValueError("A valid original V3 freeze receipt is required")
    recorded = freeze["files"]
    if not protocol_path.is_relative_to(REPOSITORY) or protocol_path.relative_to(REPOSITORY).as_posix() not in recorded:
        raise ValueError("Freeze receipt must cover this exact protocol file")
    for relative, expected in recorded.items():
        path = (REPOSITORY / relative).resolve()
        if not path.is_relative_to(REPOSITORY) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Original freeze mismatch: {relative}")


def selected_training_record(training_run, adapter, selected):
    """Attribute only the selected prefix, while disclosing all training cost."""
    root, adapter = Path(training_run).resolve(), Path(adapter).resolve()
    expected = root / "checkpoints" / f"step-{selected['step']:06d}"
    if adapter != expected:
        raise ValueError("Selected adapter must be a checkpoint of the declared training run")
    manifest = json.loads((root / "run_manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    checkpoint = json.loads((adapter / "checkpoint.json").read_text())
    if checkpoint["binding"] != manifest["binding"] or checkpoint["binding"] != summary["binding"]:
        raise ValueError("Selected checkpoint does not belong to the completed training run")
    settings = normalize_config(manifest["config"])
    if hashlib.sha256(_json_bytes(settings)).hexdigest() != checkpoint["binding"]["config_sha256"]:
        raise ValueError("Training manifest configuration differs from the checkpoint binding")
    if summary["optimizer_steps"] != manifest["config"]["max_steps"]:
        raise ValueError("Complete the registered training budget before confirmation")
    history = checkpoint["history"]
    if [row["step"] for row in history] != list(range(1, selected["step"] + 1)):
        raise ValueError("Selected checkpoint history is incomplete")
    if checkpoint["step"] != selected["step"]:
        raise ValueError("Selected step differs from checkpoint")
    complete_history = json.loads((root / "metrics.json").read_text())
    if selected["step"] > settings["max_steps"] or [row["step"] for row in complete_history] != list(range(1, settings["max_steps"] + 1)):
        raise ValueError("Completed history does not cover the registered training budget")
    if complete_history[:selected["step"]] != history:
        raise ValueError("Selected history differs from the completed run prefix")
    if summary["first_loss"] != complete_history[0]["loss"] or summary["last_loss"] != complete_history[-1]["loss"] or summary["elapsed_s"] != complete_history[-1]["elapsed_s"]:
        raise ValueError("Completed training summary differs from its actual history")
    for name in ("adapter_model.safetensors", "adapter_config.json", "training_state.safetensors"):
        if checkpoint["files"].get(name) != sha256_file(adapter / name):
            raise ValueError(f"Checkpoint file changed: {name}")
    for name, key in (("adapter_model.safetensors", "adapter_sha256"),
                      ("adapter_config.json", "adapter_config_sha256"),
                      ("checkpoint.json", "checkpoint_sha256")):
        if selected[key] != sha256_file(adapter / name):
            raise ValueError(f"Selection identity changed: {name}")
    return {
        "name": f"selected-v3-{selected['id']}", "history": history,
        "metadata": {
            "format": "evidence-gate-selected-training-summary-v1",
            "model_name": manifest["model_name"],
            "device": manifest["config"]["device"], "dtype": manifest["config"]["dtype"],
            "seed": manifest["config"]["seed"], "optimizer_steps": selected["step"],
            "selected_step": selected["step"], "completed_training_steps": summary["optimizer_steps"],
            "total_training_budget_steps": manifest["config"]["max_steps"],
            "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
            "elapsed_s": history[-1]["elapsed_s"], "total_training_elapsed_s": summary["elapsed_s"],
            "trainable_parameters": summary["trainable_parameters"], "total_parameters": summary["total_parameters"],
            "binding": checkpoint["binding"],
            "adapter_sha256": selected["adapter_sha256"],
            "adapter_config_sha256": selected["adapter_config_sha256"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_split": "dev", "test_used_for_selection": False,
            "training_is_quality_evidence": False,
            "history_scope": "Selected checkpoint prefix; full completed budget is reported separately.",
        },
    }


def confirm(*, model, adapter, training_run, selection, protocol, registry,
            dev_data, test_data, evaluations, output, reference_adapter=None):
    """Open test only after recomputing every dev gate and checking identities."""
    module = selection_module()
    chosen = module.require_test_authorization(selection, protocol, registry, adapter, dev_data, evaluations)
    # These checks happen before creating a run or launching any subprocess.
    training = selected_training_record(training_run, adapter, chosen)
    model, output = Path(model).resolve(), Path(output).resolve()
    if not model.is_dir():
        raise ValueError("Model must be an existing local snapshot")
    if output.exists():
        raise FileExistsError("Choose a fresh confirmation directory; existing evidence is never overwritten")
    protocol_record = json.loads(Path(protocol).read_text())
    verify_freeze(protocol)
    verify_frozen_code(protocol_record)
    inference = protocol_record["inference"]
    snapshots = {key: sha256_file(value) for key, value in (
        ("selection", selection), ("protocol", protocol), ("registry", registry),
        ("test_data", test_data), ("dev_data", dev_data))}
    evaluation_snapshots = {name: sha256_file(path) for name, path in evaluations.items()}
    # Check test identity against the protocol before its first model call.
    expected_test = protocol_record["data_sha256"]["test"]
    if snapshots["test_data"] != expected_test:
        raise ValueError("Test bytes differ from frozen protocol")
    # A different output directory must not silently turn the same holdout into
    # a second 'fresh' confirmation after development. This is a local guard,
    # not a secrecy/security claim: a fresh clone can replay published weights,
    # but that replay remains historical verification, not an unseen test.
    attempt = {"format": "evidence-gate-confirmation-attempt-v1",
               "started_at": datetime.now(timezone.utc).isoformat(),
               "selected": chosen, "source_sha256": snapshots,
               "dev_evaluation_sha256": evaluation_snapshots,
               "rule": "This test hash is consumed for local confirmation once inference is attempted; a crash is not permission to retune and relabel it fresh."}
    attempt_file = ATTEMPT_DIRECTORY / f"{snapshots['test_data']}.json"
    if attempt_file.exists():
        raise FileExistsError("This test hash already has a confirmation attempt; preserve it as seen and audit saved outputs")
    write_json(attempt_file, attempt)  # Exclusive creation also prevents a race.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "confirmation-start.json", {
        "format": "evidence-gate-v3-confirmation-start-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "selected": chosen, "source_sha256": snapshots,
        "dev_evaluation_sha256": evaluation_snapshots,
        "dev_revalidated": True, "test_used_for_selection": False,
        "failure_policy": "Retain every failure. Any development informed by this test makes it seen; do not reuse it as fresh confirmation.",
    })
    write_json(output / "selected-training.json", training)

    def evaluate(name, adapter_path=None, policy="hf"):
        path = output / f"{name}.json"
        args = ["evaluate", "--data", test_data, "--policy", policy, "--name", name, "--output", path]
        if policy == "hf":
            args += ["--model", model, "--device", inference["device"], "--dtype", inference["dtype"],
                     "--batch-size", str(inference["batch_size"]), "--max-new-tokens", str(inference["max_new_tokens"])]
            if adapter_path:
                args += ["--adapter", adapter_path]
        run_cli(*args)
        return path

    selected_result = evaluate("selected-v3", adapter)
    audit = module.audit_test(selected_result, test_data, protocol, selection)
    write_json(output / "quality-gate.json", audit)
    # Fixed references are reported even on failure; none can change selection.
    methods = [selected_result, evaluate("contract", policy="contract"), evaluate("base")]
    if reference_adapter:
        methods.append(evaluate("v2-evidence", reference_adapter))
    for key, path in (("selection", selection), ("protocol", protocol), ("registry", registry),
                      ("test_data", test_data), ("dev_data", dev_data)):
        if sha256_file(path) != snapshots[key]:
            raise ValueError(f"Source changed during confirmation: {key}")
    verify_frozen_code(protocol_record)
    for name, path in evaluations.items():
        if sha256_file(path) != evaluation_snapshots[name]:
            raise ValueError(f"Dev evaluation changed during confirmation: {name}")
    if module.require_test_authorization(selection, protocol, registry, adapter, dev_data, evaluations) != chosen:
        raise ValueError("Selected identity changed during confirmation")
    verify_freeze(protocol)
    run_cli("report", *methods, "--training", output / "selected-training.json", "--output", output / "report.json")
    result = {"format": "evidence-gate-v3-confirmation-v1", "completed": True,
              "quality_gate_status": audit["status"], "selected": chosen,
              "report": "report.json", "quality_gate": "quality-gate.json",
              "test_used_for_selection": False, "source_sha256": snapshots,
              "dev_evaluation_sha256": evaluation_snapshots}
    write_json(output / "confirmation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("--config", required=True)
    registration.add_argument("--train-data", required=True)
    registration.add_argument("--model", required=True)
    registration.add_argument("--protocol", required=True)
    registration.add_argument("--round-id", required=True)
    registration.add_argument("--steps", nargs="+", type=int, required=True)
    registration.add_argument("--output", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", required=True)
    common.add_argument("--registry", required=True)
    common.add_argument("--dev-data", required=True)
    common.add_argument("--evaluation", action="append", required=True, help="candidate_id=dev_result.json")
    common.add_argument("--output", required=True)
    commands.add_parser("select", parents=[common])
    test = commands.add_parser("confirm", parents=[common])
    test.add_argument("--selection", required=True)
    test.add_argument("--test-data", required=True)
    test.add_argument("--model", required=True)
    test.add_argument("--adapter", required=True)
    test.add_argument("--training-run", required=True)
    test.add_argument("--reference-adapter")
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            result = register_round(config=args.config, train_data=args.train_data, model=args.model,
                                    protocol=args.protocol, round_id=args.round_id, steps=args.steps)
            write_json(args.output, result)
        elif args.command == "select":
            evaluations = parse_evaluations(args.evaluation)
            result = selection_module().select_candidates(evaluations, args.dev_data, args.protocol, args.registry)
            write_json(args.output, result)
        else:
            evaluations = parse_evaluations(args.evaluation)
            result = confirm(model=args.model, adapter=args.adapter, training_run=args.training_run,
                             selection=args.selection, protocol=args.protocol, registry=args.registry,
                             dev_data=args.dev_data, test_data=args.test_data, evaluations=evaluations,
                             output=args.output, reference_adapter=args.reference_adapter)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result.get(key) for key in ("status", "test_allowed", "quality_gate_status", "selected")}, indent=2))


if __name__ == "__main__":
    main()
