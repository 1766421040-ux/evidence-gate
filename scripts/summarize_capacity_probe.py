"""Audit a matched LoRA-module capacity probe on previously seen V2 dev data.

This CPU-only report compares saved raw generations and training receipts. It
does not load a model, select a V3 checkpoint, or open a test dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from evidence_gate.data import audit, canonical, digest, read_jsonl
from evidence_gate.evaluation import ACTIONS, score, summarize
from evidence_gate.training import _json_bytes, normalize_config, sha256_file

STEPS = (200, 400, 600)
BASE_MODULES = {"q_proj", "v_proj"}
PROBE_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
INFERENCE = ("decoding", "max_new_tokens", "dtype", "batch_size", "device", "output_repair")


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def _source(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    return {"file": name, "sha256": sha256_file(path)}


def _same(left, right):
    return canonical(left) == canonical(right)


def _run(path):
    path = Path(path)
    manifest = _read(path / "run_manifest.json")
    summary = _read(path / "summary.json")
    history = _read(path / "metrics.json")
    config = normalize_config(manifest["config"])
    binding = manifest["binding"]
    if hashlib.sha256(_json_bytes(config)).hexdigest() != binding["config_sha256"]:
        raise ValueError("Training manifest configuration is not bound to the run")
    if hashlib.sha256(_json_bytes(manifest["model_files"])).hexdigest() != binding["model_sha256"]:
        raise ValueError("Training model snapshot manifest is inconsistent")
    if not _same(binding, summary["binding"]) or summary["optimizer_steps"] != config["max_steps"]:
        raise ValueError("Training budget must be completed with the same binding")
    if [row["step"] for row in history] != list(range(1, config["max_steps"] + 1)):
        raise ValueError("Completed training metrics are incomplete")
    for key, actual in (("elapsed_s", history[-1]["elapsed_s"]), ("first_loss", history[0]["loss"]), ("last_loss", history[-1]["loss"])):
        if not _same(summary[key], actual):
            raise ValueError("Training summary differs from the complete metrics")
    return {"path": path, "config": config, "binding": binding, "summary": summary, "history": history,
            "sources": {name: _source(path / name) for name in ("run_manifest.json", "summary.json", "metrics.json")}}


def _evaluation(path, rows, run, step):
    evaluation = _read(path)
    if evaluation.get("kind") != "model" or evaluation.get("dataset_sha256") != digest(rows):
        raise ValueError("Capacity probe needs model evaluations of the exact dev dataset")
    inference = evaluation.get("inference", {})
    if any(inference.get(key) is None for key in INFERENCE) or inference.get("adapter_used") is not True:
        raise ValueError("Evaluation lacks adapter inference settings")
    if inference["decoding"] != "greedy free generation" or inference["output_repair"] is not False:
        raise ValueError("Capacity comparison requires unrepaired greedy generation")
    provenance = evaluation.get("model_provenance", {})
    binding_keys = {"base_snapshot_sha256": "model_sha256", "training_data_sha256": "data_sha256", "training_config_sha256": "config_sha256"}
    if provenance.get("adapter_training_binding_verified") is not True or not _same(provenance.get("checkpoint_step"), step):
        raise ValueError("Evaluation has an unverified or unexpected checkpoint")
    if any(provenance.get(public) != run["binding"][internal] for public, internal in binding_keys.items()):
        raise ValueError("Evaluation checkpoint differs from the declared training run")
    checkpoint_dir = run["path"] / "checkpoints" / f"step-{step:06d}"
    checkpoint = _read(checkpoint_dir / "checkpoint.json")
    if not _same(checkpoint["binding"], run["binding"]) or not _same(checkpoint["step"], step) or not _same(checkpoint["history"], run["history"][:step]):
        raise ValueError("Checkpoint does not match the completed training prefix")
    for filename, key in (("checkpoint.json", "checkpoint_sha256"), ("adapter_model.safetensors", "adapter_sha256"), ("adapter_config.json", "adapter_config_sha256")):
        actual = sha256_file(checkpoint_dir / filename)
        if provenance.get(key) != actual or (filename != "checkpoint.json" and checkpoint["files"].get(filename) != actual):
            raise ValueError("Evaluation adapter/config/checkpoint identity changed")
    lookup = {row["id"]: row for row in rows}
    cases = evaluation.get("cases", [])
    if len(cases) != len(rows) or len({case.get("id") for case in cases}) != len(rows) or {case.get("id") for case in cases} != set(lookup):
        raise ValueError("Evaluation must contain every dev row exactly once")
    rebuilt = []
    for case in cases:
        row = lookup[case["id"]]
        if any(not _same(case.get(key), row[source]) for key, source in (("group_id", "group_id"), ("variant", "variant"), ("observation", "observation"), ("expected", "target"))):
            raise ValueError("Evaluation case differs from public dev data or its target")
        latency = case.get("latency_ms")
        if not isinstance(case.get("raw"), str) or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Raw generation and finite nonnegative latency are required")
        rescored = score(row, case["raw"], latency)
        if any(not _same(case.get(key), rescored[key]) for key in ("prediction", "metrics", "error")):
            raise ValueError("Stored case score differs from original raw generation")
        rebuilt.append(rescored)
    summary = summarize(rebuilt)
    if not _same(summary, evaluation.get("summary")):
        raise ValueError("Stored summary differs from the original raw generations")
    total = Counter(case["expected"]["action"] for case in rebuilt)
    successes = Counter(case["expected"]["action"] for case in rebuilt if case["metrics"]["success"])
    by_action = {action: {"correct": successes[action], "total": total[action], "rate": successes[action] / total[action]} for action in ACTIONS}
    macro = sum((Fraction(successes[action], total[action]) for action in ACTIONS), Fraction()) / len(ACTIONS)
    emitted = summary["denominators"]["emitted_answers"]
    risks = {
        "over_answer": {"errors": sum(bool(case["prediction"] and case["prediction"]["action"] == "answer") for case in rebuilt if case["expected"]["action"] != "answer"), "total": summary["denominators"]["unanswerable"], "rate": summary["over_answer_rate"]},
        "unsupported_answer": {"errors": sum(case["metrics"]["unsupported_answer"] for case in rebuilt), "total": emitted, "rate": summary["unsupported_answer_rate"]},
    }
    return {"step": step, "source": _source(path), "model_provenance": provenance, "inference": {key: inference[key] for key in INFERENCE},
            "by_action": by_action, "macro_strict_success": float(macro), "macro_strict_success_fraction": str(macro),
            "risk_counts": risks, "recomputed_summary": summary,
            "training_elapsed_s_at_checkpoint": run["history"][step - 1]["elapsed_s"]}


def summarize_probe(data, baseline_run, probe_run, baseline_evaluations, probe_evaluations):
    rows = read_jsonl(data)
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("Capacity diagnostics may read only nonempty split=dev data")
    data_audit = audit(rows)
    if set(data_audit["actions"]) != set(ACTIONS):
        raise ValueError("Capacity diagnostics require all four gold actions")
    if set(baseline_evaluations) != set(STEPS) or set(probe_evaluations) != set(STEPS):
        raise ValueError("Provide both runs at exactly steps 200, 400, 600")
    runs = {"baseline": _run(baseline_run), "probe": _run(probe_run)}
    baseline, probe = runs["baseline"], runs["probe"]
    differences = {key: {"baseline": baseline["config"][key], "probe": probe["config"][key]} for key in baseline["config"] if not _same(baseline["config"][key], probe["config"][key])}
    if set(differences) != {"target_modules"} or set(baseline["config"]["target_modules"]) != BASE_MODULES or set(probe["config"]["target_modules"]) != PROBE_MODULES:
        raise ValueError("Capacity comparison must change only q/v to the seven declared target modules")
    if any(baseline["binding"][key] != probe["binding"][key] for key in ("data_sha256", "model_sha256")):
        raise ValueError("Capacity comparison requires identical training data and base snapshot")
    if baseline["config"]["max_steps"] != 600 or baseline["config"]["lora_rank"] != 8:
        raise ValueError("This capacity probe requires the matched 600-step, rank-8 budget")
    comparisons, inference = [], None
    for step in STEPS:
        pair = {"baseline": _evaluation(baseline_evaluations[step], rows, baseline, step),
                "probe": _evaluation(probe_evaluations[step], rows, probe, step)}
        for candidate in pair.values():
            if inference is None:
                inference = candidate["inference"]
            elif not _same(inference, candidate["inference"]):
                raise ValueError("All capacity evaluations must share identical inference settings")
        metrics = ("success_rate", "answerable_success", "action_macro_f1", "all_variants_correct_rate", "over_answer_rate", "unsupported_answer_rate")
        delta = {key: pair["probe"]["recomputed_summary"][key] - pair["baseline"]["recomputed_summary"][key] if pair["probe"]["recomputed_summary"][key] is not None and pair["baseline"]["recomputed_summary"][key] is not None else None for key in metrics}
        comparisons.append({"step": step, **pair, "probe_minus_baseline": delta})
    training = {name: {"sources": run["sources"], "binding": run["binding"], "config": run["config"],
                       **{key: run["summary"][key] for key in ("optimizer_steps", "trainable_parameters", "total_parameters", "elapsed_s", "first_loss", "last_loss")}} for name, run in runs.items()}
    return {"format": "evidence-gate-capacity-probe-v1", "status": "audited", "split": "dev", "test_used": False,
            "data": {**_source(data), "dataset_sha256": digest(rows), "rows": len(rows), "groups": len({row["group_id"] for row in rows})},
            "control": {"configuration_differences": differences, "same_training_data": True, "same_base_snapshot": True, "same_seed_and_schedule": True, "same_inference": True},
            "training": training, "comparisons": comparisons,
            "interpretation": "Exploratory mechanism comparison on already-seen V2 dev groups. A single matched seed cannot establish general improvement. This is not V3 selection, an unseen test, or final acceptance; full training budgets are disclosed separately from checkpoint elapsed time."}


def _mapping(values):
    result = {}
    for value in values:
        left, separator, path = value.partition("=")
        if not separator or not left.isdecimal() or not path or int(left) in result:
            raise ValueError("Use unique --*-evaluation STEP=path mappings")
        result[int(left)] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "baseline-run", "probe-run", "output"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("baseline-evaluation", "probe-evaluation"):
        parser.add_argument(f"--{name}", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize_probe(args.data, args.baseline_run, args.probe_run, _mapping(args.baseline_evaluation), _mapping(args.probe_evaluation))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "split": "dev", "test_used": False}))


if __name__ == "__main__":
    main()
