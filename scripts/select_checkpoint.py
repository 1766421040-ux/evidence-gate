"""Select one of the predeclared 200/400/600 checkpoints using dev only.

The metric is the unweighted mean of strict task success within each of the four
gold actions, with earlier optimizer step breaking exact ties. No model is loaded
and no test file is read. Scores are recomputed from original raw generations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from evidence_gate.cli import write_output
from evidence_gate.data import audit, digest, read_jsonl
from evidence_gate.evaluation import ACTIONS, score
from evidence_gate.training import sha256_file

STEPS = (200, 400, 600)
INFERENCE_FIELDS = ("decoding", "max_new_tokens", "dtype", "batch_size", "device", "output_repair")
BINDING_FIELDS = ("base_snapshot_sha256", "training_data_sha256", "training_config_sha256", "adapter_config_sha256")


def public_path(path):
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else resolved.name


def select_checkpoint(data_path, evaluation_paths):
    """Validate saved dev evidence and return a publishable selection manifest."""
    rows = read_jsonl(data_path)
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("Checkpoint selection requires only nonempty split=dev data")
    audit(rows)
    if len(evaluation_paths) != len(STEPS):
        raise ValueError("Provide exactly the three predeclared checkpoint evaluations")
    dataset_sha = digest(rows)
    by_id = {row["id"]: row for row in rows}
    denominators = Counter(row["target"]["action"] for row in rows)
    if set(denominators) != set(ACTIONS):
        raise ValueError("Dev data must contain all four actions")
    candidates, signatures, bindings, seen_steps = [], set(), set(), set()
    for path in evaluation_paths:
        evaluation = json.loads(Path(path).read_text(encoding="utf-8"))
        if evaluation.get("kind") != "model" or evaluation.get("dataset_sha256") != dataset_sha:
            raise ValueError("Evaluation must be a model result bound to the exact dev dataset")
        provenance = evaluation.get("model_provenance", {})
        step = provenance.get("checkpoint_step")
        if type(step) is not int or step not in STEPS or step in seen_steps:
            raise ValueError("Evaluations must cover each of steps 200, 400, 600 exactly once")
        seen_steps.add(step)
        required = (*BINDING_FIELDS, "adapter_sha256", "checkpoint_sha256")
        if provenance.get("adapter_training_binding_verified") is not True or any(not provenance.get(key) for key in required):
            raise ValueError("Checkpoint evaluation lacks verified training provenance")
        bindings.add(tuple(provenance[key] for key in BINDING_FIELDS))
        inference = evaluation.get("inference", {})
        if any(inference.get(key) is None for key in INFERENCE_FIELDS):
            raise ValueError("Evaluation lacks controlled inference settings")
        if inference["decoding"] != "greedy free generation" or inference["output_repair"] is not False:
            raise ValueError("Selection requires unrepaired greedy free generation")
        signatures.add(tuple(inference[key] for key in INFERENCE_FIELDS))
        cases = evaluation.get("cases", [])
        if len(cases) != len(rows) or {case["id"] for case in cases} != set(by_id):
            raise ValueError("Evaluation must contain every dev example exactly once")
        successes = Counter({action: 0 for action in ACTIONS})
        for case in cases:
            row = by_id[case["id"]]
            for field in ("group_id", "variant", "observation"):
                if case.get(field) != row[field]:
                    raise ValueError(f"Evaluation case differs from dev data: {field}")
            if case.get("expected") != row["target"]:
                raise ValueError("Evaluation expected target differs from dev data")
            rescored = score(row, case.get("raw"))
            if rescored["metrics"]["success"]:
                successes[row["target"]["action"]] += 1
        macro = sum((Fraction(successes[action], denominators[action]) for action in ACTIONS), Fraction()) / len(ACTIONS)
        candidates.append({
            "step": step, "evaluation_file": public_path(path),
            "evaluation_file_sha256": sha256_file(Path(path)),
            "adapter_sha256": provenance["adapter_sha256"],
            "adapter_config_sha256": provenance["adapter_config_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "macro_strict_success": float(macro),
            "macro_strict_success_fraction": f"{macro.numerator}/{macro.denominator}",
            "strict_success": {"correct": sum(successes.values()), "total": len(rows), "rate": sum(successes.values()) / len(rows)},
            "by_action": {action: {"correct": successes[action], "total": denominators[action], "rate": successes[action] / denominators[action]} for action in ACTIONS},
        })
    if len(bindings) != 1 or len(signatures) != 1:
        raise ValueError("Checkpoints must share training data/config/base/adapter configuration and inference settings")
    candidates.sort(key=lambda item: item["step"])
    chosen = max(candidates, key=lambda item: (Fraction(item["macro_strict_success_fraction"]), -item["step"]))
    return {
        "format": "evidence-gate-dev-selection-v1",
        "selection_rule": "Highest mean strict task success over four gold actions; exact ties choose earlier optimizer step",
        "predeclared_steps": list(STEPS), "selection_split": "dev",
        "selection_data_file": public_path(data_path),
        "selection_data_file_sha256": sha256_file(Path(data_path)),
        "selection_dataset_sha256": dataset_sha,
        "selection_examples": len(rows), "selection_groups": len({row["group_id"] for row in rows}),
        "training_binding": dict(zip(BINDING_FIELDS, next(iter(bindings)))),
        "inference": dict(zip(INFERENCE_FIELDS, next(iter(signatures)))),
        "candidates": candidates,
        "selected_step": chosen["step"],
        "selected_adapter_sha256": chosen["adapter_sha256"],
        "selected_adapter_config_sha256": chosen["adapter_config_sha256"],
        "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
        "test_used_for_selection": False,
        "limitations": "Development metrics are used for selection and are not an unbiased final test estimate. No test data is read by this script.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs=3)
    parser.add_argument("--data", required=True, help="The exact dev JSONL used for all three evaluations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Output already exists; choose a fresh selection manifest")
    try:
        result = select_checkpoint(args.data, args.evaluations)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    write_output(args.output, result)
    print(json.dumps({key: result[key] for key in ("selected_step", "selected_adapter_sha256", "selection_dataset_sha256")}, indent=2))


if __name__ == "__main__":
    main()
