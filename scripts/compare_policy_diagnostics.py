"""Pair two saved train/dev diagnostics without loading a model or using test.

Reports observed changes and discordant pairs, not a significance claim: the
balanced 64-case exploration was designed to diagnose a mechanism, not confirm
an improvement. Conditional token likelihood is kept separate from generation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

from evidence_gate.data import audit, digest, read_jsonl
from evidence_gate.evaluation import ACTIONS, score, summarize
from evidence_gate.training import sha256_file


def _module(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capacity = _module("summarize_capacity_probe.py")
diagnostic = _module("diagnose_policy.py")


def _audit(record, sources, paths):
    if record.get("format") != "evidence-gate-exploratory-diagnostic-v1" or record.get("test_read") is not False or record.get("exploratory") is not True:
        raise ValueError("Only exploratory train/dev diagnostics are supported")
    if set(record["splits"]) != {"train", "dev"}:
        raise ValueError("Diagnostics may contain only train and dev splits")
    provenance = record["model_provenance"]
    if provenance.get("adapter_training_binding_verified") is not True or provenance.get("training_data_sha256") != sha256_file(paths["train"]):
        raise ValueError("Diagnostic train data differs from the model training binding")
    samples = {}
    for split, rows in sources.items():
        selected = diagnostic.sample_rows(rows, record["sampling"]["per_action_per_split"], record["sampling"]["seed"])
        samples[split] = selected
        part = record["splits"][split]
        expected = {"source_file_sha256": sha256_file(paths[split]), "source_dataset_sha256": digest(rows), "selected_dataset_sha256": digest(selected),
                    "selected_ids": [row["id"] for row in selected], "selected_groups": sorted({row["group_id"] for row in selected})}
        if any(not capacity._same(part.get(key), value) for key, value in expected.items()):
            raise ValueError("Diagnostic sampling identity differs from the original source data")
        free = part["free_generation"]
        if free.get("kind") != "model" or free.get("dataset_sha256") != digest(selected):
            raise ValueError("Diagnostic free generation is not bound to the selected rows")
        if [case["id"] for case in free["cases"]] != part["selected_ids"]:
            raise ValueError("Diagnostic free-generation rows must preserve the selected order")
        rebuilt = [score(row, case["raw"], case["latency_ms"]) for row, case in zip(selected, free["cases"])]
        for actual, stored in zip(rebuilt, free["cases"]):
            if any(not capacity._same(actual[key], stored.get(key)) for key in actual):
                raise ValueError("Diagnostic raw score or row identity was altered")
        if not capacity._same(summarize(rebuilt), free["summary"]):
            raise ValueError("Diagnostic free-generation summary differs from raw scoring")
        teacher = part["teacher_forced"]
        if [case["id"] for case in teacher["cases"]] != part["selected_ids"]:
            raise ValueError("Teacher forcing is not paired with the same selected rows")
        for row, case in zip(selected, teacher["cases"]):
            if case["expected_action"] != row["target"]["action"] or case["group_id"] != row["group_id"] or case["variant"] != row["variant"]:
                raise ValueError("Teacher-forced case identity differs from source data")
            tokens = case["tokens"]
            if not tokens or any(type(token["nll"]) not in (int, float) or not math.isfinite(token["nll"]) or token["nll"] < 0 for token in tokens):
                raise ValueError("Teacher-forced token losses must be finite and nonnegative")
            mean = math.fsum(token["nll"] for token in tokens) / len(tokens)
            if case["completion_tokens"] != len(tokens) or abs(mean - case["manual_completion_nll"]) > 1e-5 or abs(case["hf_loss"] - mean) > 1e-5:
                raise ValueError("Teacher-forced losses are not consistent with the recorded tokens")
        if not capacity._same(diagnostic.summarize_teacher(teacher["cases"]), teacher["summary"]):
            raise ValueError("Teacher-forced summary differs from its tokens")
    return samples


def _token_stats(cases):
    tokens = [token for case in cases for token in case["tokens"]]
    first = [next(token for token in case["tokens"] if token["field"] == "action") for case in cases]
    total_nll = math.fsum(token["nll"] for token in tokens)
    return {"examples": len(cases), "completion_token_mean_nll": total_nll / len(tokens),
            "first_action_mean_nll": math.fsum(token["nll"] for token in first) / len(first),
            "first_action_top1_correct": sum(token["top1_correct"] for token in first),
            "first_action_share_of_residual_nll": math.fsum(token["nll"] for token in first) / total_nll if total_nll else None}


def compare(baseline, probe, train, dev):
    paths = {"train": train, "dev": dev}
    sources = {split: read_jsonl(path) for split, path in paths.items()}
    if any(not rows or any(row.get("split") != split for row in rows) for split, rows in sources.items()):
        raise ValueError("Only separately identified train and dev source files are allowed")
    audit(sources["train"] + sources["dev"])
    records = {"baseline": capacity._read(baseline), "probe": capacity._read(probe)}
    for record in records.values():
        _audit(record, sources, paths)
    left, right = records["baseline"], records["probe"]
    for key in ("sampling", "inference"):
        if not capacity._same(left[key], right[key]):
            raise ValueError("Paired diagnostics require identical sampling and inference")
    for key in ("base_snapshot_sha256", "training_data_sha256", "checkpoint_step"):
        if not capacity._same(left["model_provenance"][key], right["model_provenance"][key]):
            raise ValueError("Paired diagnostics require the same base, training data and optimizer step")
    splits = {}
    for split in ("train", "dev"):
        parts = {name: record["splits"][split] for name, record in records.items()}
        for lcase, rcase in zip(parts["baseline"]["teacher_forced"]["cases"], parts["probe"]["teacher_forced"]["cases"]):
            token_signature = lambda case: [{key: token[key] for key in ("position", "token_id", "field")} for token in case["tokens"]]
            if not capacity._same(token_signature(lcase), token_signature(rcase)):
                raise ValueError("Paired teacher-forced token masks differ")
        by_action = {}
        for action in ACTIONS:
            free = {name: [case for case in part["free_generation"]["cases"] if case["expected"]["action"] == action] for name, part in parts.items()}
            teacher = {name: [case for case in part["teacher_forced"]["cases"] if case["expected_action"] == action] for name, part in parts.items()}
            before = [case["metrics"]["success"] for case in free["baseline"]]
            after = [case["metrics"]["success"] for case in free["probe"]]
            by_action[action] = {"examples": len(before), "baseline_correct": sum(before), "probe_correct": sum(after),
                                 "correct_count_delta": sum(after) - sum(before),
                                 "paired_improved": sum(not l and r for l, r in zip(before, after)), "paired_regressed": sum(l and not r for l, r in zip(before, after)),
                                 "teacher_forced": {name: _token_stats(cases) for name, cases in teacher.items()}}
        splits[split] = {"selected_ids": parts["baseline"]["selected_ids"], "by_action": by_action,
                         "free_generation": {name: part["free_generation"]["summary"] for name, part in parts.items()},
                         "teacher_forced": {name: _token_stats(part["teacher_forced"]["cases"]) for name, part in parts.items()}}
    return {"format": "evidence-gate-paired-diagnostic-comparison-v1", "exploratory": True, "test_read": False,
            "sources": {"baseline": capacity._source(baseline), "probe": capacity._source(probe)},
            "model_provenance": {name: record["model_provenance"] for name, record in records.items()},
            "sampling": left["sampling"], "inference": left["inference"], "splits": splits,
            "interpretation": "Observed paired changes on a small balanced diagnostic sample; no confirmatory statistical-significance claim. NLL contribution is not gradient contribution. Teacher-forced token likelihood conditions on the correct prefix and cannot substitute for free generation. Compare complete old-dev results and frozen V3 dev gates separately."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "probe", "train", "dev", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(args.baseline, args.probe, args.train, args.dev)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({split: {action: values["correct_count_delta"] for action, values in result["splits"][split]["by_action"].items()} for split in ("train", "dev")}))


if __name__ == "__main__":
    main()
