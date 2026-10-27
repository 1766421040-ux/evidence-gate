"""Strict extra V3.1 topic/agreement diagnostics from existing raw outputs.

This does not generate predictions, select a model, authorize test access, or
alter the primary summary/gates. Explicitly supply an existing evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from evidence_gate.data import canonical, digest, read_jsonl
from evidence_gate.evaluation import score, summarize


PAIRS = {"topic_filter": ("conflict", "irrelevant_topic"),
         "conflict_agreement": ("conflict", "consensus")}


def scope_pairs(rows, evaluation, protocol):
    if not rows or len({row["split"] for row in rows}) != 1:
        raise ValueError("Provide exactly one complete dev or test split")
    split = rows[0]["split"]
    if split not in ("dev", "test"):
        raise ValueError("Scope-pair reporting is restricted to explicit dev/test evaluations")
    if evaluation.get("dataset_sha256") != digest(rows):
        raise ValueError("Evaluation does not match the supplied data")
    groups = defaultdict(dict)
    lookup = {row["id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate data row identity")
    variants = Counter(protocol["variants"])
    if len(variants) != 24 or any(value != 1 for value in variants.values()):
        raise ValueError("Scope diagnostics require the declared twenty-four-cell protocol")
    for row in rows:
        groups[row["group_id"]][row["variant"]] = row
    if len(groups) != protocol[f"{split}_groups"]:
        raise ValueError("Incomplete frozen split group count")
    for group in groups:
        if Counter(row["variant"] for row in rows if row["group_id"] == group) != variants:
            raise ValueError("Incomplete or duplicate per-group variant matrix")
    original = evaluation.get("cases")
    if not isinstance(original, list) or len(original) != len(rows):
        raise ValueError("Incomplete evaluation cases")
    ids = [case.get("id") for case in original]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("Evaluation identities are duplicate, missing or unexpected")
    rebuilt = []
    for case in original:
        row = lookup[case["id"]]
        for key, value in (("group_id", row["group_id"]), ("variant", row["variant"]),
                           ("observation", row["observation"]), ("expected", row["target"])):
            if canonical(case.get(key)) != canonical(value):
                raise ValueError(f"Evaluation contains altered {key}")
        latency = case.get("latency_ms")
        if not isinstance(case.get("raw"), str) or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Original raw output and finite nonnegative latency are required")
        rescored = score(row, case["raw"], latency)
        if any(canonical(case.get(key)) != canonical(rescored[key]) for key in ("prediction", "metrics", "error")):
            raise ValueError("Stored score differs from the original raw output")
        rebuilt.append(rescored)
    if canonical(evaluation.get("summary")) != canonical(summarize(rebuilt)):
        raise ValueError("Stored main summary differs from original raw scoring")
    rescored_lookup = {case["id"]: case for case in rebuilt}
    results = {}
    for name, (left, right) in PAIRS.items():
        matched = []
        for group_id, members in sorted(groups.items()):
            for budget in (0, 1):
                a, b = (members[f"{semantic}_b{budget}"] for semantic in (left, right))
                if (a["target"]["action"], b["target"]["action"]) != ("search" if budget else "abstain", "answer"):
                    raise ValueError("Scope pair lacks the declared expected action transition")
                ca, cb = (rescored_lookup[row["id"]] for row in (a, b))
                matched.append({"group_id": group_id, "budget": budget, "left_id": a["id"], "right_id": b["id"],
                                "both_strictly_correct": ca["metrics"]["success"] and cb["metrics"]["success"],
                                "both_actions_correct": ca["metrics"]["action_correct"] and cb["metrics"]["action_correct"]})
        def aggregate(cases):
            correct = sum(case["both_strictly_correct"] for case in cases)
            return {"correct": correct, "total": len(cases), "accuracy": correct / len(cases),
                    "action_pair_accuracy": sum(case["both_actions_correct"] for case in cases) / len(cases)}
        results[name] = {**aggregate(matched),
                         "by_budget": {str(budget): aggregate([case for case in matched if case["budget"] == budget]) for budget in (0, 1)},
                         "cases": matched}
    return {"format": "evidence-gate-v31-scope-pair-diagnostic-v1", "name": evaluation.get("name"),
            "split": split, "dataset_sha256": digest(rows), "groups": len(groups), "pairs": results,
            "raw_outputs_rescored": True, "primary_summary_unchanged": True,
            "model_calls": 0, "used_for_selection": False,
            "scope": "Additional matched-budget pair diagnostics. Both members must have exact action/value/citations; correlated pairs are not independent samples. This report never changes the seven primary gates."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        sources = {key: Path(getattr(args, key)) for key in ("data", "evaluation", "protocol")}
        snapshots = {key: path.read_bytes() for key, path in sources.items()}
        protocol = json.loads(snapshots["protocol"])
        evaluation = json.loads(snapshots["evaluation"])
        rows = [json.loads(line) for line in snapshots["data"].decode("utf-8").splitlines() if line.strip()]
        split = rows[0]["split"] if rows else None
        if hashlib.sha256(snapshots["data"]).hexdigest() != protocol["data_sha256"].get(split):
            raise ValueError("Data file differs from the supplied frozen protocol SHA256")
        report = scope_pairs(rows, evaluation, protocol)
        report["sources"] = {key: {"name": sources[key].name, "sha256": hashlib.sha256(value).hexdigest()} for key, value in snapshots.items()}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({name: {key: values[key] for key in ("correct", "total", "accuracy")} for name, values in report["pairs"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
