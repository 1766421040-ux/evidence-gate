"""Run frozen wording/evidence diagnostics and fully scored retrieval episodes.

The first four held-out groups are a diagnostic subset, not new test groups.
No cloud service is contacted. Existing output is never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evidence_gate.agent import run_episode
from evidence_gate.cli import write_output
from evidence_gate.data import digest, read_jsonl
from evidence_gate.evaluation import contract_rule, evaluate, parse_prediction
from evidence_gate.hf_policy import HFPolicy


def score_episode(row, policy):
    episode = run_episode(row["observation"], row["gold"]["catalogue"], policy, kind="model")
    # A correct final guess is insufficient: every decision must be grounded in
    # the evidence actually visible at that step. The oracle reads no hidden gold.
    for event in episode["trace"]:
        expected = contract_rule(event["observation"])
        event["expected"] = expected
        try:
            predicted = parse_prediction(event["raw"])
            event["correct"] = (predicted["action"] == expected["action"]
                                and predicted["value"].strip().casefold() == expected["value"].casefold()
                                and set(predicted["citations"]) == set(expected["citations"]))
        except ValueError:
            event["correct"] = False
    episode["success"] = episode["status"] == "completed" and all(e["correct"] for e in episode["trace"])
    return {"id": row["id"], "group_id": row["group_id"], "variant": row["variant"], **episode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--answer-adapter", default="artifacts/answer-only-lora")
    parser.add_argument("--evidence-adapter", default="artifacts/evidence-lora")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32", "float16"))
    parser.add_argument("--output", default="experiments/diagnostics")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a fresh directory")
    output.mkdir(parents=True)
    test = read_jsonl("data/main/test.jsonl")
    groups = sorted({r["group_id"] for r in test})[:4]
    episodes = [r for r in test if r["group_id"] in groups]
    for name, adapter in (("base", None), ("answer-only-sft", args.answer_adapter), ("evidence-sft", args.evidence_adapter)):
        policy = HFPolicy(args.model, adapter, args.device, 72, dtype=args.dtype)
        inference = {"decoding": "greedy free generation", "max_new_tokens": 72, "device": args.device, "dtype": args.dtype, "output_repair": False}
        try:
            for diagnostic in ("paraphrase", "redacted"):
                rows = read_jsonl(f"data/diagnostics/{diagnostic}.jsonl")
                result = evaluate(rows, policy, name, "model", batch_size=4)
                result.update(model_provenance=policy.provenance, inference={**inference, "batch_size": 4}, diagnostic=diagnostic)
                write_output(output / f"{name}-{diagnostic}.json", result)
            cases = []
            for row in episodes:
                cases.append(score_episode(row, policy))
                print(f"{name} episodes: {len(cases)}/{len(episodes)}", flush=True)
            result = {"name": name, "groups": groups, "dataset_sha256": digest(episodes),
                      "model_provenance": policy.provenance, "inference": {**inference, "batch_size": 1},
                      "success_definition": "Legal terminal state and every step matches the public-evidence oracle; completion alone is not success",
                      "limitations": "First four main-test groups; not independent extra test data. Fixed-grammar oracle.",
                      "summary": {"n": len(cases), "successes": sum(c["success"] for c in cases),
                                  "success_rate": sum(c["success"] for c in cases) / len(cases),
                                  "completed": sum(c["status"] == "completed" for c in cases),
                                  "searches": sum(c["searches"] for c in cases)}, "cases": cases}
            write_output(output / f"{name}-episodes.json", result)
        finally:
            policy.close()


if __name__ == "__main__":
    main()
