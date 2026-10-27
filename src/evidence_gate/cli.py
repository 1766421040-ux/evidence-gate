from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import audit, read_jsonl, save_dataset
from .evaluation import contract_rule, evaluate, first_match


def write_output(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="EvidenceGate — local evidence-decision SFT")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate grouped fictional data")
    gen.add_argument("--output", required=True)
    gen.add_argument("--seed", type=int, default=20261027)
    gen.add_argument("--train-groups", type=int, default=80)
    gen.add_argument("--dev-groups", type=int, default=12)
    gen.add_argument("--test-groups", type=int, default=20)
    check = commands.add_parser("audit", help="Check group leakage and observable boundaries")
    check.add_argument("files", nargs="+")
    train = commands.add_parser("train", help="Run genuine local PEFT LoRA SFT")
    train.add_argument("--data", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    ev = commands.add_parser("evaluate", help="Strictly score original model outputs")
    ev.add_argument("--data", required=True)
    ev.add_argument("--policy", choices=("contract", "first-match", "hf"), required=True)
    ev.add_argument("--name")
    ev.add_argument("--model")
    ev.add_argument("--adapter")
    ev.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    ev.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--max-new-tokens", type=int, default=72)
    ev.add_argument("--output", required=True)
    report = commands.add_parser("report", help="Combine matching evaluations into a replay report")
    report.add_argument("evaluations", nargs="+")
    report.add_argument("--training", nargs="*", default=[])
    report.add_argument("--output", required=True)
    serve = commands.add_parser("serve", help="Open a read-only local experiment viewer")
    serve.add_argument("--report", default="experiments/report.json")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    episode = commands.add_parser("replay", help="Run a bounded read-only retrieval episode")
    episode.add_argument("--data", required=True)
    episode.add_argument("--id", required=True)
    episode.add_argument("--model")
    episode.add_argument("--adapter")
    episode.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    episode.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    episode.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = save_dataset(args.output, args.seed, args.train_groups, args.dev_groups, args.test_groups)
    elif args.command == "audit":
        result = audit([row for file in args.files for row in read_jsonl(file)])
    elif args.command == "train":
        from .training import train as train_model
        result = train_model(args.data, args.model, args.output, args.config, resume=args.resume)
    elif args.command == "evaluate":
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        rows = read_jsonl(args.data)
        audit(rows)
        if any(row["split"] == "train" for row in rows):
            parser.error("Evaluation refuses training examples; use dev or locked test")
        if Path(args.output).exists():
            parser.error("Output already exists; choose a new path")
        if args.policy == "hf":
            if not args.model:
                parser.error("hf policy requires --model")
            from .hf_policy import HFPolicy
            policy = HFPolicy(args.model, args.adapter, args.device, args.max_new_tokens, dtype=args.dtype)
            try:
                result = evaluate(rows, policy, args.name or ("sft" if args.adapter else "base"), "model", args.batch_size)
                result["model_provenance"] = policy.provenance
            finally:
                policy.close()
            result["inference"] = {"decoding": "greedy free generation", "max_new_tokens": args.max_new_tokens,
                                    "batch_size": args.batch_size, "device": args.device,
                                    "dtype": args.dtype,
                                    "latency_definition": "batch wall time / batch size, not request p95",
                                    "adapter_used": bool(args.adapter), "output_repair": False}
        else:
            policy = contract_rule if args.policy == "contract" else first_match
            result = evaluate(rows, policy, args.name or args.policy)
        write_output(args.output, result)
        result = result["summary"]
    elif args.command == "report":
        methods = [json.loads(Path(p).read_text()) for p in args.evaluations]
        if len({m["dataset_sha256"] for m in methods}) != 1:
            parser.error("Cannot compare methods evaluated on different datasets")
        if len({m["name"] for m in methods}) != len(methods):
            parser.error("Method names must be unique")
        model_methods = [method for method in methods if method.get("kind") == "model"]
        if model_methods:
            # This report is a controlled base/adapter comparison, not a mixed
            # model leaderboard. Rule baselines have no inference signature.
            fields = ("decoding", "max_new_tokens", "dtype", "batch_size", "device", "output_repair")
            signatures = {
                tuple(method.get("inference", {}).get(field) for field in fields)
                for method in model_methods
            }
            bases = {method.get("model_provenance", {}).get("base_snapshot_sha256") for method in model_methods}
            if len(signatures) != 1 or any(value is None for signature in signatures for value in signature):
                parser.error("Model comparisons require identical recorded dtype, device, batch size and decoding settings")
            if len(bases) != 1 or None in bases:
                parser.error("Model comparisons require the same recorded base snapshot")
        training = []
        for path in args.training:
            item = json.loads(Path(path).read_text())
            if isinstance(item, list):
                summary_path = Path(path).with_name("summary.json")
                metadata = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                training.append({"name": Path(path).parent.name, "history": item, "metadata": metadata})
            elif "adapter_sha256" in item:
                training.append({"name": Path(path).parent.name, "metadata": item, "history": []})
            else:
                training.append(item)
        adapter_hashes = {m.get("model_provenance", {}).get("adapter_sha256") for m in methods}
        for item in training:
            sha = item.get("metadata", {}).get("adapter_sha256")
            if sha and sha not in adapter_hashes:
                parser.error("Training adapter hash does not match any evaluated adapter")
            expected_base = item.get("metadata", {}).get("binding", {}).get("model_sha256")
            if sha and expected_base:
                matching = [method for method in model_methods if method.get("model_provenance", {}).get("adapter_sha256") == sha]
                if any(method["model_provenance"]["base_snapshot_sha256"] != expected_base for method in matching):
                    parser.error("Training base snapshot does not match the evaluated adapter's base")
        model_names = sorted({m["model_provenance"]["model_name"] for m in methods if "model_provenance" in m})
        result = {"title": "EvidenceGate", "generated_at": datetime.now(timezone.utc).isoformat(),
                  "provenance": {"dataset": methods[0]["dataset_sha256"],
                                 "model": ", ".join(model_names) if model_names else "Rule baselines only",
                                 "device": ", ".join(sorted({m.get("inference", {}).get("device", "CPU rules") for m in methods})),
                                 "sources": [{"title": "Source code", "url": "https://github.com/1766421040-ux/evidence-gate"}],
                                 "limitations": ["Self-authored synthetic English tasks", "No production or broad language generalization claim", "Rules and model outcomes are reported separately"]},
                  "methods": methods, "training": training}
        write_output(args.output, result)
        result = {"report": args.output, "methods": [m["name"] for m in methods]}
    elif args.command == "replay":
        from .agent import run_episode
        rows = read_jsonl(args.data)
        audit(rows)
        if args.adapter and not args.model:
            parser.error("--adapter requires --model")
        if Path(args.output).exists():
            parser.error("Output already exists; choose a new path")
        row = next((r for r in rows if r["id"] == args.id), None)
        if row is None:
            parser.error("Unknown case id")
        if args.model:
            from .hf_policy import HFPolicy
            policy = HFPolicy(args.model, args.adapter, args.device, dtype=args.dtype)
            try:
                result = run_episode(row["observation"], row["gold"]["catalogue"], policy, kind="model")
                result["model_provenance"] = policy.provenance
                result["inference"] = {"dtype": args.dtype, "device": args.device,
                                       "decoding": "greedy free generation", "max_new_tokens": 72,
                                       "batch_size": 1, "output_repair": False}
            finally:
                policy.close()
        else:
            result = run_episode(row["observation"], row["gold"]["catalogue"], contract_rule)
        result.update({"id": row["id"], "kind": "model" if args.model else "rule"})
        write_output(args.output, result)
    else:
        from .server import serve
        serve(args.report, args.host, args.port)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
