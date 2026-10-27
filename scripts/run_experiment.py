"""Reproduce the two local SFT runs and locked evaluations, without cloud APIs."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/main.json")
    parser.add_argument("--output", default="runs/reproduction")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        parser.error("Output already exists. Choose a new run directory.")
    root.mkdir(parents=True)
    config = json.loads(Path(args.config).read_text())
    config["device"] = args.device
    config_file = root / "config.json"
    config_file.write_text(json.dumps(config, indent=2))
    def run(*command):
        subprocess.run([sys.executable, "-m", "evidence_gate.cli", *map(str, command)], check=True)
    run("generate", "--output", root / "data")
    for name, data in (("answer-only", "train_answer_only.jsonl"), ("evidence", "train.jsonl")):
        run("train", "--data", root / "data" / data, "--model", args.model,
            "--config", config_file, "--output", root / name)
    eval_files = []
    for name in ("base", "answer-only", "evidence"):
        output = root / f"{name}-eval.json"
        command = ["evaluate", "--data", root / "data/test.jsonl", "--policy", "hf", "--model", args.model,
                   "--device", args.device, "--dtype", config["dtype"], "--batch-size", "4", "--name", name, "--output", output]
        if name != "base":
            command += ["--adapter", root / name / "adapter"]
        run(*command)
        eval_files.append(output)
    for policy in ("contract", "first-match"):
        output = root / f"{policy}-eval.json"
        run("evaluate", "--data", root / "data/test.jsonl", "--policy", policy, "--output", output)
        eval_files.append(output)
    run("report", *eval_files, "--output", root / "report.json",
        "--training", root / "answer-only/metrics.json", root / "evidence/metrics.json")


if __name__ == "__main__":
    main()
