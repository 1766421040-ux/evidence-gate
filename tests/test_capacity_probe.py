"""The capacity report must establish matched controls from saved evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import canonical, digest, generate_group, write_jsonl
from evidence_gate.evaluation import score, summarize
from evidence_gate.training import _json_bytes, normalize_config, sha256_file

spec = importlib.util.spec_from_file_location("capacity_probe", Path(__file__).resolve().parents[1] / "scripts/summarize_capacity_probe.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def probe(tmp_path):
    rows = generate_group(0, "dev", 42) + generate_group(1, "dev", 42)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, rows)
    paths, evaluations = {}, {}
    for name, modules in (("baseline", capacity.BASE_MODULES), ("probe", capacity.PROBE_MODULES)):
        root = tmp_path / name
        paths[name] = root
        config = normalize_config({"max_steps": 600, "lora_rank": 8, "target_modules": sorted(modules), "device": "cpu"})
        model_files = {"model.safetensors": "a" * 64}
        binding = {"data_sha256": "d" * 64, "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(), "model_sha256": hashlib.sha256(_json_bytes(model_files)).hexdigest()}
        history = [{"step": step, "loss": 1 / step, "elapsed_s": step * (2 if name == "probe" else 1)} for step in range(1, 601)]
        manifest = {"binding": binding, "config": config, "model_files": model_files}
        summary = {"binding": binding, "optimizer_steps": 600, "elapsed_s": history[-1]["elapsed_s"], "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"], "trainable_parameters": 4000 if name == "probe" else 500, "total_parameters": 10000}
        write(root / "run_manifest.json", manifest)
        write(root / "summary.json", summary)
        write(root / "metrics.json", history)
        evaluations[name] = {}
        for step in capacity.STEPS:
            checkpoint = root / "checkpoints" / f"step-{step:06d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_model.safetensors").write_bytes(f"hash-only-{name}-{step}".encode())
            write(checkpoint / "adapter_config.json", {"r": 8, "target_modules": sorted(modules)})
            files = {filename: sha256_file(checkpoint / filename) for filename in ("adapter_model.safetensors", "adapter_config.json")}
            write(checkpoint / "checkpoint.json", {"binding": binding, "step": step, "files": files, "history": history[:step]})
            cases = []
            for row in rows:
                target = deepcopy(row["target"])
                if name == "baseline" and row["variant"] == "supported":
                    target = {"action": "abstain", "value": "", "citations": []}
                cases.append(score(row, canonical(target), 2.0))
            evaluation = {"kind": "model", "cases": cases, "summary": summarize(cases), "dataset_sha256": digest(rows),
                          "inference": {"decoding": "greedy free generation", "dtype": "float32", "device": "cpu", "max_new_tokens": 72, "batch_size": 4, "adapter_used": True, "output_repair": False},
                          "model_provenance": {"adapter_training_binding_verified": True, "checkpoint_step": step,
                                               "training_data_sha256": binding["data_sha256"], "training_config_sha256": binding["config_sha256"], "base_snapshot_sha256": binding["model_sha256"],
                                               "adapter_sha256": files["adapter_model.safetensors"], "adapter_config_sha256": files["adapter_config.json"], "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json")}}
            result = tmp_path / f"{name}-{step}.json"
            write(result, evaluation)
            evaluations[name][step] = result
    return {"data": data, "rows": rows, "paths": paths, "evaluations": evaluations}


def run(e):
    return capacity.summarize_probe(e["data"], e["paths"]["baseline"], e["paths"]["probe"], e["evaluations"]["baseline"], e["evaluations"]["probe"])


def test_matched_capacity_report_discloses_action_risk_counts_and_full_budget(probe):
    result = run(probe)
    assert result["status"] == "audited" and result["test_used"] is False
    assert set(result["control"]["configuration_differences"]) == {"target_modules"}
    assert result["training"]["probe"]["elapsed_s"] == 1200
    assert result["comparisons"][0]["probe"]["training_elapsed_s_at_checkpoint"] == 400
    for pair in result["comparisons"]:
        assert pair["baseline"]["by_action"]["answer"] == {"correct": 6, "total": 8, "rate": .75}
        assert pair["probe"]["by_action"]["answer"] == {"correct": 8, "total": 8, "rate": 1.0}
        assert pair["probe"]["risk_counts"]["unsupported_answer"] == {"errors": 0, "total": 8, "rate": 0}
        assert pair["probe_minus_baseline"]["answerable_success"] == .25


@pytest.mark.parametrize("mutation,match", [("raw", "Stored case score"), ("summary", "Stored summary"), ("inference", "identical inference"), ("binding", "declared training run")])
def test_raw_and_inference_integrity_are_not_trusted(probe, mutation, match):
    path = probe["evaluations"]["probe"][200]
    value = json.loads(path.read_text())
    if mutation == "raw": value["cases"][0]["raw"] = "invalid"
    if mutation == "summary": value["summary"]["answerable_success"] = .123
    if mutation == "inference": value["inference"]["batch_size"] = 1
    if mutation == "binding": value["model_provenance"]["training_config_sha256"] = "0" * 64
    write(path, value)
    with pytest.raises(ValueError, match=match):
        run(probe)


def test_no_test_data_is_accepted(probe):
    for row in probe["rows"]:
        row["split"] = "test"
    write_jsonl(probe["data"], probe["rows"])
    with pytest.raises(ValueError, match="only nonempty split=dev"):
        run(probe)


def test_configuration_change_cannot_hide_behind_old_hash(probe):
    path = probe["paths"]["probe"] / "run_manifest.json"
    value = json.loads(path.read_text())
    value["config"]["learning_rate"] *= 2
    write(path, value)
    with pytest.raises(ValueError, match="not bound"):
        run(probe)


def test_completed_run_must_include_complete_metrics(probe):
    path = probe["paths"]["probe"] / "metrics.json"
    write(path, json.loads(path.read_text())[:-1])
    with pytest.raises(ValueError, match="metrics are incomplete"):
        run(probe)


def test_adapter_weights_are_checked_against_generation_receipt(probe):
    path = probe["paths"]["probe"] / "checkpoints/step-000200/adapter_model.safetensors"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        run(probe)
