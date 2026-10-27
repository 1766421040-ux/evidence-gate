"""Checkpoint choice is dev-only, provenance-bound and rescored from raw output."""
import importlib.util
import json
from pathlib import Path

import pytest

from evidence_gate.data import canonical, digest, generate_group, write_jsonl
from evidence_gate.evaluation import score
from evidence_gate.training import sha256_file

spec = importlib.util.spec_from_file_location("select_checkpoint", Path(__file__).resolve().parents[1] / "scripts" / "select_checkpoint.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


@pytest.fixture
def saved_development(tmp_path):
    rows = generate_group(7, "dev", 51)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, rows)
    paths = []
    first_search = next(row["id"] for row in rows if row["target"]["action"] == "search")
    for step in (200, 400, 600):
        correct = {row["id"] for row in rows if row["target"]["action"] in (("search", "abstain") if step == 200 else ("answer", "clarify"))}
        if step != 200:
            correct.add(first_search)
        cases = [score(row, canonical(row["target"]) if row["id"] in correct else "invalid output") for row in rows]
        result = {
            "name": f"step-{step}", "kind": "model", "dataset_sha256": digest(rows),
            "summary": {"success_rate": 1.0},  # Deliberately false: never trust a supplied aggregate.
            "cases": cases,
            "model_provenance": {
                "base_snapshot_sha256": "base", "training_data_sha256": "training-data",
                "training_config_sha256": "training-config", "adapter_config_sha256": "lora-config",
                "adapter_sha256": f"weights-{step}", "checkpoint_sha256": f"checkpoint-{step}",
                "checkpoint_step": step, "adapter_training_binding_verified": True,
            },
            "inference": {"decoding": "greedy free generation", "max_new_tokens": 72, "dtype": "bfloat16", "device": "mps", "batch_size": 4, "output_repair": False},
        }
        path = tmp_path / f"dev-{step}.json"
        path.write_text(json.dumps(result))
        paths.append(path)
    return data, paths


def test_macro_strict_success_uses_raw_and_earlier_tie_break(saved_development):
    data, paths = saved_development
    before = data.read_bytes()
    result = selection.select_checkpoint(data, list(reversed(paths)))
    assert result["selected_step"] == 400
    assert result["selected_adapter_sha256"] == "weights-400"
    assert result["selection_data_file_sha256"] == sha256_file(data)
    assert result["test_used_for_selection"] is False
    early, chosen, tied = result["candidates"]
    assert early["strict_success"]["correct"] == 9
    assert chosen["strict_success"]["correct"] == 6
    assert early["macro_strict_success"] == .5
    assert chosen["macro_strict_success"] == tied["macro_strict_success"] == .55
    assert chosen["macro_strict_success_fraction"] == "11/20"
    assert chosen["by_action"]["clarify"] == {"correct": 1, "total": 1, "rate": 1.0}
    assert data.read_bytes() == before


@pytest.mark.parametrize("split", ["train", "test"])
def test_selection_rejects_non_dev_data(saved_development, split):
    data, paths = saved_development
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    for row in rows:
        row["split"] = split
    write_jsonl(data, rows)
    with pytest.raises(ValueError, match="split=dev"):
        selection.select_checkpoint(data, paths)


@pytest.mark.parametrize("change,match", [
    ("dataset", "exact dev dataset"), ("missing", "every dev example"),
    ("duplicate", "every dev example"), ("expected", "expected target"),
    ("observation", "differs from dev data"), ("dtype", "share training"),
    ("training", "share training"), ("step", "exactly once"),
    ("unverified", "verified training provenance"),
])
def test_rejects_mismatched_or_incomplete_evidence(saved_development, change, match):
    data, paths = saved_development
    result = json.loads(paths[1].read_text())
    if change == "dataset": result["dataset_sha256"] = "wrong"
    elif change == "missing": result["cases"].pop()
    elif change == "duplicate": result["cases"][-1] = result["cases"][0]
    elif change == "expected": result["cases"][0]["expected"]["value"] = "999 days"
    elif change == "observation": result["cases"][0]["observation"]["budget"] = 999
    elif change == "dtype": result["inference"]["dtype"] = "float32"
    elif change == "training": result["model_provenance"]["training_data_sha256"] = "other training"
    elif change == "step": result["model_provenance"]["checkpoint_step"] = 200
    elif change == "unverified": result["model_provenance"]["adapter_training_binding_verified"] = False
    paths[1].write_text(json.dumps(result))
    with pytest.raises(ValueError, match=match):
        selection.select_checkpoint(data, paths)


def test_selection_cli_never_overwrites(saved_development, tmp_path):
    data, paths = saved_development
    output = tmp_path / "selection.json"
    arguments = [*(str(path) for path in paths), "--data", str(data), "--output", str(output)]
    selection.main(arguments)
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        selection.main(arguments)
    assert error.value.code == 2
    assert output.read_bytes() == before
