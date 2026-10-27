"""CPU review of the published diagnostic before comparing a new adapter."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnostic_comparison", ROOT / "scripts/compare_policy_diagnostics.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
BASELINE = ROOT / "experiments/v3/diagnostics/v2-selected-64.json"


def run(probe):
    return comparison.compare(BASELINE, probe, ROOT / "data/v2/train.jsonl", ROOT / "data/v2/dev.jsonl")


def test_identity_comparison_reaudits_real_diagnostic_without_model():
    result = run(BASELINE)
    assert result["exploratory"] and result["test_read"] is False
    for split in ("train", "dev"):
        assert len(result["splits"][split]["selected_ids"]) == 32
        for action in result["splits"][split]["by_action"].values():
            assert action["correct_count_delta"] == action["paired_improved"] == action["paired_regressed"] == 0
    assert result["splits"]["dev"]["by_action"]["answer"]["baseline_correct"] == 1


@pytest.mark.parametrize("mutation,match", [("ids", "sampling identity"), ("raw", "raw score"), ("token", "recorded tokens"), ("inference", "identical sampling and inference")])
def test_paired_evidence_changes_fail(tmp_path, mutation, match):
    value = json.loads(BASELINE.read_text())
    if mutation == "ids": value["splits"]["train"]["selected_ids"][0] = "different"
    if mutation == "raw": value["splits"]["dev"]["free_generation"]["cases"][0]["raw"] = "invalid"
    if mutation == "token": value["splits"]["dev"]["teacher_forced"]["cases"][0]["tokens"][0]["nll"] = 100
    if mutation == "inference": value["inference"]["batch_size"] = 1
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        run(path)
