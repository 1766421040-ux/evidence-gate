"""Recompute released scores and verify data/adapter identities without ML deps."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from evidence_gate.data import audit, canonical, digest, generate, read_jsonl
from evidence_gate.evaluation import score, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_script(root, name):
    spec = importlib.util.spec_from_file_location(name, root / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_same(actual, expected, message):
    # Avoid Python's True == 1 accepting changes to recorded JSON evidence.
    assert canonical(actual) == canonical(expected), message


def release_path(root, recorded):
    """Resolve a published repository-relative reference, never an external file."""
    assert isinstance(recorded, str) and recorded, "Missing release file reference"
    relative = Path(recorded)
    assert not relative.is_absolute(), "Release file references must be repository-relative"
    path = (root / relative).resolve()
    assert path.is_relative_to(root.resolve()), "Invalid release file reference"
    return path


def verify_v2_release(root):
    """Recompute a completed V2 release; an unfinished release stays pending.

    No model is loaded. The two audit manifests are the completion boundary:
    once both exist, missing or inconsistent evidence is an error, not pending.
    """
    root = Path(root).resolve()
    folder = root / "experiments/v2"
    selection_path, gate_path = folder / "selection.json", folder / "quality-gate.json"
    present = [path.name for path in (selection_path, gate_path) if path.is_file()]
    if len(present) != 2:
        return {"status": "pending", "audit_manifests_present": present}

    selection = json.loads(selection_path.read_text())
    dev_path, test_path = root / "data/v2/dev.jsonl", root / "data/v2/test.jsonl"
    assert release_path(root, selection["selection_data_file"]) == dev_path, "Selection must use released V2 dev data"
    candidates = [release_path(root, item["evaluation_file"]) for item in selection["candidates"]]
    selector = load_script(root, "select_checkpoint")
    rebuilt_selection = selector.select_checkpoint(dev_path, candidates)
    assert_same(selection, rebuilt_selection, "Stored V2 selection differs from recomputed dev selection")

    selected_path = folder / "selected-v2.json"
    protocol_path = folder / "protocol.json"
    gate = load_script(root, "quality_gate")
    rebuilt_gate = gate.audit_gate(selected_path, test_path, protocol_path)
    assert_same(json.loads(gate_path.read_text()), rebuilt_gate,
                "Stored V2 quality gate differs from recomputed raw-output audit")

    selected = json.loads(selected_path.read_text())
    assert selected.get("kind") == "model", "Selected V2 result must be a model evaluation"
    provenance = selected["model_provenance"]
    assert provenance.get("adapter_training_binding_verified") is True, "Selected V2 training binding is unverified"
    expected_fingerprints = {
        "adapter_sha256": selection["selected_adapter_sha256"],
        "adapter_config_sha256": selection["selected_adapter_config_sha256"],
        "checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "checkpoint_step": selection["selected_step"],
        **selection["training_binding"],
    }
    for field, expected in expected_fingerprints.items():
        assert_same(provenance.get(field), expected, f"Selected V2 {field} differs from dev selection")
    assert_same({key: selected.get("inference", {}).get(key) for key in selection["inference"]},
                selection["inference"], "Selected V2 test inference differs from dev selection")

    adapter = root / "artifacts/evidence-v2-lora"
    for filename, field in (("adapter_model.safetensors", "selected_adapter_sha256"),
                            ("adapter_config.json", "selected_adapter_config_sha256"),
                            ("checkpoint.json", "selected_checkpoint_sha256")):
        assert sha(adapter / filename) == selection[field], f"Released V2 {filename} differs from dev selection"
    checkpoint = json.loads((adapter / "checkpoint.json").read_text())
    step = selection["selected_step"]
    assert checkpoint.get("format") == "evidence-gate-checkpoint-v1", "Invalid released V2 checkpoint format"
    assert_same(checkpoint.get("step"), step, "Released V2 checkpoint has the wrong step")
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        assert checkpoint["files"][filename] == sha(adapter / filename), "V2 checkpoint file binding mismatch"

    report = json.loads((folder / "report.json").read_text())
    matching_methods = [item for item in report["methods"] if item.get("name") == selected["name"]]
    assert len(matching_methods) == 1, "Report must contain exactly one selected V2 method"
    assert_same(matching_methods[0], selected, "Report selected V2 result differs from its evaluation file")
    training = json.loads((folder / "selected-training.json").read_text())
    metadata = training["metadata"]
    matching_training = [item for item in report.get("training", [])
                         if item.get("metadata", {}).get("adapter_sha256") == selection["selected_adapter_sha256"]]
    assert len(matching_training) == 1, "Report must contain exactly one selected V2 training record"
    assert_same(matching_training[0], training, "Report selected V2 training differs from its source record")
    for field in ("optimizer_steps", "selected_step"):
        assert_same(metadata.get(field), step, f"V2 training {field} differs from selected weights")
    for field in ("total_training_budget_steps", "completed_training_steps"):
        assert_same(metadata.get(field), 600, "V2 selection requires the complete 600-step training budget")
    assert metadata.get("selection_split") == "dev" and metadata.get("test_used_for_selection") is False, "V2 training must record dev-only selection"
    for field in ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256"):
        assert_same(metadata.get(field), expected_fingerprints[field], f"V2 training {field} differs from selected weights")
    binding = {"model_sha256": provenance["base_snapshot_sha256"],
               "data_sha256": provenance["training_data_sha256"],
               "config_sha256": provenance["training_config_sha256"]}
    assert_same(metadata.get("binding"), binding, "Selected V2 training binding mismatch")
    assert_same(checkpoint.get("binding"), binding, "Released V2 checkpoint training binding mismatch")
    history = training["history"]
    assert_same([item.get("step") for item in history], list(range(1, step + 1)), "Selected V2 history must end at the selected step")
    assert_same(history, checkpoint["history"], "Selected V2 history differs from the selected checkpoint")
    for field, expected in (("first_loss", history[0]["loss"]), ("last_loss", history[-1]["loss"]),
                            ("elapsed_s", history[-1]["elapsed_s"])):
        assert_same(metadata.get(field), expected, f"Selected V2 {field} differs from checkpoint history")
    training_folder = root / "experiments/training/evidence-v2"
    assert_same(json.loads((training_folder / "summary.json").read_text()), metadata,
                "Published V2 training summary differs from the selected record")
    assert_same(json.loads((training_folder / "metrics.json").read_text()), history,
                "Published V2 training metrics differ from the selected history")
    return {"status": "verified", "dev_candidates_recomputed": len(candidates),
            "quality_gate_recomputed": True, "quality_gate_status": rebuilt_gate["status"],
            "selected_step": step}


def verify_method(rows, method):
    lookup = {r["id"]: r for r in rows}
    assert method["dataset_sha256"] == digest(rows), "Evaluation dataset mismatch"
    assert len(method["cases"]) == len(rows), "Missing predictions"
    assert {c["id"] for c in method["cases"]} == set(lookup), "Unexpected case ids"
    checked = []
    for case in method["cases"]:
        row = lookup[case["id"]]
        assert case["observation"] == row["observation"] and case["expected"] == row["target"]
        rebuilt = score(row, case["raw"], case["latency_ms"])
        for key in ("group_id", "variant", "prediction", "metrics"):
            assert rebuilt[key] == case[key], f"Stored {key} does not match original output"
        checked.append(rebuilt)
    assert summarize(checked) == method["summary"], "Summary does not match cases"


def verify(root=Path(".")):
    root = Path(root)
    manifest = json.loads((root / "data/main/manifest.json").read_text())
    rows = [r for split in ("train", "dev", "test") for r in read_jsonl(root / f"data/main/{split}.jsonl")]
    assert audit(rows)["dataset_sha256"] == manifest["dataset_sha256"], "Dataset differs from manifest"
    assert rows == generate(manifest["seed"]), "Published dataset cannot be regenerated"
    test_rows = [r for r in rows if r["split"] == "test"]
    report = json.loads((root / "experiments/report.json").read_text())
    methods = report["methods"]
    for method in methods:
        verify_method(test_rows, method)
    for training in report.get("training", []):
        metadata = training.get("metadata", {})
        adapter_sha = metadata.get("adapter_sha256")
        assert any(m.get("model_provenance", {}).get("adapter_sha256") == adapter_sha for m in methods), "Unmatched training artifact"
    diagnostics = 0
    for name in ("paraphrase", "redacted"):
        diagnostic_rows = read_jsonl(root / f"data/diagnostics/{name}.jsonl")
        for path in sorted((root / "experiments/diagnostics").glob(f"*-{name}.json")):
            verify_method(diagnostic_rows, json.loads(path.read_text()))
            diagnostics += 1
    v2_rows, v2_methods = [], []
    if (root / "data/v2/manifest.json").exists():
        v2_manifest = json.loads((root / "data/v2/manifest.json").read_text())
        v2_rows = [r for split in ("train", "dev", "test") for r in read_jsonl(root / f"data/v2/{split}.jsonl")]
        builder = load_script(root, "build_v2")
        assert v2_rows == builder.generate(v2_manifest["seed"]), "V2 data cannot be regenerated"
        builder.validate(v2_rows, {r["gold"]["product"] for r in rows})
        for filename, record in v2_manifest["files"].items():
            assert sha(root / "data/v2" / filename) == record["sha256"]
        if (root / "experiments/v2/report.json").exists():
            v2_report = json.loads((root / "experiments/v2/report.json").read_text())
            v2_methods = v2_report["methods"]
            for method in v2_methods:
                verify_method([r for r in v2_rows if r["split"] == "test"], method)
    v2_release = verify_v2_release(root)
    released = json.loads((root / "artifacts/manifest.json").read_text())
    for filename, expected in released["files"].items():
        path = (root / filename).resolve()
        assert path.is_relative_to(root.resolve()), "Invalid artifact path"
        assert sha(path) == expected, f"Checksum mismatch: {filename}"
    return {"v1_data_rows": len(rows), "v2_data_rows": len(v2_rows), "test_cases_per_method": len(test_rows),
            "main_methods_recomputed": len(methods) + len(v2_methods), "diagnostics_recomputed": diagnostics,
            "artifact_files_verified": len(released["files"]), "v2_release": v2_release}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
