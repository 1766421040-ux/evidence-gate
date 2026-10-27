"""Recompute released scores and verify data/adapter identities without ML deps."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from evidence_gate.data import audit, digest, generate, read_jsonl
from evidence_gate.evaluation import score, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_script(root, name):
    spec = importlib.util.spec_from_file_location(name, root / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    released = json.loads((root / "artifacts/manifest.json").read_text())
    for filename, expected in released["files"].items():
        path = (root / filename).resolve()
        assert path.is_relative_to(root.resolve()), "Invalid artifact path"
        assert sha(path) == expected, f"Checksum mismatch: {filename}"
    return {"v1_data_rows": len(rows), "v2_data_rows": len(v2_rows), "test_cases_per_method": len(test_rows),
            "main_methods_recomputed": len(methods) + len(v2_methods), "diagnostics_recomputed": diagnostics,
            "artifact_files_verified": len(released["files"])}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
