"""Recompute released scores and verify data/adapter identities without ML deps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evidence_gate.data import audit, digest, generate, read_jsonl
from evidence_gate.evaluation import score, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root=Path(".")):
    root = Path(root)
    manifest = json.loads((root / "data/main/manifest.json").read_text())
    rows = [r for split in ("train", "dev", "test") for r in read_jsonl(root / f"data/main/{split}.jsonl")]
    assert audit(rows)["dataset_sha256"] == manifest["dataset_sha256"], "Dataset differs from manifest"
    assert rows == generate(manifest["seed"]), "Published dataset cannot be regenerated"
    test_rows = [r for r in rows if r["split"] == "test"]
    lookup = {r["id"]: r for r in test_rows}
    report = json.loads((root / "experiments/report.json").read_text())
    methods = report["methods"]
    for method in methods:
        assert method["dataset_sha256"] == digest(test_rows), "Evaluation dataset mismatch"
        assert len(method["cases"]) == len(test_rows), "Missing predictions"
        assert {c["id"] for c in method["cases"]} == set(lookup), "Unexpected case ids"
        checked = []
        for case in method["cases"]:
            row = lookup[case["id"]]
            assert case["observation"] == row["observation"] and case["expected"] == row["target"]
            rebuilt = score(row, case["raw"], case["latency_ms"])
            assert rebuilt["metrics"] == case["metrics"], "Stored metrics do not match raw output"
            checked.append(rebuilt)
        assert summarize(checked) == method["summary"], "Summary does not match cases"
    for training in report.get("training", []):
        metadata = training.get("metadata", {})
        adapter_sha = metadata.get("adapter_sha256")
        assert any(m.get("model_provenance", {}).get("adapter_sha256") == adapter_sha for m in methods), "Unmatched training artifact"
    released = json.loads((root / "artifacts/manifest.json").read_text())
    for filename, expected in released["files"].items():
        path = (root / filename).resolve()
        assert path.is_relative_to(root.resolve()), "Invalid artifact path"
        assert sha(path) == expected, f"Checksum mismatch: {filename}"
    return {"data_rows": len(rows), "test_cases_per_method": len(test_rows),
            "methods_recomputed": len(methods), "artifact_files_verified": len(released["files"])}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
