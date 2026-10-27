"""Audit raw test predictions, then apply the already locked quality gates.

A truthful failure is a successful audit: it is saved and exits 0 by default.
Use --require-pass to exit 2 when the model misses any gate. Invalid or altered
evaluation evidence raises an error instead of receiving a quality verdict.
This script imports no training framework and never calls a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from evidence_gate.data import VARIANTS, audit, canonical, digest
from evidence_gate.evaluation import score, summarize


# Names are protocol fields; values identify existing evaluator metrics. There
# are no fallback thresholds here: every threshold must come from the protocol.
GATE_METRICS = {
    "strict_success_min": ("success_rate", ">="),
    "answerable_success_min": ("answerable_success", ">="),
    "action_macro_f1_min": ("action_macro_f1", ">="),
    "all_variants_correct_min": ("all_variants_correct_rate", ">="),
    "over_answer_rate_max": ("over_answer_rate", "<="),
    "format_valid_rate_min": ("format_valid_rate", ">="),
}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _decode(value):
    return json.loads(value, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def _snapshot(path):
    path = Path(path)
    raw = path.read_bytes()
    return raw.decode("utf-8"), {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same(actual, expected):
    # JSON equality distinguishes true from 1, unlike Python's plain ==.
    return canonical(actual) == canonical(expected)


def audit_gate(evaluation_path, data_path, protocol_path) -> dict:
    """Return a reproducible gate audit, or raise for untrustworthy evidence."""
    evaluation_text, evaluation_source = _snapshot(evaluation_path)
    data_text, data_source = _snapshot(data_path)
    protocol_text, protocol_source = _snapshot(protocol_path)
    evaluation, protocol = _decode(evaluation_text), _decode(protocol_text)
    rows = [_decode(line) for line in data_text.splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Data must contain non-empty JSONL records")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("Quality gates require the test split; train/dev records are forbidden")
    data_audit = audit(rows)
    if not isinstance(protocol, dict) or not isinstance(protocol.get("gate"), dict) or not protocol["gate"]:
        raise ValueError("Protocol must contain a non-empty gate object")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("cases"), list):
        raise ValueError("Evaluation must contain a cases list")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row["variant"])
    for group_id, variants in grouped.items():
        if Counter(variants) != Counter(VARIANTS):
            raise ValueError(f"Incomplete or duplicated variants in test group: {group_id}")
    expected_groups = protocol.get("test_groups")
    if type(expected_groups) is not int or expected_groups < 1:
        raise ValueError("Protocol must declare a positive test_groups count")
    if len(grouped) != expected_groups:
        raise ValueError("Test group count does not match the locked protocol")
    if evaluation.get("dataset_sha256") != digest(rows):
        raise ValueError("Evaluation dataset_sha256 does not match the supplied test data")

    lookup = {row["id"]: row for row in rows}
    cases = evaluation["cases"]
    if len(cases) != len(rows) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("Evaluation cases are incomplete")
    case_ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) for case_id in case_ids):
        raise ValueError("Each evaluation case needs a string id")
    if len(set(case_ids)) != len(case_ids) or set(case_ids) != set(lookup):
        raise ValueError("Evaluation case ids are duplicated, missing or unexpected")

    rebuilt = []
    for case in cases:
        row = lookup[case["id"]]
        for key, expected in (
            ("group_id", row["group_id"]), ("variant", row["variant"]),
            ("observation", row["observation"]), ("expected", row["target"]),
        ):
            if key not in case or not _same(case[key], expected):
                raise ValueError(f"Case {case['id']} has altered {key}")
        raw = case.get("raw")
        latency = case.get("latency_ms")
        if not isinstance(raw, str):
            raise ValueError(f"Case {case['id']} is missing original text output")
        if not _finite_number(latency) or latency < 0:
            raise ValueError(f"Case {case['id']} has invalid latency_ms")
        rescored = score(row, raw, latency)
        for key in ("prediction", "metrics", "error"):
            if key not in case or not _same(case[key], rescored[key]):
                raise ValueError(f"Case {case['id']} {key} does not match strict raw-output scoring")
        rebuilt.append(rescored)
    summary = summarize(rebuilt)
    if not isinstance(evaluation.get("summary"), dict) or not _same(evaluation["summary"], summary):
        raise ValueError("Stored summary does not match the strictly rescored raw outputs")

    checks = []
    for gate_name, threshold in protocol["gate"].items():
        if gate_name not in GATE_METRICS:
            raise ValueError(f"Unknown protocol gate: {gate_name}")
        if not _finite_number(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Gate {gate_name} requires a finite threshold in [0,1]")
        metric, direction = GATE_METRICS[gate_name]
        actual = summary.get(metric)
        eligible = _finite_number(actual)
        reason = None if eligible else "Metric is undefined; null is not treated as zero."
        # over_answer_rate itself has an unanswerable-case denominator and can
        # numerically be zero when every output is invalid. Preserve that value,
        # but require at least one parseable answer before it can support this
        # safety gate. This eligibility condition does not change any threshold.
        if metric == "over_answer_rate" and summary["denominators"]["emitted_answers"] == 0:
            eligible = False
            reason = "No parseable answer was emitted; a numerical zero is not evidence of safe answering."
        passed = eligible and (actual >= threshold if direction == ">=" else actual <= threshold)
        if eligible:
            reason = "Threshold met." if passed else "Threshold not met."
        checks.append({"gate": gate_name, "metric": metric, "actual": actual,
                       "threshold": threshold, "direction": direction,
                       "eligible": eligible, "passed": bool(passed), "reason": reason})

    passed = all(check["passed"] for check in checks)
    return {
        "format": "evidence-gate-quality-audit-v1",
        "name": evaluation.get("name"), "kind": evaluation.get("kind"),
        "status": "pass" if passed else "fail", "passed": passed,
        "integrity": {"verified": True, "test_rows": len(rows), "test_groups": len(grouped),
                      "dataset_sha256": data_audit["dataset_sha256"],
                      "case_identity_checked": True, "raw_outputs_rescored": True,
                      "stored_summary_matches": True},
        "sources": {"evaluation": evaluation_source, "data": data_source, "protocol": protocol_source},
        "checks": checks, "recomputed_summary": summary,
        "risk_eligibility": "Risk-rate gates require a defined metric and at least one parseable emitted answer. Numerical zeros and null metrics are not substituted for safety evidence.",
        "interpretation": "Quality failure is preserved. Audit completion verifies consistency, not that the model meets the protocol gates or is production-ready.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--protocol", default="experiments/v2/protocol.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true", help="Return exit 2 after saving a truthful failed quality audit")
    args = parser.parse_args(argv)
    try:
        result = audit_gate(args.evaluation, args.data, args.protocol)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"audit_completed": True, "quality_gate": result["status"],
                      "passed_checks": sum(check["passed"] for check in result["checks"]),
                      "total_checks": len(result["checks"]), "output": str(output)}, ensure_ascii=False))
    return 2 if args.require_pass and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
