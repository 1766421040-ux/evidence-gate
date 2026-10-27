"""CPU-only V3 dev selection, test authorization and strict final audit.

Seven hard gates precede ranking. No eligible dev candidate means no test
authorization. The protocol supplies the exact variant list and frozen data
hashes; V1/V2 selectors and historical scores are never changed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

from evidence_gate.data import audit, canonical, digest
from evidence_gate.evaluation import ACTIONS, score, summarize

GATES = {
    "strict_success_min": ("success_rate", ">=", .85),
    "answerable_success_min": ("answerable_success", ">=", .80),
    "action_macro_f1_min": ("action_macro_f1", ">=", .80),
    "all_variants_correct_min": ("all_variants_correct_rate", ">=", .50),
    "over_answer_rate_max": ("over_answer_rate", "<=", .05),
    "format_valid_rate_min": ("format_valid_rate", ">=", 1.0),
    "unsupported_answer_rate_max": ("unsupported_answer_rate", "<=", .05),
}
ORDERING = [
    {"metric": "answerable_success", "direction": "max"},
    {"metric": "all_variants_correct_rate", "direction": "max"},
    {"metric": "success_rate", "direction": "max"},
    {"metric": "unsupported_answer_rate", "direction": "min"},
    {"metric": "registration_order", "direction": "min"},
]
INFERENCE = ("decoding", "max_new_tokens", "dtype", "batch_size", "device", "output_repair")
BINDINGS = ("base_snapshot_sha256", "training_data_sha256", "training_config_sha256")
ARTIFACT_HASHES = ("adapter_sha256", "adapter_config_sha256", "checkpoint_sha256")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _decode(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def _snapshot(path):
    path = Path(path)
    raw = path.read_bytes()
    return raw.decode("utf-8"), {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _json(path):
    text, source = _snapshot(path)
    return _decode(text), source


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same(a, b):
    return canonical(a) == canonical(b)


def _protocol(path):
    protocol, source = _json(path)
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    variants = protocol.get("variants")
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) or not v for v in variants) or len(set(variants)) != len(variants):
        raise ValueError("Protocol must explicitly declare unique nonempty variants")
    for split in ("dev", "test"):
        n = protocol.get(f"{split}_groups")
        if type(n) is not int or n < 1 or not _hash(protocol.get("data_sha256", {}).get(split)):
            raise ValueError(f"Protocol requires {split}_groups and frozen {split} file SHA256")
    if not isinstance(protocol.get("gate"), dict) or set(protocol["gate"]) != set(GATES):
        raise ValueError("V3 protocol must contain exactly the seven quality gates")
    for name, (_, direction, floor) in GATES.items():
        threshold = protocol["gate"][name]
        if not _finite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid gate threshold: {name}")
        if (direction == ">=" and threshold < floor) or (direction == "<=" and threshold > floor):
            raise ValueError(f"Protocol cannot lower the existing quality threshold: {name}")
    risk = protocol.get("answer_risk_eligibility", {})
    fraction = risk.get("minimum_emitted_answers_fraction_of_answerable")
    if not _finite(fraction) or not .8 <= fraction <= 1 or risk.get("rounding") != "ceil" or risk.get("undefined_metrics_pass") is not False:
        raise ValueError("Protocol must specify the non-vacuous answer-risk denominator rule")
    if not _same(protocol.get("selection", {}).get("ordering"), ORDERING):
        raise ValueError("Protocol must declare the fixed eligible-candidate ranking")
    inference = protocol.get("inference", {})
    if any(inference.get(key) is None for key in INFERENCE) or inference.get("decoding") != "greedy free generation" or inference.get("output_repair") is not False:
        raise ValueError("Protocol requires fixed unrepaired greedy inference settings")
    if any(type(inference[key]) is not int or inference[key] < 1 for key in ("batch_size", "max_new_tokens")):
        raise ValueError("Protocol inference batch size and token limit must be positive")
    return protocol, source


def _registry(path, protocol_source=None):
    registry, source = _json(path)
    if not isinstance(registry, dict) or registry.get("format") != "evidence-gate-v3-candidate-registry-v1":
        raise ValueError("Unsupported candidate registry format")
    if "protocol_sha256" in registry and (not protocol_source or registry["protocol_sha256"] != protocol_source["sha256"]):
        raise ValueError("Candidate registry is bound to a different protocol")
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate registry must not be empty")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not candidate["id"] or candidate["id"] in seen:
            raise ValueError("Candidate ids must be nonempty and unique")
        seen.add(candidate["id"])
        if type(candidate.get("checkpoint_step")) is not int or candidate["checkpoint_step"] < 1 or any(not _hash(candidate.get(key)) for key in BINDINGS):
            raise ValueError("Each candidate needs a step and exact base/data/config hashes")
    return candidates, source


def _metric_fractions(cases, summary):
    answers = [case for case in cases if case["expected"]["action"] == "answer"]
    denied = [case for case in cases if case["expected"]["action"] != "answer"]
    emitted = [case for case in cases if case["prediction"] and case["prediction"]["action"] == "answer"]
    groups = defaultdict(list)
    for case in cases:
        groups[case["group_id"]].append(case)
    counts = {
        "success_rate": (sum(case["metrics"]["success"] for case in cases), len(cases)),
        "answerable_success": (sum(case["metrics"]["success"] for case in answers), len(answers)),
        "all_variants_correct_rate": (sum(all(case["metrics"]["success"] for case in group) for group in groups.values()), len(groups)),
        "over_answer_rate": (sum(bool(case["prediction"] and case["prediction"]["action"] == "answer") for case in denied), len(denied)),
        "format_valid_rate": (sum(case["metrics"]["format_valid"] for case in cases), len(cases)),
        "unsupported_answer_rate": (sum(case["metrics"]["unsupported_answer"] for case in emitted), len(emitted)),
    }
    fractions = {key: Fraction(n, d) if d else None for key, (n, d) in counts.items()}
    f1s = []
    confusion = summary["confusion"]
    for action in ACTIONS:
        tp = confusion[action][action]
        fp = sum(confusion[other][action] for other in ACTIONS if other != action)
        fn = sum(n for predicted, n in confusion[action].items() if predicted != action)
        f1s.append(Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction())
    fractions["action_macro_f1"] = sum(f1s, Fraction()) / len(ACTIONS)
    return counts, fractions


def _audit(evaluation_path, data_path, protocol, protocol_source, split):
    evaluation, evaluation_source = _json(evaluation_path)
    text, data_source = _snapshot(data_path)
    if data_source["sha256"] != protocol["data_sha256"][split]:
        raise ValueError(f"Supplied {split} data differs from the frozen protocol hash")
    rows = [_decode(line) for line in text.splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or row.get("split") != split for row in rows):
        raise ValueError(f"This entry requires only split={split} data")
    data_audit = audit(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["variant"])
    if len(groups) != protocol[f"{split}_groups"]:
        raise ValueError(f"Incomplete {split} group count")
    for group, variants in groups.items():
        if Counter(variants) != Counter(protocol["variants"]):
            raise ValueError(f"Incomplete or duplicated protocol variants: {group}")
    if set(data_audit["actions"]) != set(ACTIONS):
        raise ValueError("Acceptance data must cover all four actions")
    if not isinstance(evaluation, dict) or evaluation.get("kind") != "model":
        raise ValueError("Only original model evaluations can be selected or accepted")
    if evaluation.get("dataset_sha256") != digest(rows):
        raise ValueError("Evaluation is not bound to the supplied dataset")
    inference = evaluation.get("inference", {})
    if any(not _same(inference.get(key), protocol["inference"][key]) for key in INFERENCE) or inference.get("adapter_used") is not True:
        raise ValueError("Model inference differs from the frozen adapter inference contract")
    cases = evaluation.get("cases")
    if not isinstance(cases, list) or len(cases) != len(rows) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("Evaluation cases are incomplete")
    lookup = {row["id"]: row for row in rows}
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) for case_id in ids) or len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("Evaluation case ids are missing, duplicated or unexpected")
    rebuilt = []
    for case in cases:
        row = lookup[case["id"]]
        for key, expected in (("group_id", row["group_id"]), ("variant", row["variant"]), ("observation", row["observation"]), ("expected", row["target"])):
            if key not in case or not _same(case[key], expected):
                raise ValueError(f"Case {case['id']} has altered {key}")
        if not isinstance(case.get("raw"), str) or not _finite(case.get("latency_ms")) or case["latency_ms"] < 0:
            raise ValueError("Original raw output and finite nonnegative latency are required")
        rescored = score(row, case["raw"], case["latency_ms"])
        if any(key not in case or not _same(case[key], rescored[key]) for key in ("prediction", "metrics", "error")):
            raise ValueError("Stored case score differs from original raw-output scoring")
        rebuilt.append(rescored)
    summary = summarize(rebuilt)
    if not _same(evaluation.get("summary"), summary):
        raise ValueError("Stored summary differs from strictly rescored raw outputs")
    counts, fractions = _metric_fractions(rebuilt, summary)
    required_answers = math.ceil(Fraction(str(protocol["answer_risk_eligibility"]["minimum_emitted_answers_fraction_of_answerable"])) * summary["denominators"]["answerable"])
    emitted = summary["denominators"]["emitted_answers"]
    checks = []
    for gate, (metric, direction, _) in GATES.items():
        actual, exact = summary[metric], fractions[metric]
        eligible = _finite(actual) and exact is not None
        reason = "Metric is undefined; null does not pass." if not eligible else None
        if metric in ("over_answer_rate", "unsupported_answer_rate") and (emitted == 0 or emitted < required_answers):
            eligible, reason = False, "Too few parseable emitted answers for the declared risk denominator rule."
        threshold = Fraction(str(protocol["gate"][gate]))
        passed = bool(eligible and (exact >= threshold if direction == ">=" else exact <= threshold))
        numerator, denominator = counts.get(metric, (None, None))
        checks.append({"gate": gate, "metric": metric, "actual": actual, "threshold": protocol["gate"][gate],
                       "direction": direction, "numerator": numerator, "denominator": denominator,
                       "exact_fraction": str(exact) if exact is not None else None,
                       "eligible": eligible, "passed": passed,
                       "reason": reason or ("Threshold met." if passed else "Threshold not met.")})
    passed = all(check["passed"] for check in checks)
    result = {"format": "evidence-gate-v3-quality-audit-v1", "name": evaluation.get("name"), "kind": "model", "split": split,
              "status": "pass" if passed else "fail", "passed": passed,
              "integrity": {"verified": True, f"{split}_rows": len(rows), f"{split}_groups": len(groups),
                            "dataset_sha256": digest(rows), "case_identity_checked": True, "raw_outputs_rescored": True, "stored_summary_matches": True},
              "sources": {"evaluation": evaluation_source, "data": data_source, "protocol": protocol_source},
              "answer_risk_eligibility": {"required_emitted_answers": required_answers, "actual_emitted_answers": emitted},
              "checks": checks, "recomputed_summary": summary}
    return result, evaluation, fractions


def _artifact(candidate, evaluation, order):
    provenance = evaluation.get("model_provenance", {})
    if provenance.get("adapter_training_binding_verified") is not True:
        raise ValueError("Candidate adapter training provenance is not verified")
    if not _same(provenance.get("checkpoint_step"), candidate["checkpoint_step"]) or any(provenance.get(key) != candidate[key] for key in BINDINGS):
        raise ValueError("Candidate differs from its preregistered checkpoint or training binding")
    if any(not _hash(provenance.get(key)) for key in ARTIFACT_HASHES):
        raise ValueError("Candidate lacks exact adapter/config/checkpoint hashes")
    return {"id": candidate["id"], "step": candidate["checkpoint_step"], "registration_order": order,
            **{key: provenance[key] for key in (*BINDINGS, *ARTIFACT_HASHES)}}


def select_candidates(evaluations, dev_data, protocol_path, registry_path):
    protocol, protocol_source = _protocol(protocol_path)
    registry, registry_source = _registry(registry_path, protocol_source)
    if not isinstance(evaluations, dict) or set(evaluations) != {candidate["id"] for candidate in registry}:
        raise ValueError("Provide exactly one dev evaluation for every registered candidate")
    candidates, eligible = [], []
    for order, candidate in enumerate(registry):
        result, evaluation, fractions = _audit(evaluations[candidate["id"]], dev_data, protocol, protocol_source, "dev")
        artifact = _artifact(candidate, evaluation, order)
        candidates.append({"id": candidate["id"], "registration_order": order, "artifact": artifact, "audit": result})
        if result["passed"]:
            rank = (fractions["answerable_success"], fractions["all_variants_correct_rate"], fractions["success_rate"], -fractions["unsupported_answer_rate"], -order)
            eligible.append((rank, artifact))
    eligible.sort(key=lambda entry: entry[0], reverse=True)
    chosen = eligible[0][1] if eligible else None
    return {"format": "evidence-gate-v3-selection-v1", "status": "selected" if chosen else "no_eligible_candidate",
            "selection_split": "dev", "test_used_for_selection": False, "selected": chosen, "test_allowed": chosen is not None,
            "sources": {"protocol": protocol_source, "registry": registry_source, "data": candidates[0]["audit"]["sources"]["data"]},
            "ordering": ORDERING, "candidate_audits": candidates, "eligible_candidate_ids": [item[1]["id"] for item in eligible],
            "interpretation": "All seven development gates precede ranking. Development scores are adaptive selection evidence, not final test estimates."}


def _selected(selection_path, protocol_source=None):
    selection, source = _json(selection_path)
    if not isinstance(selection, dict) or selection.get("format") != "evidence-gate-v3-selection-v1" or selection.get("status") != "selected" or selection.get("test_allowed") is not True or not isinstance(selection.get("selected"), dict):
        raise ValueError("Test is forbidden: there is no eligible selected development candidate")
    if selection.get("selection_split") != "dev" or selection.get("test_used_for_selection") is not False:
        raise ValueError("Test authorization requires dev-only selection")
    if protocol_source and selection.get("sources", {}).get("protocol", {}).get("sha256") != protocol_source["sha256"]:
        raise ValueError("Protocol changed after development selection")
    return selection, source


def require_test_authorization(selection_path, protocol_path, registry_path, adapter_dir, dev_data, evaluations):
    saved, _ = _selected(selection_path)
    rebuilt = select_candidates(evaluations, dev_data, protocol_path, registry_path)
    if not _same(saved, rebuilt):
        raise ValueError("Saved selection differs from the current protocol, registry or raw dev evidence")
    chosen = rebuilt["selected"]
    adapter = Path(adapter_dir)
    for key, filename in (("adapter_sha256", "adapter_model.safetensors"), ("adapter_config_sha256", "adapter_config.json"), ("checkpoint_sha256", "checkpoint.json")):
        if _sha(adapter / filename) != chosen[key]:
            raise ValueError(f"Selected artifact checksum changed: {filename}")
    checkpoint, _ = _json(adapter / "checkpoint.json")
    expected_binding = {"model_sha256": chosen["base_snapshot_sha256"], "data_sha256": chosen["training_data_sha256"], "config_sha256": chosen["training_config_sha256"]}
    if checkpoint.get("format") != "evidence-gate-checkpoint-v1" or not _same(checkpoint.get("step"), chosen["step"]) or not _same(checkpoint.get("binding"), expected_binding):
        raise ValueError("Selected checkpoint has inconsistent step or training binding")
    for key, filename in (("adapter_sha256", "adapter_model.safetensors"), ("adapter_config_sha256", "adapter_config.json")):
        if checkpoint.get("files", {}).get(filename) != chosen[key]:
            raise ValueError("Selected checkpoint file manifest is inconsistent")
    return chosen


def audit_test(evaluation_path, test_data, protocol_path, selection_path):
    # Fail before reading test data when no dev candidate has qualified.
    selection, selection_source = _selected(selection_path)
    protocol, protocol_source = _protocol(protocol_path)
    if selection["sources"]["protocol"]["sha256"] != protocol_source["sha256"]:
        raise ValueError("Protocol changed after development selection")
    result, evaluation, _ = _audit(evaluation_path, test_data, protocol, protocol_source, "test")
    selected = selection["selected"]
    candidate = {"id": selected["id"], "checkpoint_step": selected["step"], **{key: selected[key] for key in BINDINGS}}
    actual = _artifact(candidate, evaluation, selected["registration_order"])
    if not _same(selected, actual):
        raise ValueError("Test model is not the selected development artifact")
    result["sources"]["selection"] = selection_source
    result["selected"] = selected
    result["confirmation_status"] = "confirmed_pass" if result["passed"] else "confirmed_fail"
    result["interpretation"] = "Raw-output integrity and seven quality gates are checked separately. A failed final test remains a failed confirmation."
    return result


def _evaluation_map(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Use --evaluation candidate_id=path")
        candidate, path = value.split("=", 1)
        if not candidate or not path or candidate in result:
            raise ValueError("Candidate evaluation mappings must be nonempty and unique")
        result[candidate] = path
    return result


def _write(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "authorize", "audit-test"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", required=True)
        sub.add_argument("--protocol", required=True)
        if name != "select":
            sub.add_argument("--selection", required=True)
        if name != "audit-test":
            sub.add_argument("--registry", required=True)
            sub.add_argument("--evaluation", action="append", required=True)
        else:
            sub.add_argument("--evaluation", required=True)
        if name == "authorize":
            sub.add_argument("--adapter", required=True)
        else:
            sub.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            result = select_candidates(_evaluation_map(args.evaluation), args.data, args.protocol, args.registry)
        elif args.command == "authorize":
            result = require_test_authorization(args.selection, args.protocol, args.registry, args.adapter, args.data, _evaluation_map(args.evaluation))
        else:
            result = audit_test(args.evaluation, args.data, args.protocol, args.selection)
        if args.command != "authorize":
            _write(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result.get("status", "authorized"), "selected": result.get("selected", result if args.command == "authorize" else None)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
