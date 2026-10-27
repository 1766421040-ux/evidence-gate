"""Strict output scoring and group-level paired evaluation.

Predictions are scored exactly as generated. No JSON repair, guardrail rewriting,
or gold-conditioned decoding is allowed in the model metrics.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from datetime import date

from .data import canonical, decision, digest, messages_for

ACTIONS = ("answer", "search", "clarify", "abstain")


def parse_prediction(raw: str) -> dict:
    def unique_pairs(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("Duplicate JSON key")
            obj[k] = v
        return obj
    try:
        obj = json.loads(raw, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Output is not a single JSON object") from exc
    if not isinstance(obj, dict) or set(obj) != {"action", "value", "citations"}:
        raise ValueError("Expected exactly action, value, citations")
    if obj["action"] not in ACTIONS or not isinstance(obj["value"], str):
        raise ValueError("Invalid action/value type")
    citations = obj["citations"]
    if not isinstance(citations, list) or any(not isinstance(x, str) for x in citations):
        raise ValueError("Citations must be strings")
    if len(set(citations)) != len(citations):
        raise ValueError("Duplicate citation")
    if obj["action"] != "answer" and (obj["value"] or citations):
        raise ValueError("Non-answer actions must have empty value/citations")
    if obj["action"] == "answer" and (not obj["value"].strip() or not citations):
        raise ValueError("Answer requires value and citations")
    return obj


def score(row, raw, latency_ms=0.0) -> dict:
    expected = row["target"]
    error = None
    try:
        predicted = parse_prediction(raw)
    except (ValueError, TypeError) as exc:
        predicted, error = None, str(exc)
    action = predicted["action"] if predicted else "invalid"
    action_ok = action == expected["action"]
    citation_ok = bool(predicted) and set(predicted["citations"]) == set(expected["citations"])
    value_ok = bool(predicted) and predicted["value"].strip().casefold() == expected["value"].strip().casefold()
    success = action_ok and citation_ok and value_ok
    unanswerable = expected["action"] != "answer"
    unsupported = action == "answer" and (unanswerable or not value_ok or not citation_ok)
    metrics = {"format_valid": predicted is not None, "action_correct": action_ok,
               "answer_correct": success if expected["action"] == "answer" else None,
               "citation_correct": citation_ok if action == "answer" else None,
               "unsupported_answer": unsupported, "unanswerable": unanswerable,
               "over_refusal": expected["action"] == "answer" and action in ("abstain", "clarify"),
               "unnecessary_search": expected["action"] == "answer" and action == "search",
               "success": bool(success)}
    if error is None and not success:
        error = "wrong_action" if not action_ok else ("wrong_value" if not value_ok else "wrong_citations")
    return {"id": row["id"], "group_id": row["group_id"], "variant": row["variant"],
            "observation": row["observation"], "expected": expected,
            "prediction": predicted, "raw": raw, "metrics": metrics,
            "error": error, "latency_ms": latency_ms}


def ratio(n, d):
    return n / d if d else None


def bootstrap_group_ci(cases, seed=17, repetitions=1000):
    """Resample whole question groups, never correlated variants independently."""
    grouped = defaultdict(list)
    for c in cases:
        grouped[c["group_id"]].append(float(c["metrics"]["success"]))
    groups = list(grouped.values())
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        values.append(sum(sum(x) for x in sampled) / sum(map(len, sampled)))
    values.sort()
    return [values[int(.025 * repetitions)], values[min(repetitions - 1, int(.975 * repetitions))]]


def summarize(cases):
    if not cases:
        raise ValueError("Cannot score zero cases")
    n = len(cases)
    count = lambda metric: sum(bool(c["metrics"][metric]) for c in cases)
    answers = [c for c in cases if c["expected"]["action"] == "answer"]
    emitted = [c for c in cases if c["prediction"] and c["prediction"]["action"] == "answer"]
    denied = [c for c in cases if c["metrics"]["unanswerable"]]
    confusion = {a: {p: 0 for p in (*ACTIONS, "invalid")} for a in ACTIONS}
    groups, variants = defaultdict(list), defaultdict(list)
    for c in cases:
        confusion[c["expected"]["action"]][c["prediction"]["action"] if c["prediction"] else "invalid"] += 1
        groups[c["group_id"]].append(c)
        variants[c["variant"]].append(c)
    f1s = []
    for a in ACTIONS:
        tp = confusion[a][a]
        fp = sum(confusion[t][a] for t in ACTIONS if t != a)
        fn = sum(v for p, v in confusion[a].items() if p != a)
        f1s.append(ratio(2 * tp, 2 * tp + fp + fn) or 0.0)
    pairs = {"value_change": ("supported", "value_changed"),
             "distractor_invariance": ("supported", "distractor"),
             "order_invariance": ("supported", "reordered"),
             "evidence_removal": ("supported", "missing"),
             "budget_exhaustion": ("missing_search", "missing_stop"),
             "expiry_boundary": ("supported", "expired")}
    pair_scores = {}
    for name, (a, b) in pairs.items():
        matched = []
        for members in groups.values():
            lookup = {c["variant"]: c for c in members}
            if "supported_b0" in lookup:
                # V3 crosses each semantic intervention with both budgets.
                # Count both matched-budget pairs, never silently reduce a
                # 20-cell group to one arbitrary last-seen variant.
                if name == "budget_exhaustion":
                    keys = [("missing_b0", "missing_b1")]
                else:
                    keys = [(f"{a}_b{budget}", f"{b}_b{budget}") for budget in (0, 1)]
                for left, right in keys:
                    if left in lookup and right in lookup:
                        matched.append(lookup[left]["metrics"]["success"] and lookup[right]["metrics"]["success"])
                continue
            # Isolate the evidence intervention: do not change budget along
            # with evidence now that supported states have either budget.
            actual_b = b
            if b in ("missing", "expired") and a in lookup:
                actual_b = b + ("_search" if lookup[a]["observation"]["budget"] > 0 else "_stop")
            if a in lookup and actual_b in lookup:
                matched.append(lookup[a]["metrics"]["success"] and lookup[actual_b]["metrics"]["success"])
        pair_scores[name] = {"correct": sum(matched), "total": len(matched),
                             "accuracy": ratio(sum(matched), len(matched))}
    if any("supported_b0" == case["variant"] for case in cases):
        # These reports only exist in V3, preserving every released V1/V2
        # summary byte-for-byte under independent raw-output verification.
        for name, left, right in (
            ("answer_budget_invariance", "supported_b0", "supported_b1"),
            ("clarify_budget_invariance", "ambiguous_b0", "ambiguous_b1"),
        ):
            matched = []
            for members in groups.values():
                lookup = {case["variant"]: case for case in members}
                if left in lookup and right in lookup:
                    matched.append(lookup[left]["metrics"]["success"] and lookup[right]["metrics"]["success"])
            pair_scores[name] = {"correct": sum(matched), "total": len(matched),
                                 "accuracy": ratio(sum(matched), len(matched))}
    return {"n": n, "success_rate": count("success") / n,
            "success_group_bootstrap_95ci": bootstrap_group_ci(cases),
            "format_valid_rate": count("format_valid") / n,
            # sum(float) changed in Python 3.12. Explicit fsum keeps released
            # metrics identical across supported Python versions without
            # weakening exact audit comparisons or rounding away differences.
            "action_accuracy": count("action_correct") / n, "action_macro_f1": math.fsum(f1s) / 4,
            "answer_coverage": len(emitted) / n,
            "answerable_success": ratio(sum(c["metrics"]["success"] for c in answers), len(answers)),
            "unsupported_answer_rate": ratio(sum(c["metrics"]["unsupported_answer"] for c in emitted), len(emitted)),
            "over_answer_rate": ratio(sum(bool(c["prediction"] and c["prediction"]["action"] == "answer") for c in denied), len(denied)),
            "over_refusal_rate": ratio(count("over_refusal"), len(answers)),
            "unnecessary_search_rate": ratio(count("unnecessary_search"), len(answers)),
            "citation_exact_rate": ratio(sum(c["metrics"]["citation_correct"] for c in emitted), len(emitted)),
            "all_variants_correct_rate": sum(all(c["metrics"]["success"] for c in g) for g in groups.values()) / len(groups),
            "groups": len(groups), "mean_latency_ms": math.fsum(c["latency_ms"] for c in cases) / n,
            "denominators": {"answerable": len(answers), "unanswerable": len(denied), "emitted_answers": len(emitted)},
            "confusion": confusion, "pairs": pair_scores,
            "by_variant": {v: {"n": len(cs), "success_rate": sum(c["metrics"]["success"] for c in cs) / len(cs)} for v, cs in variants.items()}}


def contract_rule(observation):
    """Strong transparent baseline specialized to the declared synthetic grammar.

    It only reads public text/metadata, not dataset labels. It is not presented as
    a general natural-language verifier or as evidence that SFT is necessary.
    """
    question = observation["question"]
    match = re.search(r"For (\S+) on the (\w+) plan", question)
    if not match:
        return decision("clarify")
    product, plan = match.groups()
    phrases = {"audit-log retention": "audit logs are retained for",
               "deleted-file recovery": "deleted files can be restored for",
               "backup retention": "workspace backups are stored for"}
    predicate = next((p for topic, p in phrases.items() if topic in question), None)
    found = []
    today = date.fromisoformat(observation["as_of"])
    for d in observation["documents"]:
        valid = date.fromisoformat(d["valid_from"]) <= today
        if d["valid_until"]:
            valid &= today < date.fromisoformat(d["valid_until"])
        if d["source"] != "official" or not valid or not predicate:
            continue
        pattern = rf"On {re.escape(product)}'s {re.escape(plan)} plan, {predicate} (\d+) days\."
        m = re.fullmatch(pattern, d["text"])
        if m:
            found.append((m[1] + " days", d["id"]))
    if found and len({v for v, _ in found}) == 1:
        return decision("answer", found[0][0], [i for _, i in found])
    return decision("search" if observation["budget"] > 0 else "abstain")


def first_match(observation):
    """Weak retrieval baseline: matches scope but ignores validity and authority."""
    match = re.search(r"For (\S+) on the (\w+) plan", observation["question"])
    if not match:
        return decision("clarify")
    product, plan = match.groups()
    for doc in observation["documents"]:
        if f"{product}'s {plan} plan" in doc["text"]:
            value = re.search(r"(\d+) days", doc["text"])
            if value:
                return decision("answer", value[1] + " days", [doc["id"]])
    return decision("search" if observation["budget"] > 0 else "abstain")


def evaluate(rows, policy, name, kind="rule", batch_size=8):
    cases = []
    if kind == "model":
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            outputs = policy.predict_batch([r["messages"] for r in batch])
            if len(outputs) != len(batch):
                raise ValueError("Model output count does not match input count")
            for r, o in zip(batch, outputs):
                case = score(r, o["raw"], o.get("latency_ms", 0))
                case["generation"] = {k: v for k, v in o.items() if k not in ("raw", "latency_ms")}
                cases.append(case)
            print(f"{name}: {len(cases)}/{len(rows)}", flush=True)
    else:
        for row in rows:
            start = time.perf_counter()
            output = policy(row["observation"])
            cases.append(score(row, canonical(output), (time.perf_counter() - start) * 1000))
    return {"name": name, "kind": kind, "dataset_sha256": digest(rows),
            "summary": summarize(cases), "cases": cases}
