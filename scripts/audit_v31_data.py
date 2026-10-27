"""Independent V3.1 integrity and known insufficient-rule audit, CPU only.

The frozen V3 auditor is reused without modifying it. New strict rule probes
run only on train/dev and are diagnostics, never candidate model evaluations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import re

from evidence_gate.data import TOPICS, canonical, decision, read_jsonl
from evidence_gate.evaluation import score, summarize


ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = _module("frozen_v3_audit_for_v31", "audit_v3_data.py")
selection = _module("v3_gates_for_v31_data_audit", "selection_v3.py")
SEMANTICS = ("supported", "value_changed", "distractor", "reordered", "missing",
             "expired", "future", "conflict", "ambiguous", "untrusted",
             "irrelevant_topic", "consensus")
VARIANTS = tuple(f"{semantic}_b{budget}" for semantic in SEMANTICS for budget in (0, 1))
PREDICATES = {predicate for _, predicate, _ in TOPICS}
FACT = re.compile(r"On (\S+)'s (\w+) plan, (.+) (\d+) days\.")


def _fact(document):
    match = FACT.fullmatch(document["text"])
    if not match or match[3] not in PREDICATES:
        raise ValueError("Expected an official fact in the declared synthetic grammar")
    return dict(zip(("product", "plan", "predicate", "value"), match.groups()))


def _current(document, observation):
    return (document["valid_from"] <= observation["as_of"] and
            (not document["valid_until"] or observation["as_of"] < document["valid_until"]))


def insufficient_rule(observation, patched=False):
    """Two explicit incomplete public-text rules; neither reads labels or gold.

    Original: refuse mixed official topics or any candidate count other than 1.
    Patched: resolve scope and agreement but check the question topic ONLY when
    official topics differ. Uniform wrong-topic evidence exposes that omission.
    """
    match = re.search(r"For (\S+) on the (\w+) plan", observation["question"])
    if not match:
        return decision("clarify")
    product, plan = match.groups()
    official = [(doc, _fact(doc)) for doc in observation["documents"] if doc["source"] == "official"]
    mixed = len({fact["predicate"] for _, fact in official}) > 1
    candidates = [(doc, fact) for doc, fact in official
                  if fact["product"] == product and fact["plan"] == plan and _current(doc, observation)]
    fallback = decision("search" if observation["budget"] else "abstain")
    if not patched:
        if mixed or len(candidates) != 1:
            return fallback
    else:
        if mixed:
            predicate = next((p for _, p, topic in TOPICS if topic in observation["question"]), None)
            candidates = [(doc, fact) for doc, fact in candidates if fact["predicate"] == predicate]
        if not candidates or len({fact["value"] for _, fact in candidates}) != 1:
            return fallback
    return decision("answer", candidates[0][1]["value"] + " days", [doc["id"] for doc, _ in candidates])


def _strict_probe(rows, patched):
    cases = [score(row, canonical(insufficient_rule(row["observation"], patched))) for row in rows]
    summary = summarize(cases)
    _, fractions = selection._metric_fractions(cases, summary)
    minimum = math.ceil(Fraction(4, 5) * summary["denominators"]["answerable"])
    checks = []
    for name, (metric, direction, threshold) in selection.GATES.items():
        exact = fractions[metric]
        eligible = exact is not None
        if metric in ("over_answer_rate", "unsupported_answer_rate"):
            eligible &= summary["denominators"]["emitted_answers"] >= minimum
        bound = Fraction(str(threshold))
        passed = eligible and (exact >= bound if direction == ">=" else exact <= bound)
        checks.append({"gate": name, "metric": metric, "actual": summary[metric],
                       "threshold": threshold, "eligible": bool(eligible), "passed": bool(passed),
                       "exact_fraction": str(exact) if exact is not None else None})
    return {"kind": "insufficient_public_rule_diagnostic", "summary": summary,
            "checks": checks, "passes_all_seven": all(check["passed"] for check in checks),
            "minimum_emitted_answers": minimum,
            "note": "No model calls; original public-rule outputs strictly scored without repair. Not model-selection evidence."}


def _changed_only(source, altered, document_id, replacement):
    expected = copy.deepcopy(source)
    matches = [doc for doc in expected["documents"] if doc["id"] == document_id]
    if len(matches) != 1:
        raise ValueError("Official document identities must be unique")
    matches[0]["text"] = replacement
    return altered == expected


def audit_v31(rows, previously_seen_rows=(), expected_variants=None):
    if expected_variants is not None and tuple(expected_variants) != VARIANTS:
        raise ValueError("V3.1 requires the exact declared twenty-four variants")
    base = v3.audit_v3(rows, previously_seen_rows, VARIANTS)
    groups = defaultdict(dict)
    for row in rows:
        if re.fullmatch(rf"v31-{row['split']}-g\d{{4,}}", row["group_id"]) is None:
            raise ValueError("V3.1 requires fresh versioned split/group identities")
        if row["id"] != f"{row['group_id']}:{row['variant']}":
            raise ValueError("V3.1 row identity must bind its group and variant")
        groups[row["group_id"]][row["variant"]] = row
        documents = row["observation"]["documents"]
        if len(documents) != 3 or Counter(doc["source"] for doc in documents) != {"official": 2, "community": 1}:
            raise ValueError("V3.1 must have three documents with the same source counts")
        if sum(v3.NOISE in doc["text"] for doc in documents) != 1:
            raise ValueError("Every V3.1 context needs the same shared injection tail")
        if len({doc["id"] for doc in documents}) != 3:
            raise ValueError("Visible document IDs must be unique")
        official_ids = {doc["id"] for doc in documents if doc["source"] == "official"}
        if official_ids & {doc["id"] for doc in row["gold"]["catalogue"] if doc["source"] == "official"}:
            raise ValueError("Recovery official IDs must be independent of visible evidence")

    missing = {split: Counter() for split in v3.SPLITS}
    interventions = Counter()
    for members in groups.values():
        for semantic in SEMANTICS:
            left, right = (copy.deepcopy(members[f"{semantic}_b{budget}"]["observation"]) for budget in (0, 1))
            if left.pop("budget") != 0 or right.pop("budget") != 1 or left != right:
                raise ValueError("Each semantic budget pair must change only its budget")
        primary_id, = members["supported_b0"]["target"]["citations"]
        for budget in (0, 1):
            conflict = members[f"conflict_b{budget}"]["observation"]
            official = {doc["id"]: doc for doc in conflict["documents"] if doc["source"] == "official"}
            if primary_id not in official or any(not _current(doc, conflict) for doc in official.values()):
                raise ValueError("Scope counterexamples require both conflict documents to be current")
            secondary_id, = set(official) - {primary_id}
            primary, secondary = _fact(official[primary_id]), _fact(official[secondary_id])
            if any(primary[key] != secondary[key] for key in ("product", "plan", "predicate")) or primary["value"] == secondary["value"]:
                raise ValueError("Conflict must differ only in value within the same official scope")

            irrelevant = members[f"irrelevant_topic_b{budget}"]
            changed_doc = next(doc for doc in irrelevant["observation"]["documents"] if doc["id"] == secondary_id)
            changed_fact = _fact(changed_doc)
            if (changed_fact["predicate"] == secondary["predicate"] or
                    any(changed_fact[key] != secondary[key] for key in ("product", "plan", "value")) or
                    not _changed_only(conflict, irrelevant["observation"], secondary_id, changed_doc["text"])):
                raise ValueError("Irrelevant-topic intervention must change only the second official predicate")
            if irrelevant["target"] != decision("answer", primary["value"] + " days", [primary_id]):
                raise ValueError("Irrelevant-topic answer must cite only the applicable official fact")
            interventions["topic_filter"] += 1

            consensus = members[f"consensus_b{budget}"]
            replacement = official[secondary_id]["text"].replace(secondary["value"] + " days", primary["value"] + " days")
            if not _changed_only(conflict, consensus["observation"], secondary_id, replacement):
                raise ValueError("Consensus intervention must change only the second official value")
            if consensus["target"] != decision("answer", primary["value"] + " days", [primary_id, secondary_id]):
                raise ValueError("Consensus target must cite both agreeing official documents")
            interventions["conflict_agreement"] += 1

        row = members["missing_b0"]
        facts = [_fact(doc) for doc in row["observation"]["documents"] if doc["source"] == "official"]
        question_topic = next(p for _, p, topic in TOPICS if topic in row["observation"]["question"])
        uniform = len({fact["predicate"] for fact in facts}) == 1
        if uniform and any(fact["predicate"] == question_topic for fact in facts):
            raise ValueError("Uniform-topic missing case must contain only a wrong topic")
        missing[row["split"]]["uniform_wrong_topic" if uniform else "mixed_topic"] += 1

    for split, counts in missing.items():
        if not all(counts[name] for name in ("uniform_wrong_topic", "mixed_topic")) or abs(counts["uniform_wrong_topic"] - counts["mixed_topic"]) > 1:
            raise ValueError(f"Missing-topic subtypes must be balanced within {split}")

    probes = {}
    for split in ("train", "dev"):
        subset = [row for row in rows if row["split"] == split]
        probes[split] = {"v3_topic_and_count_omitter": _strict_probe(subset, False),
                         "patched_uniform_topic_omitter": _strict_probe(subset, True)}
        if any(probe["passes_all_seven"] for probe in probes[split].values()):
            raise ValueError(f"A known insufficient public rule still passes all seven gates on {split}")
    return {**base, "format": "evidence-gate-v31-independent-data-audit-v1",
            "variants": list(VARIANTS), "cells_per_group": len(VARIANTS),
            "single_factor_interventions": dict(interventions),
            "missing_topic_subtype_counts": {split: dict(counts) for split, counts in missing.items()},
            "strict_insufficient_rule_probes": probes,
            "limits": base["limits"] + " New strict insufficient-rule probes run only on train/dev. Their failure rejects the two specified omissions, not every possible shortcut. No model test prediction is generated or read."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/v31")
    parser.add_argument("--seen", nargs="+", default=["data/main", "data/v2", "data/v3"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        directory = Path(args.data)
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = [row for split in v3.SPLITS for row in read_jsonl(directory / f"{split}.jsonl")]
        seen = [row for source in args.seen for split in v3.SPLITS for row in read_jsonl(Path(source) / f"{split}.jsonl")]
        report = audit_v31(rows, seen, manifest["variants"])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in ("rows", "split_action_counts", "single_factor_interventions", "previously_seen_entity_overlap")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
