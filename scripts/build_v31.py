"""Build/freeze V3.1: fresh entities, 24 cells, explicit topic/agreement pairs.

CPU only; preserves every older frozen artifact. No model output is read.
    python scripts/build_v31.py --output data/v31 --metadata experiments/v31
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

from evidence_gate.data import TOPICS, audit, messages_for, read_jsonl, write_jsonl
from evidence_gate.evaluation import contract_rule


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261030
SPLITS = ("train", "dev", "test")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


v3 = module("frozen_v3_for_v31", "scripts/build_v3.py")
VARIANTS = (*v3.VARIANTS, "irrelevant_topic_b0", "irrelevant_topic_b1", "consensus_b0", "consensus_b1")
FACT = re.compile(r"^On (\S+)'s (\w+) plan, (.*?) (\d+) days\.$")
sha256 = v3.sha256
public_path = v3.public_path


def _replace_fact(text, *, predicate=None, value=None):
    match = FACT.fullmatch(text)
    if match is None:
        raise ValueError("Unexpected official fact grammar")
    entity, plan, old_predicate, old_value = match.groups()
    return f"On {entity}'s {plan} plan, {predicate or old_predicate} {value or old_value} days."


def generate(seed=SEED, train_groups=80, dev_groups=24, test_groups=60):
    sizes = dict(zip(SPLITS, (train_groups, dev_groups, test_groups)))
    if type(seed) is not int or any(type(count) is not int or count < 12 or count % 2 for count in sizes.values()):
        raise ValueError("V3.1 needs an integer seed and even split sizes of at least twelve groups")
    rows = []
    for split, count in sizes.items():
        domain = f"evidence-gate-v31:{seed}:{split}"
        rng = random.Random(f"{domain}:schedule")
        combinations = [(product, related, number) for product in (False, True)
                        for related in (False, True) for number in (0, 1, 2)]
        factors = [combinations[index % 12] for index in range(count)]
        signatures = [v3.v2.SIGNATURES[index % 3] for index in range(count)]
        uniform = [bool(index % 2) for index in range(count)]
        topic_offsets = [1 + index % 2 for index in range(count)]
        for schedule in (factors, signatures, uniform, topic_offsets):
            rng.shuffle(schedule)
        for index in range(count):
            base = v3.generate_group(index, split, domain, signatures[index], *factors[index])
            group_id = f"v31-{split}-g{index:04d}"
            for row in base:
                row["group_id"] = group_id
                row["id"] = f"{group_id}:{row['variant']}"
                row["design"].update(missing_uniform_topic_factor=uniform[index],
                                     irrelevant_topic_offset=topic_offsets[index])
            lookup = {row["variant"]: row for row in base}
            primary_id = lookup["supported_b0"]["target"]["citations"][0]
            predicate = FACT.fullmatch(next(doc["text"] for doc in lookup["supported_b0"]["observation"]["documents"] if doc["id"] == primary_id)).group(3)
            predicates = [topic[1] for topic in TOPICS]
            other_predicate = predicates[(predicates.index(predicate) + topic_offsets[index]) % len(predicates)]
            if uniform[index]:
                for budget in (0, 1):
                    row = lookup[f"missing_b{budget}"]
                    missing_predicate = FACT.fullmatch(next(doc["text"] for doc in row["observation"]["documents"] if doc["id"] == primary_id)).group(3)
                    for document in row["observation"]["documents"]:
                        if document["source"] == "official" and document["id"] != primary_id:
                            document["text"] = _replace_fact(document["text"], predicate=missing_predicate)
                    row["messages"] = messages_for(row["observation"])
            additions = []
            for semantic in ("irrelevant_topic", "consensus"):
                for budget in (0, 1):
                    row = copy.deepcopy(lookup[f"conflict_b{budget}"])
                    observation = row["observation"]
                    primary = next(doc for doc in observation["documents"] if doc["id"] == primary_id)
                    primary_value = FACT.fullmatch(primary["text"]).group(4)
                    secondary = next(doc for doc in observation["documents"] if doc["source"] == "official" and doc["id"] != primary_id)
                    secondary["text"] = (_replace_fact(secondary["text"], predicate=other_predicate)
                                         if semantic == "irrelevant_topic" else _replace_fact(secondary["text"], value=primary_value))
                    row["variant"] = f"{semantic}_b{budget}"
                    row["id"] = f"{group_id}:{row['variant']}"
                    row["target"] = contract_rule(observation)
                    row["messages"] = messages_for(observation)
                    # Mirror this public state with separate authenticated IDs;
                    # a two-citation answer never cites hidden visible-world IDs.
                    old_recovery_ids = [doc["id"] for doc in row["gold"]["catalogue"] if doc["source"] == "official"]
                    catalogue = [copy.deepcopy(primary), copy.deepcopy(secondary)]
                    for document, identifier in zip(catalogue, old_recovery_ids):
                        document["id"] = identifier
                    catalogue.append(copy.deepcopy(next(doc for doc in observation["documents"] if doc["source"] == "community")))
                    recovered = {**observation, "documents": catalogue}
                    row["gold"].update(catalogue=catalogue, recoverable=False,
                                       recovery_target=contract_rule(recovered))
                    additions.append(row)
            rows.extend([*base, *additions])
    return rows


def validate(rows, seen_rows=()):
    # Reuse unchanged validation of the inherited twenty-cell semantics. IDs
    # are projected privately only for that old version's namespace assertion.
    projected = []
    for row in rows:
        if re.fullmatch(rf"v31-{row['split']}-g\d{{4,}}", row["group_id"]) is None or row["id"] != f"{row['group_id']}:{row['variant']}":
            raise ValueError("V3.1 identity must bind its version, split and cell")
        if row["variant"] in v3.VARIANTS:
            copied = copy.deepcopy(row)
            copied["group_id"] = copied["group_id"].replace("v31-", "v3-", 1)
            copied["id"] = f"{copied['group_id']}:{copied['variant']}"
            projected.append(copied)
    v3.validate(projected, v3.entities(seen_rows))
    independent = module("independent_v31_audit", "scripts/audit_v31_data.py")
    return independent.audit_v31(rows, seen_rows, expected_variants=VARIANTS)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build(output="data/v31", metadata="experiments/v31", sources=("data/main", "data/v2", "data/v3"),
          seed=SEED, train_groups=80, dev_groups=24, test_groups=60):
    output, metadata = Path(output), Path(metadata)
    if output.exists() or any((metadata / name).exists() for name in ("protocol.json", "freeze.json", "data-audit.json", "supersession.json")):
        raise FileExistsError("V3.1 output is already frozen; choose new data and metadata directories")
    sources = [Path(path) for path in sources]
    if len(sources) != 3 or {path.name for path in sources} != {"main", "v2", "v3"}:
        raise ValueError("V3.1 isolation requires main, v2 and v3 source directories")
    snapshots, seen_rows = {}, []
    for directory in sources:
        for split in SPLITS:
            path = directory / f"{split}.jsonl"
            snapshots[path] = path.read_bytes()
            prior = read_jsonl(path)
            if any(row["split"] != split for row in prior):
                raise ValueError("Previously seen source split mismatch")
            seen_rows.extend(prior)
    rows = generate(seed, train_groups, dev_groups, test_groups)
    independent = validate(rows, seen_rows)
    if any(path.read_bytes() != value for path, value in snapshots.items()):
        raise ValueError("Older frozen data changed during V3.1 construction")
    created = datetime.now(timezone.utc).isoformat()
    prior_protocol = ROOT / "experiments/v3/protocol.json"
    plan = ROOT / "experiments/v31/acceptance-plan.md"
    if not plan.is_file():
        raise ValueError("Write the V3.1 acceptance addendum before freezing")
    protocol = copy.deepcopy(json.loads(prior_protocol.read_text()))
    code_paths = [Path(__file__), ROOT / "scripts/audit_v31_data.py", ROOT / "scripts/build_v3.py",
                  ROOT / "scripts/audit_v3_data.py", ROOT / "scripts/build_v2.py", ROOT / "src/evidence_gate/data.py",
                  ROOT / "src/evidence_gate/evaluation.py", ROOT / "src/evidence_gate/agent.py"]
    protocol.update(format="evidence-gate-v31-frozen-holdout-protocol-v1", created_at=created,
                    reason="V3 train/dev-only static audit found topic-mixture and single-candidate shortcuts; supersede before fresh-holdout model confirmation, keeping all V3 records.",
                    data_seed=seed, train_groups=train_groups, dev_groups=dev_groups, test_groups=test_groups,
                    variants=list(VARIANTS), cells_per_group=len(VARIANTS),
                    acceptance_plan_sha256=sha256(plan),
                    generator_code_sha256={public_path(path): sha256(path) for path in code_paths},
                    preceding_v3_protocol_sha256=sha256(prior_protocol),
                    confirmation_scope="Fresh product groups on the unchanged evidence contract, now including irrelevant current topic and two agreeing current officials. Not a natural-language or production safety certification.",
                    pair_reporting="Primary summarize/score remains unchanged. Topic-filter and conflict-agreement matched-budget pairs are separately audited; pair diagnostics must never overwrite stored main summaries.")
    protocol["test_policy"] = protocol["test_policy"].replace("V3 test", "V3.1 test")
    output.mkdir(parents=True, exist_ok=False)
    metadata.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    protocol["data_sha256"] = {split: sha256(output / f"{split}.jsonl") for split in SPLITS}
    _write(metadata / "protocol.json", protocol)
    _write(metadata / "data-audit.json", independent)
    supersession = {"format": "evidence-gate-v31-supersession-v1", "created_at": created,
                    "supersedes": "experiments/v3/protocol.json", "superseded_protocol_sha256": sha256(prior_protocol),
                    "superseded_freeze_sha256": sha256(ROOT / "experiments/v3/freeze.json"),
                    "reasoning_evidence": {"experiments/v3/train-dev-audit.json": sha256(ROOT / "experiments/v3/train-dev-audit.json")},
                    "decision_inputs": "V3 train/dev and generator-source CPU coverage audit only; no V3 dev/test model prediction used.",
                    "old_artifacts_modified": False,
                    "new_identity_policy": "New seed and split domains; all queried/distractor entities disjoint from main/v2/v3 across every split.",
                    "not_a_model_result": True}
    _write(metadata / "supersession.json", supersession)
    manifest = {**audit(rows), "format": "evidence-gate-v31-data-manifest-v1", "created_at": created,
                "seed": seed, "generator": "build_v31.py", "license": "MIT", "variants": list(VARIANTS),
                "cells_per_group": len(VARIANTS), "models_called": 0, "model_predictions_read": False,
                "test_model_inference_performed": False, "original_message_projection_unchanged": True,
                "previously_seen_files_sha256": {f"{path.parent.name}/{path.name}": hashlib.sha256(value).hexdigest() for path, value in snapshots.items()},
                "protocol_sha256": sha256(metadata / "protocol.json"),
                "independent_audit_sha256": sha256(metadata / "data-audit.json"),
                "generator_code_sha256": protocol["generator_code_sha256"],
                "files": {f"{split}.jsonl": {"sha256": sha256(output / f"{split}.jsonl"),
                           "rows": sum(row["split"] == split for row in rows),
                           "actions": dict(Counter(row["target"]["action"] for row in rows if row["split"] == split))} for split in SPLITS},
                "scope": "24 correlated cells per product group. Strict score and model prompt unchanged; no model or deployment result established."}
    _write(output / "manifest.json", manifest)
    paths = [metadata / "protocol.json", metadata / "data-audit.json", metadata / "supersession.json",
             output / "manifest.json", plan, *code_paths, *(output / f"{split}.jsonl" for split in SPLITS)]
    _write(metadata / "freeze.json", {"format": "evidence-gate-v31-freeze-v1", "created_at": created,
                                      "model_predictions_read": False, "test_model_inference_performed": False,
                                      "files": {public_path(path): sha256(path) for path in paths},
                                      "meaning": "Data/contract/code hashes; no claim of model quality or secret test access control."})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/v31")
    parser.add_argument("--metadata", default="experiments/v31")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-groups", type=int, default=80)
    args = parser.parse_args(argv)
    try:
        result = build(args.output, args.metadata, seed=args.seed, train_groups=args.train_groups)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in ("seed", "groups", "split_rows", "files")}, indent=2))


if __name__ == "__main__":
    main()
