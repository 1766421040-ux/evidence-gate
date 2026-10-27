"""Freeze two paired diagnostics without model calls or training-set access.

Paraphrases preserve the main test targets. Redaction changes the observation and
therefore recomputes its contract target; it is not an additional main accuracy
sample. Both diagnostics share four original test groups and must not be pooled
with the main test as statistically independent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from evidence_gate.data import VARIANTS, audit, decision, messages_for, read_jsonl, write_jsonl


QUESTION = re.compile(
    r"^For (?P<product>[A-Z]+-\d+)(?: on the (?P<plan>Starter|Team|Business) plan)?, "
    r"how many days is the (?P<topic>audit-log retention|deleted-file recovery|backup retention) period\?$"
)
FACT = re.compile(
    r"^On (?P<product>[A-Z]+-\d+)'s (?P<plan>Starter|Team|Business) plan, "
    r"(?P<predicate>audit logs are retained for|deleted files can be restored for|workspace backups are stored for) "
    r"(?P<days>\d+) days\.$"
)
TOPICS = {"audit-log retention", "deleted-file recovery", "backup retention"}


def _question_parts(text: str) -> dict:
    match = QUESTION.fullmatch(text)
    if not match:
        raise ValueError("Unsupported source question; inspect semantic preservation before changing templates")
    return match.groupdict()


def paraphrase_question(text: str) -> str:
    fields = _question_parts(text)
    product, plan, topic = fields["product"], fields["plan"], fields["topic"]
    customers = f"customers subscribed to {plan}" if plan else "its customers"
    if topic == "audit-log retention":
        return f"How long does {product} keep audit logs for {customers}?"
    customers = f"{product} customers subscribed to {plan}" if plan else f"{product} customers"
    if topic == "deleted-file recovery":
        return f"How many days do {customers} have to recover deleted files?"
    return f"What is the number of days that workspace backups remain available to {customers}?"


def paraphrase_document(document: dict) -> dict:
    result = copy.deepcopy(document)
    if document["source"] != "official":
        return result
    match = FACT.fullmatch(document["text"])
    if not match:
        # Unrelated custom-theme documents are deliberately unchanged. Refuse
        # unfamiliar retention claims instead of silently dropping meaning.
        if "custom themes are available." in document["text"]:
            return result
        raise ValueError("Unsupported official source document; manual semantic review required")
    product, plan, days = match["product"], match["plan"], match["days"]
    if match["predicate"] == "audit logs are retained for":
        text = f"The {plan} subscription to {product} has an audit-log retention window of {days} days."
    elif match["predicate"] == "deleted files can be restored for":
        text = f"Under the {plan} subscription to {product}, the deleted-file recovery window lasts {days} days."
    else:
        text = f"For {product} subscribers on {plan}, workspace backups remain available for {days} days."
    result["text"] = text
    return result


def select_groups(rows: list[dict], count: int = 4) -> list[dict]:
    if type(count) is not int or count < 1:
        raise ValueError("group count must be a positive integer")
    audit(rows)
    if any(row["split"] != "test" for row in rows):
        raise ValueError("Diagnostics may only be built from the locked test split")
    group_ids = sorted({row["group_id"] for row in rows})
    if len(group_ids) < count:
        raise ValueError("Not enough test groups")
    chosen = group_ids[:count]
    selected = []
    for group_id in chosen:
        members = [row for row in rows if row["group_id"] == group_id]
        if Counter(row["variant"] for row in members) != Counter(VARIANTS):
            raise ValueError(f"Incomplete diagnostic group: {group_id}")
        selected.extend(sorted(members, key=lambda row: VARIANTS.index(row["variant"])))
    if {_question_parts(row["observation"]["question"])["topic"] for row in selected} != TOPICS:
        raise ValueError("The selected diagnostic groups must cover all three topics")
    return selected


def transform_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    paraphrases, redactions = [], []
    for original in rows:
        paraphrase = copy.deepcopy(original)
        observation = paraphrase["observation"]
        observation["question"] = paraphrase_question(observation["question"])
        observation["documents"] = [paraphrase_document(document) for document in observation["documents"]]
        paraphrase["messages"] = messages_for(observation)
        paraphrase["diagnostic"] = "paraphrase"
        paraphrases.append(paraphrase)

        redacted = copy.deepcopy(original)
        observation = redacted["observation"]
        observation["documents"] = []
        # Plan presence is read from the public question, never gold or variant.
        if _question_parts(observation["question"])["plan"] is None:
            action = "clarify"
        else:
            action = "search" if observation["budget"] > 0 else "abstain"
        redacted["target"] = decision(action)
        redacted["messages"] = messages_for(observation)
        redacted["diagnostic"] = "redacted"
        redactions.append(redacted)
    audit(paraphrases)
    audit(redactions)
    return paraphrases, redactions


def build(source: str | Path, output: str | Path, group_count: int = 4) -> dict:
    source, output = Path(source), Path(output)
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    selected = select_groups(read_jsonl(source), group_count)
    paraphrases, redactions = transform_rows(selected)
    if source.read_bytes() != source_bytes:
        raise RuntimeError("Source changed while building diagnostics")
    paths = [output / name for name in ("paraphrase.jsonl", "redacted.jsonl", "manifest.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Locked diagnostic files already exist; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("paraphrase.jsonl", paraphrases), ("redacted.jsonl", redactions)):
        write_jsonl(output / filename, rows)
    manifest = {
        "schema_version": 1,
        "generator": "evidence-gate-diagnostics-v1",
        "source_file": source.name,
        "source_file_sha256": source_hash,
        "selection": "lexicographically first complete test groups, all 14 variants per group",
        "group_ids": sorted({row["group_id"] for row in selected}),
        "group_count": group_count,
        "rows_per_diagnostic": len(selected),
        "topics": sorted(TOPICS),
        "models_called": 0,
        "training_data_read": False,
        "locked_use": "Held-out diagnostic only; do not tune training, prompts, or checkpoints on these results.",
        "independent_of_main_test": False,
        "statistical_unit": "Original group_id. These rows share test groups with main and each other; do not pool as independent observations.",
        "paraphrase": {
            "changes": "Question and relevant official-document wording only; source, dates, budget, values, identities, targets and citations are preserved.",
            "unchanged": "Community documents, unrelated custom-theme documents, system instruction, split and private gold metadata.",
            "interpretation": "A small fixed wording-shift probe, not proof of broad natural-language generalization.",
        },
        "redacted": {
            "changes": "Original question retained, all visible documents removed, contract target recomputed from observable plan and budget.",
            "target_policy": "Missing plan -> clarify; explicit plan and budget>0 -> search; explicit plan and budget=0 -> abstain.",
            "interpretation": "Evidence-dependence diagnostic, not accuracy against original gold and not a stand-alone SFT gain claim.",
        },
        "metadata_note": "diagnostic, variant, group_id, id, target and gold are outside model messages. Variant names describe the source case.",
        "files": {
            name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(), "audit": audit(rows)}
            for name, rows in (("paraphrase.jsonl", paraphrases), ("redacted.jsonl", redactions))
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/main/test.jsonl")
    parser.add_argument("--output", default="data/diagnostics")
    parser.add_argument("--groups", type=int, default=4)
    args = parser.parse_args(argv)
    manifest = build(args.source, args.output, args.groups)
    print(json.dumps({"groups": manifest["group_ids"], "rows_per_diagnostic": manifest["rows_per_diagnostic"], "source_file_sha256": manifest["source_file_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
