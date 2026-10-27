"""Exploratory train/dev diagnostics; never train, read test, or repair outputs.

Free generation receives only audited public messages. A separate teacher-forced
forward pass uses the target to measure conditional token losses, not predictions.
Candidate action probabilities are diagnostic only and never affect decoding.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from evidence_gate.cli import write_output
from evidence_gate.data import audit, digest, read_jsonl
from evidence_gate.evaluation import ACTIONS, evaluate
from evidence_gate.hf_policy import HFPolicy
from evidence_gate.training import sha256_file, target_text


def sample_rows(rows, per_action=8, seed=73):
    """Round-robin over shuffled groups before taking a second group variant."""
    if type(per_action) is not int or per_action < 1:
        raise ValueError("per_action must be positive")
    selected = []
    for action in ACTIONS:
        groups = defaultdict(list)
        for row in sorted(rows, key=lambda item: item["id"]):
            if row["target"]["action"] == action:
                groups[row["group_id"]].append(row)
        if sum(map(len, groups.values())) < per_action:
            raise ValueError(f"Not enough {action} rows for the declared sample")
        rng = random.Random(f"{seed}:{action}")
        names = sorted(groups)
        rng.shuffle(names)
        for members in groups.values():
            rng.shuffle(members)
        chosen = []
        while len(chosen) < per_action:
            for name in names:
                if groups[name]:
                    chosen.append(groups[name].pop())
                    if len(chosen) == per_action:
                        break
        selected.extend(chosen)
    return sorted(selected, key=lambda row: row["id"])


def encode_case(row, tokenizer):
    """Identify field spans using actual fast-tokenizer offsets, never guesses."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Exact token-field diagnostics require a fast tokenizer")
    completion = target_text(row["target"])
    prompt_text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    messages = row["messages"] + [{"role": "assistant", "content": completion}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full_ids[:len(prompt_ids)] != prompt_ids or not full_text.startswith(prompt_text + completion):
        raise ValueError("Training/inference prefix alignment failed")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != full_ids:
        raise ValueError("Rendered chat and tokenized chat differ")
    spans = {}
    for key in ("action", "value", "citations"):
        marker = '"' + key + '":'
        start = completion.index(marker) + len(marker)
        value = json.dumps(row["target"][key], ensure_ascii=False, separators=(",", ":"))
        # String values exclude surrounding quotes. Citation arrays include
        # brackets so that empty-list formatting has an explicit denominator.
        trim = 1 if isinstance(row["target"][key], str) else 0
        spans[key] = (len(prompt_text) + start + trim, len(prompt_text) + start + len(value) - trim)
    positions, fields = [], []
    for index in range(len(prompt_ids), len(full_ids)):
        start, stop = encoded["offset_mapping"][index]
        field = next((key for key, (left, right) in spans.items() if left < right and start < right and stop > left), "format")
        if full_ids[index] == tokenizer.eos_token_id:
            field = "eos"
        positions.append(index)
        fields.append(field)
    if tokenizer.eos_token_id not in full_ids[len(prompt_ids):] or "action" not in fields:
        raise ValueError("Completion must include EOS and an action span")
    action_index = positions[fields.index("action")]
    common_prefix = full_ids[:action_index]
    action_tokens = {}
    for action in ACTIONS:
        candidate = {**row["target"], "action": action}
        candidate_ids = tokenizer.apply_chat_template(row["messages"] + [{"role": "assistant", "content": target_text(candidate)}], tokenize=True, add_generation_prompt=False)
        if candidate_ids[:action_index] != common_prefix:
            raise ValueError("Actions do not share an identical label-free prefix")
        action_tokens[action] = candidate_ids[action_index]
    if len(set(action_tokens.values())) != len(ACTIONS):
        raise ValueError("First action token is not distinct for all four actions")
    return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            "positions": positions, "fields": fields, "action_index": action_index,
            "action_first_token_ids": action_tokens}


def teacher_forced(row, policy):
    torch, tokenizer = policy.torch, policy.tokenizer
    encoded = encode_case(row, tokenizer)
    maximum = getattr(policy.model.config, "max_position_embeddings", None)
    if maximum and len(encoded["input_ids"]) > maximum:
        raise ValueError("Diagnostic input exceeds model context; truncation is forbidden")
    ids = torch.tensor([encoded["input_ids"]], device=policy.device)
    labels = torch.tensor([encoded["labels"]], device=policy.device)
    with torch.inference_mode():
        output = policy.model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
        indices = torch.tensor([position - 1 for position in encoded["positions"]], device=policy.device)
        logits = output.logits[0].index_select(0, indices).float()
        gold = ids[0, torch.tensor(encoded["positions"], device=policy.device)]
        losses = torch.nn.functional.cross_entropy(logits, gold, reduction="none")
        argmax = logits.argmax(-1)
        model_loss = float(output.loss.item())
        manual_loss = float(losses.mean().item())
        loss_error = abs(model_loss - manual_loss)
        if not math.isfinite(manual_loss) or loss_error > 1e-5:
            raise ValueError(f"HF causal loss differs from manual shifted completion-only CE: {loss_error}")
        first = encoded["fields"].index("action")
        first_probs = logits[first].softmax(-1)
        candidate_probs = {action: float(first_probs[token_id].item()) for action, token_id in encoded["action_first_token_ids"].items()}
        top_id = int(argmax[first].item())
        first_top_probability = float(first_probs[top_id].item())
        loss_values, correct_values = losses.cpu().tolist(), (argmax == gold).cpu().tolist()
    tokens = [{"position": position, "token_id": encoded["input_ids"][position],
               "token": tokenizer.convert_ids_to_tokens(encoded["input_ids"][position]),
               "field": field, "nll": loss, "gold_probability": math.exp(-loss), "top1_correct": correct}
              for position, field, loss, correct in zip(encoded["positions"], encoded["fields"], loss_values, correct_values)]
    result = {"id": row["id"], "group_id": row["group_id"], "variant": row["variant"], "expected_action": row["target"]["action"],
              "input_tokens": len(encoded["input_ids"]), "completion_tokens": len(tokens),
              "hf_loss": model_loss, "manual_completion_nll": manual_loss, "loss_abs_error": loss_error,
              "action_first_token": {"gold_action": row["target"]["action"], "candidate_probabilities": candidate_probs,
                                     "candidate_token_ids": encoded["action_first_token_ids"],
                                     "top_token_id": top_id, "top_token": tokenizer.convert_ids_to_tokens(top_id),
                                     "top_probability": first_top_probability, "gold_top1": correct_values[first]},
              "tokens": tokens}
    del output, logits, gold, losses, argmax, first_probs, indices, ids, labels
    if policy.device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return result


def summarize_teacher(cases):
    def describe(members):
        tokens = [token for case in members for token in case["tokens"]]
        fields = {}
        for field in ("action", "value", "citations", "format", "eos"):
            part = [token for token in tokens if token["field"] == field]
            fields[field] = {"tokens": len(part), "mean_nll": sum(token["nll"] for token in part) / len(part) if part else None,
                             "top1_correct": sum(token["top1_correct"] for token in part)}
        return {"examples": len(members), "supervised_tokens": len(tokens),
                "completion_token_mean_nll": sum(token["nll"] for token in tokens) / len(tokens),
                "action_first_token_correct": sum(case["action_first_token"]["gold_top1"] for case in members),
                "max_loss_abs_error": max(case["loss_abs_error"] for case in members), "fields": fields}
    return {"overall": describe(cases), "by_action": {action: describe([case for case in cases if case["expected_action"] == action]) for action in ACTIONS}}


def diagnose(rows, policy, name, batch_size=4):
    # Run free generation first and pass only public messages through evaluate.
    free = evaluate(rows, policy, name, kind="model", batch_size=batch_size)
    teacher = []
    for row in rows:
        teacher.append(teacher_forced(row, policy))
        if len(teacher) % 8 == 0 or len(teacher) == len(rows):
            print(f"{name}: teacher forcing {len(teacher)}/{len(rows)}", flush=True)
    return {"free_generation": free, "teacher_forced": {"summary": summarize_teacher(teacher), "cases": teacher}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--per-action", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Output exists; diagnostics must not overwrite prior evidence")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    sources = {split: read_jsonl(getattr(args, split)) for split in ("train", "dev")}
    for split, rows in sources.items():
        if not rows or any(row.get("split") != split for row in rows):
            parser.error(f"The {split} input must contain only split={split}; test data is forbidden")
    audit(sources["train"] + sources["dev"])
    samples = {split: sample_rows(rows, args.per_action, args.seed) for split, rows in sources.items()}
    policy = HFPolicy(args.model, args.adapter, args.device, args.max_new_tokens, dtype=args.dtype)
    try:
        expected_train = policy.provenance.get("training_data_sha256")
        if expected_train and expected_train != sha256_file(Path(args.train)):
            raise ValueError("Train diagnostic input differs from the adapter's actual training dataset")
        result = {
            "format": "evidence-gate-exploratory-diagnostic-v1", "exploratory": True, "test_read": False,
            "model_provenance": policy.provenance,
            "sampling": {"seed": args.seed, "per_action_per_split": args.per_action,
                         "rule": "Round-robin groups before reusing a group within an action; no selection by model outputs"},
            "interpretation": "Balanced exploratory sample, not an unbiased benchmark. Teacher-forced losses condition on correct prior output tokens; they are not free-generation accuracy. Candidate probabilities never alter decoding.",
            "inference": {"device": args.device, "dtype": args.dtype, "batch_size": args.batch_size,
                          "decoding": "greedy free generation", "max_new_tokens": args.max_new_tokens, "output_repair": False},
            "splits": {},
        }
        for split, rows in samples.items():
            result["splits"][split] = {
                "source_file_sha256": sha256_file(Path(getattr(args, split))),
                "source_dataset_sha256": digest(sources[split]), "selected_dataset_sha256": digest(rows),
                "selected_ids": [row["id"] for row in rows], "selected_groups": sorted({row["group_id"] for row in rows}),
                **diagnose(rows, policy, split, args.batch_size),
            }
        write_output(args.output, result)
    finally:
        policy.close()
    print(json.dumps({split: {"free": value["free_generation"]["summary"]["success_rate"], "teacher": value["teacher_forced"]["summary"]["overall"]} for split, value in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
