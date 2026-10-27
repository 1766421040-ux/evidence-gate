"""Small, auditable PEFT SFT runs; all checkpoints use JSON + safetensors.

The model sees only ``messages`` and the target completion. Evaluation metadata
never enters the tokenizer. Training requires a downloaded local model snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any


DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "seed": 42,
    "device": "cpu",
    "dtype": "float32",
    "max_steps": 200,
    "batch_size": 2,
    "gradient_accumulation_steps": 2,
    "max_length": 768,
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "warmup_steps": 5,
    "max_grad_norm": 1.0,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "save_steps": 50,
    "gradient_checkpointing": False,
    "attention_implementation": "sdpa",
    "num_threads": 4,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_config(config: dict | str | Path) -> dict:
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary or a JSON file")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **config}
    for key in ("max_steps", "batch_size", "gradient_accumulation_steps", "max_length", "lora_rank", "save_steps", "num_threads"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(result["seed"]) is not int or type(result["warmup_steps"]) is not int or result["warmup_steps"] < 0:
        raise ValueError("seed and warmup_steps must be integers; warmup_steps must be non-negative")
    if result["device"] not in ("cpu", "mps", "cuda") or result["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Unsupported device or dtype")
    if not isinstance(result["model_name"], str) or not result["model_name"] or Path(result["model_name"]).is_absolute():
        raise ValueError("model_name must be a public identifier, not an absolute local path")
    if not isinstance(result["target_modules"], list) or not result["target_modules"] or not all(isinstance(x, str) and x for x in result["target_modules"]):
        raise ValueError("target_modules must be a nonempty list")
    for key in ("learning_rate", "lora_alpha", "max_grad_norm"):
        if not isinstance(result[key], (float, int)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= result["lora_dropout"] < 1 or result["weight_decay"] < 0:
        raise ValueError("Invalid dropout or weight decay")
    return result


def target_text(target: dict) -> str:
    if not isinstance(target, dict) or set(target) != {"action", "value", "citations"}:
        raise ValueError("target must contain exactly action, value, citations")
    if target["action"] not in {"answer", "search", "clarify", "abstain"}:
        raise ValueError("Unknown target action")
    if not isinstance(target["value"], str) or not isinstance(target["citations"], list) or not all(isinstance(x, str) for x in target["citations"]):
        raise ValueError("target value must be text and citations must be a list of strings")
    # Stable field order is also the format used at inference.
    return json.dumps({key: target[key] for key in ("action", "value", "citations")}, ensure_ascii=False, separators=(",", ":"))


def prepare_examples(rows: list[dict], tokenizer: Any, max_length: int) -> tuple[list[dict], dict]:
    """Apply the actual chat template and supervise completion, including EOS.

    Refuses truncation: cutting a citation, stop token, or governing source would
    change the task. The caller must increase the budget or fix the data.
    """
    if not rows:
        raise ValueError("Training dataset is empty")
    examples, seen_ids, lengths, supervised = [], set(), [], []
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must declare an EOS token")
    for number, row in enumerate(rows, 1):
        if row.get("split") != "train":
            raise ValueError(f"Row {number}: only split=train may enter training")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError(f"Row {number}: missing or duplicate id")
        seen_ids.add(row_id)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
            raise ValueError(f"Row {number}: messages must end with a user turn")
        if any(not isinstance(m, dict) or set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant", "tool"} or not isinstance(m["content"], str) for m in messages):
            raise ValueError(f"Row {number}: invalid messages")
        completion = target_text(row.get("target"))
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": completion}], tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt:
            raise ValueError(f"Row {number}: training/inference chat template prefix mismatch")
        answer_tokens = full[len(prompt):]
        if not answer_tokens or eos not in answer_tokens:
            raise ValueError(f"Row {number}: completion mask omits EOS")
        if len(full) > max_length:
            raise ValueError(f"Row {number}: {len(full)} tokens exceeds max_length={max_length}; silent truncation is forbidden")
        examples.append({"input_ids": full, "labels": [-100] * len(prompt) + answer_tokens})
        lengths.append(len(full))
        supervised.append(len(answer_tokens))
    return examples, {
        "examples": len(examples), "max_sequence_tokens": max(lengths),
        "mean_sequence_tokens": sum(lengths) / len(lengths),
        "supervised_tokens": sum(supervised), "total_tokens": sum(lengths),
        "supervised_fraction": sum(supervised) / sum(lengths),
        "truncated_examples": 0, "eos_supervised_examples": len(examples),
    }


def _model_manifest(model_path: Path) -> dict:
    if not model_path.is_dir():
        raise ValueError("Training requires a local downloaded model snapshot, not an unpinned Hub identifier")
    files = sorted(p for p in model_path.rglob("*") if p.is_file() and (p.suffix in {".safetensors", ".json", ".model"} or p.name in {"merges.txt", "vocab.txt", "tokenizer_config.json"}) and ".cache" not in p.relative_to(model_path).parts)
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local model snapshot must contain safetensors weights")
    records = {str(p.relative_to(model_path)): sha256_file(p) for p in files}
    return {"sha256": hashlib.sha256(_json_bytes(records)).hexdigest(), "files": records}


def _encode_state(value: Any, tensors: dict, torch: Any) -> Any:
    """Typed JSON tree keeps optimizer integer keys and tuples without pickle."""
    if torch.is_tensor(value):
        name = f"tensor_{len(tensors):06d}"
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {"kind": "tensor", "name": name}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[_encode_state(k, tensors, torch), _encode_state(v, tensors, torch)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [_encode_state(v, tensors, torch) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def _decode_state(value: dict, tensors: dict) -> Any:
    kind = value["kind"]
    if kind == "tensor":
        return tensors[value["name"]]
    if kind == "scalar":
        return value["value"]
    if kind in {"list", "tuple"}:
        items = [_decode_state(v, tensors) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {_decode_state(k, tensors): _decode_state(v, tensors) for k, v in value["items"]}
    raise ValueError("Unknown checkpoint state kind")


def _rng_state(torch: Any, device: str) -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device == "mps":
        state["device"] = torch.mps.get_rng_state()
    elif device == "cuda":
        state["device"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, torch: Any, device: str) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device == "mps":
        torch.mps.set_rng_state(state["device"])
    elif device == "cuda":
        torch.cuda.set_rng_state_all(state["device"])


def _sanitize_adapter(path: Path, model_name: str) -> None:
    filename = path / "adapter_config.json"
    config = json.loads(filename.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = model_name
    _write_json(filename, config)
    # PEFT's generated card may contain the local source path. The project has a
    # dedicated model card; do not publish a machine-dependent generated card.
    (path / "README.md").unlink(missing_ok=True)


def _save_checkpoint(output: Path, model: Any, optimizer: Any, scheduler: Any, binding: dict, step: int, cursor: dict, history: list, torch: Any, config: dict) -> Path:
    from safetensors.torch import save_file
    folder = output / "checkpoints" / f"step-{step:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    # No tokens/embeddings are added by this trainer. Explicit False also avoids
    # PEFT's automatic Hub vocabulary-size lookup when resuming a public adapter.
    model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
    _sanitize_adapter(folder, config["model_name"])
    tensors: dict = {}
    tree = _encode_state({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": _rng_state(torch, config["device"])}, tensors, torch)
    save_file(tensors, str(folder / "training_state.safetensors"))
    metadata = {
        "format": "evidence-gate-checkpoint-v1", "binding": binding,
        "step": step, "cursor": cursor, "history": history, "state": tree,
        "files": {name: sha256_file(folder / name) for name in ("adapter_model.safetensors", "adapter_config.json", "training_state.safetensors")},
    }
    _write_json(folder / "checkpoint.json", metadata)
    return folder


def _read_checkpoint(path: Path, output: Path, binding: dict) -> dict:
    if not path.resolve().is_relative_to((output / "checkpoints").resolve()):
        raise ValueError("Resume accepts only this run's own output_dir/checkpoints directory")
    metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "evidence-gate-checkpoint-v1" or metadata.get("binding") != binding:
        raise ValueError("Checkpoint is not this run's format, data, config, or model")
    expected = {"adapter_model.safetensors", "adapter_config.json", "training_state.safetensors"}
    if set(metadata.get("files", {})) != expected:
        raise ValueError("Checkpoint file manifest is invalid")
    for name, digest in metadata["files"].items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return metadata


def _next_indices(size: int, batch_size: int, seed: int, cursor: dict) -> list[int]:
    indices = []
    while len(indices) < batch_size:
        permutation = list(range(size))
        random.Random(seed + cursor["epoch"]).shuffle(permutation)
        count = min(batch_size - len(indices), size - cursor["offset"])
        indices.extend(permutation[cursor["offset"]:cursor["offset"] + count])
        cursor["offset"] += count
        if cursor["offset"] == size:
            cursor["epoch"] += 1
            cursor["offset"] = 0
    return indices


def _batch(examples: list[dict], indices: list[int], pad_token: int, torch: Any, device: str, pad_to_length: int | None = None) -> dict:
    selected = [examples[i] for i in indices]
    longest = max(len(x["input_ids"]) for x in selected)
    width = longest if pad_to_length is None else pad_to_length
    if width < longest:
        raise ValueError("Fixed padding length cannot truncate examples")
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad_token] * (width - len(x["input_ids"])) for x in selected], device=device),
        "labels": torch.tensor([x["labels"] + [-100] * (width - len(x["labels"])) for x in selected], device=device),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (width - len(x["input_ids"])) for x in selected], device=device),
    }


def train(data_path: str | Path, model_path: str | Path, output_dir: str | Path, config: dict | str | Path, resume: str | Path | None = None) -> dict:
    """Train a genuine causal LM LoRA adapter; returns a path-free run summary.

    ``resume`` is a completed optimizer-step checkpoint from this same run.
    Data/config/model hashes and file checksums must match. No pickle is loaded.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = normalize_config(config)
    data_path, model_path, output = Path(data_path), Path(model_path), Path(output_dir)
    device = config["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process; no silent CPU fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; no silent CPU fallback")
    if output.exists() and (output / "run_manifest.json").exists() and resume is None:
        raise ValueError("Output already contains a run; choose a new directory or explicitly resume")
    memory_backend = getattr(torch, device, None) if device in {"mps", "cuda"} else None
    if memory_backend is not None and hasattr(memory_backend, "reset_peak_memory_stats"):
        memory_backend.reset_peak_memory_stats()
    torch.set_num_threads(config["num_threads"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device == "mps":
        torch.mps.manual_seed(config["seed"])
    elif device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Validate the canonical public-observation boundary at the actual training
    # entry point. A stored messages field must not smuggle gold labels or other
    # metadata into a prompt even if its chat-template masks are otherwise valid.
    from .data import audit as audit_dataset
    contract_audit = audit_dataset(rows)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Only split=train may enter training")
    model_snapshot = _model_manifest(model_path)
    binding = {
        "data_sha256": sha256_file(data_path),
        "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
        "model_sha256": model_snapshot["sha256"],
    }
    metadata = _read_checkpoint(Path(resume), output, binding) if resume else None
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, audit = prepare_examples(rows, tokenizer, config["max_length"])
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, config["dtype"]), attn_implementation=config["attention_implementation"])
    model.config.use_cache = False
    if resume:
        model = PeftModel.from_pretrained(model, str(resume), is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules=config["target_modules"]))
    model.to(device)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def schedule(step: int) -> float:
        if step < config["warmup_steps"]:
            return (step + 1) / max(1, config["warmup_steps"])
        return max(0.0, (config["max_steps"] - step) / max(1, config["max_steps"] - config["warmup_steps"]))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    cursor, history, start_step = {"epoch": 0, "offset": 0}, [], 0
    if metadata:
        from safetensors.torch import load_file
        state = _decode_state(metadata["state"], load_file(str(Path(resume) / "training_state.safetensors")))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"], torch, device)
        cursor, history, start_step = metadata["cursor"], metadata["history"], metadata["step"]
        if not 0 <= start_step <= config["max_steps"] or not 0 <= cursor["offset"] < len(examples) or cursor["epoch"] < 0:
            raise ValueError("Checkpoint cursor is invalid")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "evidence-gate-training-v1", "model_name": config["model_name"],
        "binding": binding, "config": config, "model_files": model_snapshot["files"], "data_audit": audit,
        "dataset_contract_audit": contract_audit,
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__},
        "supervision": "final assistant completion and EOS only; no truncation",
        "batch_padding": "Fixed max_length shape to bound MPS graph-cache shape diversity; padding labels are masked",
    }
    _write_json(output / "run_manifest.json", manifest)
    started = time.perf_counter()
    previous_elapsed = history[-1]["elapsed_s"] if history else 0.0
    for step in range(start_step, config["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        weighted_loss, supervised_count, token_count = 0.0, 0, 0
        batches = []
        for _ in range(config["gradient_accumulation_steps"]):
            indices = _next_indices(len(examples), config["batch_size"], config["seed"], cursor)
            batches.append(_batch(examples, indices, tokenizer.pad_token_id, torch, device, config["max_length"]))
        total_supervised = sum(int((batch["labels"][:, 1:] != -100).sum().item()) for batch in batches)
        for batch in batches:
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss = model(**batch).loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
            (loss * (count / total_supervised)).backward()
            weighted_loss += float(loss.detach().item()) * count
            supervised_count += count
            token_count += int(batch["attention_mask"].sum().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        grad_norm_value = float(grad_norm)
        # The last loss still holds a reference to its autograd graph after
        # backward. Do not retain it while constructing the next forward pass.
        del loss, batch, batches, grad_norm
        optimizer.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
            if (step + 1) % 10 == 0:
                # Releases only unused cache; no allocator watermarks are raised.
                torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.synchronize()
        history.append({
            "step": step + 1, "loss": weighted_loss / supervised_count,
            "learning_rate": learning_rate, "grad_norm": grad_norm_value,
            "supervised_tokens": supervised_count, "input_tokens": token_count,
            "elapsed_s": previous_elapsed + time.perf_counter() - started,
        })
        # Rewrite a small bounded log, so a resumed checkpoint cannot leave later
        # metrics from an abandoned continuation in the experiment record.
        _write_json(output / "metrics.json", history)
        if step == start_step or (step + 1) % 10 == 0 or step + 1 == config["max_steps"]:
            print(json.dumps({"step": step + 1, "total_steps": config["max_steps"], "loss": round(history[-1]["loss"], 6), "elapsed_s": round(history[-1]["elapsed_s"], 2)}), flush=True)
        if (step + 1) % config["save_steps"] == 0 or step + 1 == config["max_steps"]:
            _save_checkpoint(output, model, optimizer, scheduler, binding, step + 1, cursor.copy(), history.copy(), torch, config)

    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    _sanitize_adapter(adapter, config["model_name"])
    tokenizer.save_pretrained(adapter)
    summary = {
        "format": "evidence-gate-training-summary-v1", "model_name": config["model_name"],
        "device": device, "dtype": config["dtype"], "seed": config["seed"],
        "optimizer_steps": config["max_steps"], "resumed_from_step": start_step,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
        "elapsed_s": history[-1]["elapsed_s"], "data_audit": audit, "binding": binding,
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "training_is_quality_evidence": False,
    }
    if memory_backend is not None and hasattr(memory_backend, "max_memory_allocated"):
        summary["peak_memory_allocated_bytes"] = int(memory_backend.max_memory_allocated())
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_manifest.json", {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and "checkpoints" not in p.relative_to(output).parts and p.name != "artifact_manifest.json"})
    del optimizer, model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return summary
