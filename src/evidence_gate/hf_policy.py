"""Greedy free-generation policy. No candidate scoring or decision repair."""
from __future__ import annotations

from pathlib import Path
import gc
import json
import time


def _check_adapter_metadata(adapter_path: Path, base_sha: str, adapter_sha: str) -> dict:
    """Check the originating run when available, without requiring it for imports.

    Weights alone do not identify a LoRA policy: alpha, target modules and other
    adapter configuration also affect its behavior.
    """
    from .training import sha256_file
    config_sha = sha256_file(adapter_path / "adapter_config.json")
    base_verified = False
    weight_verified = False
    checkpoint_provenance = {}
    checkpoint_path = adapter_path / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("format") != "evidence-gate-checkpoint-v1":
            raise ValueError("Unsupported adapter checkpoint format")
        binding = checkpoint.get("binding", {})
        if binding.get("model_sha256") != base_sha:
            raise ValueError("Checkpoint training base snapshot does not match inference base snapshot")
        for filename, actual in (("adapter_model.safetensors", adapter_sha), ("adapter_config.json", config_sha)):
            if checkpoint.get("files", {}).get(filename) != actual:
                raise ValueError(f"Checkpoint adapter checksum mismatch: {filename}")
        step = checkpoint.get("step")
        if type(step) is not int or step < 1 or not binding.get("data_sha256") or not binding.get("config_sha256"):
            raise ValueError("Checkpoint is missing its training step or binding")
        checkpoint_provenance = {
            "checkpoint_step": step,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_data_sha256": binding["data_sha256"],
            "training_config_sha256": binding["config_sha256"],
        }
        base_verified = weight_verified = True
    for filename in ("run_manifest.json", "summary.json"):
        path = adapter_path.parent / filename
        if not path.exists():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        expected_base = metadata.get("binding", {}).get("model_sha256")
        if expected_base:
            if expected_base != base_sha:
                raise ValueError("Adapter training base snapshot does not match inference base snapshot")
            base_verified = True
        expected_adapter = metadata.get("adapter_sha256")
        if expected_adapter:
            if expected_adapter != adapter_sha:
                raise ValueError("Adapter weights do not match the originating training summary")
            weight_verified = True
    artifact_path = adapter_path.parent / "artifact_manifest.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        expected_config = artifact.get(f"{adapter_path.name}/adapter_config.json")
        if expected_config and expected_config != config_sha:
            raise ValueError("Adapter config checksum does not match the training artifact manifest")
    release_path = adapter_path.parent / "manifest.json"
    if release_path.exists():
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if release.get("format") == "evidence-gate-release-manifest-v1":
            root = release_path.parent.parent.resolve()
            files = release["files"]
            for filename, actual in (("adapter_config.json", config_sha), ("adapter_model.safetensors", adapter_sha)):
                relative = str((adapter_path / filename).resolve().relative_to(root))
                if files.get(relative) != actual:
                    raise ValueError(f"Released adapter checksum mismatch: {filename}")
            # Published adapters have no adjacent run directory. Resolve their
            # training evidence through the existing release manifest, matching
            # by weight digest instead of relying on a folder naming convention.
            for relative, expected_hash in files.items():
                if not (relative.startswith("experiments/training/") and relative.endswith("/summary.json")):
                    continue
                summary_path = (root / relative).resolve()
                if not summary_path.is_relative_to(root):
                    raise ValueError("Release manifest path escapes the repository")
                metadata = json.loads(summary_path.read_text(encoding="utf-8"))
                if metadata.get("adapter_sha256") != adapter_sha:
                    continue
                if sha256_file(summary_path) != expected_hash:
                    raise ValueError("Released training summary checksum mismatch")
                if metadata.get("binding", {}).get("model_sha256") != base_sha:
                    raise ValueError("Released adapter training base snapshot does not match inference base snapshot")
                run_artifact = summary_path.with_name("artifact_manifest.json")
                artifact_relative = str(run_artifact.relative_to(root))
                if files.get(artifact_relative) != sha256_file(run_artifact):
                    raise ValueError("Released training artifact manifest checksum mismatch")
                run_hashes = json.loads(run_artifact.read_text(encoding="utf-8"))
                if run_hashes.get("adapter/adapter_config.json") != config_sha:
                    raise ValueError("Released adapter config differs from its training artifact")
                base_verified = weight_verified = True
    return {**checkpoint_provenance, "adapter_config_sha256": config_sha,
            "adapter_training_binding_verified": base_verified and weight_verified}


class HFPolicy:
    def __init__(self, model_path: str | Path, adapter_path: str | Path | None = None, device: str = "cpu", max_new_tokens: int = 72, dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; no silent CPU fallback")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no silent CPU fallback")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype")
        self.torch, self.device, self.max_new_tokens = torch, device, max_new_tokens
        source = str(model_path)
        local = Path(model_path).is_dir()
        if not local:
            raise ValueError("Use a downloaded local snapshot; unpinned Hub inference is disabled")
        from .training import _model_manifest, sha256_file
        snapshot = _model_manifest(Path(model_path))
        known_qwen = snapshot["files"].get("model.safetensors") == "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
        self.provenance = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct" if known_qwen else "local model snapshot",
            "base_snapshot_sha256": snapshot["sha256"],
            "base_files": snapshot["files"],
            "adapter_sha256": sha256_file(Path(adapter_path) / "adapter_model.safetensors") if adapter_path else None,
            "adapter_config_sha256": None,
        }
        if adapter_path is not None:
            self.provenance.update(_check_adapter_metadata(Path(adapter_path), snapshot["sha256"], self.provenance["adapter_sha256"]))
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(source, local_files_only=local, trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa")
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), local_files_only=True)
        self.model.to(device)
        self.model.eval()
        self.provenance["base_dtype"] = str(next(self.model.parameters()).dtype)
        self.provenance["adapter_dtypes"] = sorted({str(parameter.dtype) for name, parameter in self.model.named_parameters() if "lora_" in name})

    def predict(self, messages: list[dict]) -> dict:
        return self.predict_batch([messages])[0]

    def predict_batch(self, list_of_messages: list[list[dict]]) -> list[dict]:
        if not list_of_messages:
            return []
        for messages in list_of_messages:
            if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user" or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
                raise ValueError("Each prompt must contain text messages ending with user")
        tokens = [self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) for messages in list_of_messages]
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum and max(map(len, tokens)) + self.max_new_tokens > maximum:
            raise ValueError("Prompt plus generation budget exceeds model context; no silent truncation")
        batch = self.tokenizer.pad({"input_ids": tokens, "attention_mask": [[1] * len(ids) for ids in tokens]}, padding=True, return_tensors="pt").to(self.device)
        self._synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        self._synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        width = batch["input_ids"].shape[1]
        results = []
        for index, generated in enumerate(output[:, width:]):
            ids = generated.tolist()
            eos = self.tokenizer.eos_token_id
            if eos in ids:
                ids = ids[:ids.index(eos) + 1]
            results.append({
                "raw": self.tokenizer.decode(ids, skip_special_tokens=True),
                # Batch wall time amortized over examples; not individual
                # request latency. batch_latency_ms retains the original time.
                "latency_ms": elapsed / len(tokens), "batch_latency_ms": elapsed,
                "batch_size": len(tokens), "input_tokens": len(tokens[index]),
                "output_tokens": len(ids), "hit_generation_limit": len(ids) == self.max_new_tokens and eos not in ids,
            })
        del generated, output, batch
        if self.device == "mps":
            self.torch.mps.empty_cache()
        return results

    def _synchronize(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device == "cuda":
            self.torch.cuda.synchronize()

    def close(self):
        """Release a model between paired base/adapter evaluations."""
        self.model = None
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()
