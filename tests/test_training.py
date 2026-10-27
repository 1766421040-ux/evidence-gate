"""Contract checks plus a real tiny offline LoRA/resume integration test."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from evidence_gate.training import _read_checkpoint, normalize_config, prepare_examples, target_text, train


class CharacterTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        self.calls.append(deepcopy(messages))
        ids = []
        for message in messages:
            ids.extend([4 if message["role"] == "assistant" else 3])
            ids.extend(ord(character) + 10 for character in message["content"])
            ids.append(self.eos_token_id)
        if add_generation_prompt:
            ids.append(4)
        return ids


def row(row_id="sample-0", split="train"):
    return {
        "id": row_id, "group_id": "secret-group", "split": split,
        "variant": "SECRET_VARIANT", "gold": {"secret": "SECRET_GOLD"},
        "observation": {"secret": "SECRET_OBSERVATION"},
        "messages": [{"role": "system", "content": "Use evidence."}, {"role": "user", "content": "What is the limit?"}],
        "target": {"action": "answer", "value": "30", "citations": ["doc1"]},
    }


def test_completion_mask_eos_and_metadata_isolation():
    tokenizer = CharacterTokenizer()
    examples, audit = prepare_examples([row()], tokenizer, 256)
    example = examples[0]
    first_supervised = next(i for i, label in enumerate(example["labels"]) if label != -100)
    prompt = tokenizer.apply_chat_template(row()["messages"], add_generation_prompt=True)
    assert first_supervised == len(prompt)
    assert example["labels"][-1] == tokenizer.eos_token_id
    assert example["labels"][first_supervised:] == example["input_ids"][first_supervised:]
    assert "SECRET_" not in json.dumps(tokenizer.calls)
    assert "secret-group" not in json.dumps(tokenizer.calls)
    assert audit["eos_supervised_examples"] == 1
    assert audit["truncated_examples"] == 0


@pytest.mark.parametrize("split", ["eval", "test", "validation", None])
def test_no_evaluation_rows_in_training(split):
    with pytest.raises(ValueError, match="only split=train"):
        prepare_examples([row(split=split)], CharacterTokenizer(), 256)


def test_rejects_truncation_and_duplicate_ids():
    with pytest.raises(ValueError, match="silent truncation"):
        prepare_examples([row()], CharacterTokenizer(), 10)
    with pytest.raises(ValueError, match="duplicate id"):
        prepare_examples([row(), row()], CharacterTokenizer(), 256)


def test_rejects_template_mismatch_and_unsupervised_eos():
    class Mismatch(CharacterTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return [99] + result if add_generation_prompt else result
    class MissingEos(CharacterTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return result[:-1] if not add_generation_prompt else result
    with pytest.raises(ValueError, match="prefix mismatch"):
        prepare_examples([row()], Mismatch(), 256)
    with pytest.raises(ValueError, match="omits EOS"):
        prepare_examples([row()], MissingEos(), 256)


def test_config_and_target_fail_closed():
    with pytest.raises(ValueError, match="Unknown training options"):
        normalize_config({"max_stpes": 3})
    with pytest.raises(ValueError, match="positive integer"):
        normalize_config({"max_steps": 0})
    with pytest.raises(ValueError, match="exactly"):
        target_text({"action": "answer", "value": "x", "citations": [], "gold": "leak"})
    with pytest.raises(ValueError, match="absolute local path"):
        normalize_config({"model_name": "/private/local-model"})
    malformed = row()
    malformed["messages"] = ["not a message"]
    with pytest.raises(ValueError, match="messages"):
        prepare_examples([malformed], CharacterTokenizer(), 256)


@pytest.fixture
def tiny_snapshot(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {token: i for i, token in enumerate([
        "<pad>", "<bos>", "<eos>", "<unk>", "<system>", "<user>", "<assistant>",
        "Use", "evidence", ".", "What", "is", "the", "limit", "?", "30", "doc1",
        "action", "answer", "search", "clarify", "abstain", "value", "citations",
        "{", "}", '"', ":", ",", "[", "]", "31",
    ])}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    snapshot = tmp_path / "tiny-base"
    tokenizer.save_pretrained(snapshot)
    torch.manual_seed(101)
    config = transformers.Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    transformers.Qwen2ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    return snapshot


def test_real_lora_resume_matches_uninterrupted_and_guards_bindings(tmp_path, tiny_snapshot):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from evidence_gate.data import generate_group
    data = tmp_path / "train.jsonl"
    rows = generate_group(0, "train", 42)[:5]
    data.write_text("".join(json.dumps(example) + "\n" for example in rows))
    config = {
        "model_name": "offline-tiny-qwen-test", "max_steps": 3,
        "batch_size": 2, "gradient_accumulation_steps": 2,
        "max_length": 1024, "save_steps": 1, "warmup_steps": 0,
        "lora_dropout": 0.2, "num_threads": 1,
    }
    output = tmp_path / "run"
    summary = train(data, tiny_snapshot, output, config)
    assert summary["optimizer_steps"] == 3
    assert 0 < summary["trainable_parameters"] < summary["total_parameters"]
    assert summary["data_audit"]["eos_supervised_examples"] == 5
    weights = load_file(str(output / "adapter" / "adapter_model.safetensors"))
    assert any(torch.count_nonzero(value) > 0 for key, value in weights.items() if "lora_B" in key)
    losses = [x["loss"] for x in json.loads((output / "metrics.json").read_text())]
    checkpoint = output / "checkpoints" / "step-000001"
    from evidence_gate.hf_policy import HFPolicy
    checkpoint_policy = HFPolicy(tiny_snapshot, checkpoint, max_new_tokens=3)
    assert checkpoint_policy.provenance["checkpoint_step"] == 1
    assert checkpoint_policy.provenance["adapter_training_binding_verified"] is True
    assert isinstance(checkpoint_policy.predict(rows[0]["messages"])["raw"], str)
    checkpoint_policy.close()
    resumed = train(data, tiny_snapshot, output, config, resume=checkpoint)
    assert resumed["resumed_from_step"] == 1
    restored = load_file(str(output / "adapter" / "adapter_model.safetensors"))
    for key in weights:
        torch.testing.assert_close(weights[key], restored[key], rtol=0, atol=0)
    assert losses == [x["loss"] for x in json.loads((output / "metrics.json").read_text())]
    assert not list(output.rglob("*.pt")) and not list(output.rglob("*.bin"))
    adapter_config = json.loads((output / "adapter" / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == "offline-tiny-qwen-test"
    for name in ("run_manifest.json", "summary.json", "artifact_manifest.json"):
        assert str(tmp_path) not in (output / name).read_text()
    with pytest.raises(ValueError, match="already contains"):
        train(data, tiny_snapshot, output, config)
    with pytest.raises(ValueError, match="data, config, or model"):
        train(data, tiny_snapshot, output, {**config, "learning_rate": 1e-3}, resume=checkpoint)
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="data, config, or model"):
        train(data, tiny_snapshot, output, config, resume=checkpoint)
    with pytest.raises(ValueError, match="own output_dir"):
        _read_checkpoint(checkpoint, tmp_path / "different-run", summary["binding"])
    tensor_file = checkpoint / "training_state.safetensors"
    tensor_file.write_bytes(tensor_file.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _read_checkpoint(checkpoint, output, summary["binding"])


def test_train_rejects_label_injection_before_loading_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from evidence_gate.data import generate_group
    example = generate_group(0, "train", 42)[0]
    example["messages"][0]["content"] += "\nSECRET_GOLD=" + json.dumps(example["target"])
    data = tmp_path / "injected.jsonl"
    data.write_text(json.dumps(example) + "\n")
    output = tmp_path / "rejected-run"
    with pytest.raises(ValueError, match="Observation/message boundary violation"):
        train(data, tmp_path / "nonexistent-model", output, {"max_steps": 1})
    assert not output.exists()


def test_free_generation_batch_preserves_raw_and_order(tiny_snapshot):
    from evidence_gate.hf_policy import HFPolicy
    policy = HFPolicy(tiny_snapshot, max_new_tokens=3)
    prompts = [row()["messages"], [{"role": "user", "content": "What?"}]]
    outputs = policy.predict_batch(prompts)
    singles = [policy.predict(prompt) for prompt in prompts]
    assert [x["raw"] for x in outputs] == [x["raw"] for x in singles]
    assert outputs[0]["input_tokens"] > outputs[1]["input_tokens"]
    assert all(x["batch_size"] == 2 and x["latency_ms"] >= 0 for x in outputs)
    assert policy.predict_batch([]) == []
    policy.close()
