import importlib.util
from pathlib import Path

import pytest

from evidence_gate.data import generate_group, write_jsonl

spec = importlib.util.spec_from_file_location("diagnose_policy", Path(__file__).resolve().parents[1] / "scripts" / "diagnose_policy.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_stratification_is_deterministic_and_spreads_question_groups():
    rows = [row for group in range(12) for row in generate_group(group, "train", 42)]
    first = diagnostics.sample_rows(rows, per_action=8, seed=73)
    second = diagnostics.sample_rows(list(reversed(rows)), per_action=8, seed=73)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == len({row["id"] for row in first}) == 32
    for action in diagnostics.ACTIONS:
        members = [row for row in first if row["target"]["action"] == action]
        assert len(members) == len({row["group_id"] for row in members}) == 8


def test_cli_rejects_test_before_loading_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Model must not load for test input")
    monkeypatch.setattr(diagnostics, "HFPolicy", forbidden)
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_jsonl(train, generate_group(0, "train", 42))
    write_jsonl(dev, generate_group(1, "test", 42))
    with pytest.raises(SystemExit) as error:
        diagnostics.main(["--model", "missing", "--adapter", "missing", "--train", str(train), "--dev", str(dev), "--output", str(tmp_path / "result.json")])
    assert error.value.code == 2


@pytest.fixture
def tiny_policy(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocabulary = {word: index for index, word in enumerate([
        "<pad>", "<eos>", "<unk>", "system", "user", "assistant", "action", "value", "citations",
        "answer", "search", "clarify", "abstain", "30", "days", "doc", "1", "{", "}", '"', ":", ",", "[", "]",
    ])}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(4)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048, eos_token_id=1, pad_token_id=0))
    model.save_pretrained(tmp_path, safe_serialization=True)
    policy = diagnostics.HFPolicy(tmp_path, max_new_tokens=3)
    yield policy
    policy.close()


def test_actual_causal_loss_matches_manual_shift_and_fields(tiny_policy):
    row = generate_group(0, "train", 42)[0]
    encoded = diagnostics.encode_case(row, tiny_policy.tokenizer)
    assert "action" in encoded["fields"] and "value" in encoded["fields"] and "eos" in encoded["fields"]
    assert set(encoded["action_first_token_ids"]) == set(diagnostics.ACTIONS)
    result = diagnostics.teacher_forced(row, tiny_policy)
    assert result["loss_abs_error"] < 1e-5
    assert sum(result["action_first_token"]["candidate_probabilities"].values()) <= 1.000001
    assert all(token["nll"] >= 0 and 0 <= token["gold_probability"] <= 1 for token in result["tokens"])


def test_free_generation_receives_only_original_public_messages(tiny_policy):
    rows = diagnostics.sample_rows(generate_group(0, "train", 42), per_action=1)
    received = []
    predict = tiny_policy.predict_batch
    def capture(messages):
        received.extend(messages)
        return predict(messages)
    tiny_policy.predict_batch = capture
    result = diagnostics.diagnose(rows, tiny_policy, "tiny-diagnostic", batch_size=2)
    assert received == [row["messages"] for row in rows]
    assert len(result["free_generation"]["cases"]) == 4
    assert len(result["teacher_forced"]["cases"]) == 4
    assert result["teacher_forced"]["summary"]["overall"]["max_loss_abs_error"] < 1e-5
