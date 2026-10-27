"""Provenance and replay regression checks without model downloads or GPUs."""
import json

import pytest

from evidence_gate.cli import main
from evidence_gate.data import generate_group, write_jsonl
from evidence_gate.hf_policy import _check_adapter_metadata
from evidence_gate.training import sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def model_result(name, adapter=None):
    return {
        "name": name, "kind": "model", "dataset_sha256": "test-data",
        "model_provenance": {"model_name": "test-model", "base_snapshot_sha256": "base-sha", "adapter_sha256": adapter},
        "inference": {"decoding": "greedy free generation", "dtype": "bfloat16", "device": "mps", "max_new_tokens": 72, "batch_size": 4, "output_repair": False},
        "summary": {}, "cases": [],
    }


def test_replay_passes_dtype_and_records_binding(tmp_path, monkeypatch):
    from evidence_gate import hf_policy
    calls = {}
    class Policy:
        def __init__(self, model, adapter, device, dtype):
            calls.update(model=model, adapter=adapter, device=device, dtype=dtype)
            self.provenance = {"base_snapshot_sha256": "base", "adapter_sha256": "adapter"}
        def predict(self, messages):
            assert all(set(message) == {"role", "content"} for message in messages)
            return {"raw": '{"action":"abstain","value":"","citations":[]}'}
        def close(self):
            calls["closed"] = True
    monkeypatch.setattr(hf_policy, "HFPolicy", Policy)
    row = generate_group(5, "test", 42)[0]
    source, output = tmp_path / "test.jsonl", tmp_path / "replay.json"
    write_jsonl(source, [row])
    main(["replay", "--data", str(source), "--id", row["id"], "--model", "base", "--adapter", "adapter", "--device", "mps", "--dtype", "bfloat16", "--output", str(output)])
    assert calls == {"model": "base", "adapter": "adapter", "device": "mps", "dtype": "bfloat16", "closed": True}
    result = json.loads(output.read_text())
    assert result["inference"]["dtype"] == "bfloat16"
    assert result["model_provenance"]["adapter_sha256"] == "adapter"


def test_report_accepts_older_base_schema_and_rules(tmp_path):
    base = model_result("base")  # Older base evaluation lacks adapter_config_sha256.
    adapted = model_result("sft", "adapter-sha")
    rule = {"name": "contract", "kind": "rule", "dataset_sha256": "test-data", "summary": {}, "cases": []}
    inputs = [write_json(tmp_path / f"{i}.json", method) for i, method in enumerate([base, adapted, rule])]
    summary = write_json(tmp_path / "training" / "summary.json", {"adapter_sha256": "adapter-sha", "binding": {"model_sha256": "base-sha"}})
    output = tmp_path / "report.json"
    main(["report", *inputs, "--training", summary, "--output", str(output)])
    assert len(json.loads(output.read_text())["methods"]) == 3


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("batch_size", 8), ("max_new_tokens", 144)])
def test_report_rejects_uncontrolled_inference_comparison(tmp_path, field, value):
    base, adapted = model_result("base"), model_result("sft", "adapter")
    adapted["inference"][field] = value
    inputs = [write_json(tmp_path / f"{i}.json", item) for i, item in enumerate([base, adapted])]
    with pytest.raises(SystemExit) as error:
        main(["report", *inputs, "--output", str(tmp_path / "report.json")])
    assert error.value.code == 2


def test_report_rejects_training_base_mismatch_even_same_adapter_weights(tmp_path):
    method = write_json(tmp_path / "eval.json", model_result("sft", "adapter-sha"))
    summary = write_json(tmp_path / "summary.json", {"adapter_sha256": "adapter-sha", "binding": {"model_sha256": "different-base"}})
    with pytest.raises(SystemExit) as error:
        main(["report", method, "--training", summary, "--output", str(tmp_path / "report.json")])
    assert error.value.code == 2


def test_adapter_binding_checks_base_weights_and_configuration(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    config.write_text('{"lora_alpha": 16}')
    write_json(tmp_path / "run_manifest.json", {"binding": {"model_sha256": "base"}})
    write_json(tmp_path / "summary.json", {"binding": {"model_sha256": "base"}, "adapter_sha256": "weights"})
    write_json(tmp_path / "artifact_manifest.json", {"adapter/adapter_config.json": sha256_file(config)})
    metadata = _check_adapter_metadata(adapter, "base", "weights")
    assert metadata["adapter_training_binding_verified"] is True
    assert metadata["adapter_config_sha256"] == sha256_file(config)
    with pytest.raises(ValueError, match="base snapshot"):
        _check_adapter_metadata(adapter, "wrong-base", "weights")
    with pytest.raises(ValueError, match="weights"):
        _check_adapter_metadata(adapter, "base", "different-weights")
    config.write_text('{"lora_alpha": 32}')
    with pytest.raises(ValueError, match="config checksum"):
        _check_adapter_metadata(adapter, "base", "weights")


def test_external_adapter_config_still_fingerprinted_without_training_metadata(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"lora_alpha": 16}')
    metadata = _check_adapter_metadata(tmp_path, "base", "weights")
    assert metadata["adapter_training_binding_verified"] is False
    assert metadata["adapter_config_sha256"]


def test_checkpoint_adapter_checks_binding_and_inference_files(tmp_path):
    adapter = tmp_path / "checkpoints" / "step-000200"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"lora_alpha":16}')
    checkpoint = {
        "format": "evidence-gate-checkpoint-v1", "step": 200,
        "binding": {"model_sha256": "base", "data_sha256": "train", "config_sha256": "config"},
        "files": {"adapter_model.safetensors": "weights", "adapter_config.json": sha256_file(adapter / "adapter_config.json")},
    }
    write_json(adapter / "checkpoint.json", checkpoint)
    metadata = _check_adapter_metadata(adapter, "base", "weights")
    assert metadata["adapter_training_binding_verified"] is True
    assert metadata["checkpoint_step"] == 200
    assert metadata["checkpoint_sha256"] == sha256_file(adapter / "checkpoint.json")
    assert metadata["training_data_sha256"] == "train"
    assert metadata["training_config_sha256"] == "config"
    with pytest.raises(ValueError, match="base snapshot"):
        _check_adapter_metadata(adapter, "other-base", "weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _check_adapter_metadata(adapter, "base", "different-weights")
    (adapter / "adapter_config.json").write_text('{"lora_alpha":32}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        _check_adapter_metadata(adapter, "base", "weights")


def test_published_adapter_resolves_release_training_binding(tmp_path):
    adapter = tmp_path / "artifacts" / "named-adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"lora_alpha":16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"test-weight-digest-only")
    weights_sha = sha256_file(adapter / "adapter_model.safetensors")
    config_sha = sha256_file(adapter / "adapter_config.json")
    summary_path = tmp_path / "experiments" / "training" / "different-run-name" / "summary.json"
    artifact_path = summary_path.with_name("artifact_manifest.json")
    write_json(summary_path, {"adapter_sha256": weights_sha, "binding": {"model_sha256": "base"}})
    write_json(artifact_path, {"adapter/adapter_config.json": config_sha})
    files = {str(path.relative_to(tmp_path)): sha256_file(path) for path in [adapter / "adapter_config.json", adapter / "adapter_model.safetensors", summary_path, artifact_path]}
    write_json(tmp_path / "artifacts" / "manifest.json", {"format": "evidence-gate-release-manifest-v1", "files": files})
    assert _check_adapter_metadata(adapter, "base", weights_sha)["adapter_training_binding_verified"] is True
    with pytest.raises(ValueError, match="base snapshot"):
        _check_adapter_metadata(adapter, "different-base", weights_sha)
    (adapter / "adapter_config.json").write_text('{"lora_alpha":32}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        _check_adapter_metadata(adapter, "base", weights_sha)
