import copy
import importlib.util
import json
from pathlib import Path
import re

import pytest

from evidence_gate.data import VARIANTS, audit, generate, messages_for, read_jsonl, write_jsonl


spec = importlib.util.spec_from_file_location("build_diagnostics", Path(__file__).resolve().parents[1] / "scripts" / "build_diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


@pytest.fixture
def source_rows():
    rows = [row for row in generate(seed=42, train_groups=3, dev_groups=1, test_groups=5) if row["split"] == "test"]
    return list(reversed(rows))  # Selection must not depend on file order.


def test_complete_groups_semantics_and_observable_fields_preserved(source_rows):
    original = copy.deepcopy(source_rows)
    selected = diagnostics.select_groups(source_rows)
    paraphrases, redactions = diagnostics.transform_rows(selected)
    assert source_rows == original
    assert len(selected) == len(paraphrases) == len(redactions) == 56
    expected_groups = sorted({row["group_id"] for row in source_rows})[:4]
    assert sorted({row["group_id"] for row in selected}) == expected_groups
    assert {row["gold"]["topic"] for row in selected} == diagnostics.TOPICS
    for group in expected_groups:
        assert {row["variant"] for row in selected if row["group_id"] == group} == set(VARIANTS)
    for source, paraphrase in zip(selected, paraphrases):
        assert paraphrase["split"] == source["split"] == "test"
        assert paraphrase["id"] == source["id"] and paraphrase["group_id"] == source["group_id"]
        assert paraphrase["target"] == source["target"]
        assert paraphrase["gold"] == source["gold"]
        before, after = source["observation"], paraphrase["observation"]
        assert before["as_of"] == after["as_of"] and before["budget"] == after["budget"]
        product, plan = source["gold"]["product"], source["gold"]["plan"]
        assert product in after["question"]
        assert after["question"] != before["question"]
        assert (plan in after["question"]) == (plan in before["question"])
        if source["variant"] == "ambiguous":
            assert all(plan not in after["question"] for plan in ("Starter", "Team", "Business"))
        for old, new in zip(before["documents"], after["documents"]):
            assert {k: v for k, v in old.items() if k != "text"} == {k: v for k, v in new.items() if k != "text"}
            assert re.findall(r"\d+", old["text"]) == re.findall(r"\d+", new["text"])
            for entity in (product, "Starter", "Team", "Business"):
                assert (entity in old["text"]) == (entity in new["text"])
            if old["source"] != "official" or "custom themes" in old["text"]:
                assert old == new
            else:
                assert new["text"] != old["text"]
        assert paraphrase["messages"] == messages_for(after)
        assert '"diagnostic"' not in paraphrase["messages"][-1]["content"]
    assert audit(paraphrases)["group_leaks"] == audit(redactions)["group_leaks"] == 0


def test_redaction_targets_follow_observable_plan_and_budget(source_rows):
    selected = diagnostics.select_groups(source_rows)
    _, redactions = diagnostics.transform_rows(selected)
    for source, redacted in zip(selected, redactions):
        assert redacted["observation"]["documents"] == []
        assert redacted["observation"]["question"] == source["observation"]["question"]
        assert redacted["split"] == "test"
        assert redacted["target"]["value"] == "" and redacted["target"]["citations"] == []
        plan_is_explicit = any(plan in source["observation"]["question"] for plan in ("Starter", "Team", "Business"))
        expected = "clarify" if not plan_is_explicit else ("search" if source["observation"]["budget"] else "abstain")
        assert redacted["target"]["action"] == expected
        assert redacted["messages"] == messages_for(redacted["observation"])
    assert not any(row["target"]["action"] == "answer" for row in redactions)


def test_build_preserves_main_and_refuses_overwrite(tmp_path, source_rows):
    source = tmp_path / "main" / "test.jsonl"
    write_jsonl(source, source_rows)
    before = source.read_bytes()
    output = tmp_path / "diagnostics"
    manifest = diagnostics.build(source, output)
    assert source.read_bytes() == before
    assert manifest["independent_of_main_test"] is False
    assert manifest["models_called"] == 0 and manifest["training_data_read"] is False
    assert manifest["rows_per_diagnostic"] == 56
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name in ("paraphrase.jsonl", "redacted.jsonl"):
        assert len(read_jsonl(output / name)) == 56
    with pytest.raises(FileExistsError, match="already exist"):
        diagnostics.build(source, output)


def test_incomplete_or_non_test_input_is_rejected(source_rows):
    first_group = sorted({row["group_id"] for row in source_rows})[0]
    incomplete = [row for row in source_rows if not (row["group_id"] == first_group and row["variant"] == "supported")]
    with pytest.raises(ValueError, match="Incomplete"):
        diagnostics.select_groups(incomplete)
    for row in source_rows:
        row["split"] = "train"
    with pytest.raises(ValueError, match="locked test"):
        diagnostics.select_groups(source_rows)


def test_unknown_wording_fails_instead_of_guessing():
    with pytest.raises(ValueError, match="semantic preservation"):
        diagnostics.paraphrase_question("An unseen and ambiguous question")
    with pytest.raises(ValueError, match="semantic review"):
        diagnostics.paraphrase_document({"source": "official", "text": "An unsupported policy says something else."})
