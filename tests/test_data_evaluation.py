import copy
import json

import pytest

from evidence_gate.agent import run_episode
from evidence_gate.data import audit, canonical, generate, messages_for, save_dataset
from evidence_gate.evaluation import contract_rule, evaluate, parse_prediction, score


@pytest.fixture
def rows():
    return generate(train_groups=2, dev_groups=2, test_groups=2)


def test_generator_reproducible_and_split_groups(rows):
    assert rows == generate(train_groups=2, dev_groups=2, test_groups=2)
    report = audit(rows)
    assert report["groups"] == 6
    assert report["rows"] == 84
    assert all("gold" not in messages_for(r["observation"])[1]["content"] for r in rows)


def test_rule_agrees_with_independently_constructed_targets(rows):
    # Covers all evidence interventions, inclusive/exclusive boundaries and order.
    result = evaluate(rows, contract_rule, "contract")
    assert result["summary"]["success_rate"] == 1
    assert result["summary"]["all_variants_correct_rate"] == 1
    assert result["summary"]["action_macro_f1"] == 1


def test_group_leakage_is_rejected(rows):
    corrupt = copy.deepcopy(rows)
    corrupt[0]["split"] = "test"
    with pytest.raises(ValueError, match="Group leakage"):
        audit(corrupt)


def test_hidden_answer_cannot_enter_messages(rows):
    corrupt = copy.deepcopy(rows)
    corrupt[0]["messages"][1]["content"] += str(corrupt[0]["target"])
    with pytest.raises(ValueError, match="boundary"):
        audit(corrupt)


def test_nested_document_metadata_not_exposed(rows):
    obs = copy.deepcopy(rows[0]["observation"])
    obs["documents"][0]["expected_action"] = "answer"
    obs["documents"][0]["gold"] = "SECRET LABEL"
    assert "SECRET LABEL" not in canonical(messages_for(obs))
    assert "expected_action" not in canonical(messages_for(obs))


def test_zero_budget_still_allows_answer_and_clarify():
    data = generate()
    for split in ("train", "dev", "test"):
        actions = {r["target"]["action"] for r in data if r["split"] == split and r["observation"]["budget"] == 0}
        assert {"answer", "clarify", "abstain"} <= actions


def test_evidence_pairs_keep_budget_fixed(rows):
    from evidence_gate.evaluation import summarize
    cases = [score(r, canonical(r["target"])) for r in rows]
    # Fail only the unmatched expiry version; the matched pair must stay perfect.
    for c in cases:
        group = next(r for r in rows if r["group_id"] == c["group_id"] and r["variant"] == "supported")
        unmatched = "expired_search" if group["observation"]["budget"] == 0 else "expired_stop"
        if c["variant"] == unmatched:
            c["metrics"]["success"] = False
    assert summarize(cases)["pairs"]["expiry_boundary"]["accuracy"] == 1


@pytest.mark.parametrize("raw", [
    '```json\n{"action":"abstain","value":"","citations":[]}\n```',
    '{"action":"abstain","action":"search","value":"","citations":[]}',
    '{"action":"abstain","value":"30 days","citations":[]}',
    '{"action":"answer","value":"30 days","citations":[]}',
    '{"action":"answer","value":"30 days","citations":["x","x"]}',
    '{"action":"answer","value":30,"citations":["x"]}',
    '{"action":"answer","value":"30 days","citations":["x"],"extra":1}',
    '[]', 'null', '{"action":[],"value":"","citations":[]}',
])
def test_invalid_outputs_not_repaired(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_prediction(raw)


def test_wrong_citation_is_failure_even_correct_action(rows):
    row = rows[0]
    predicted = {**row["target"], "citations": ["invented"]}
    case = score(row, canonical(predicted))
    assert case["metrics"]["action_correct"]
    assert not case["metrics"]["success"]
    assert case["metrics"]["unsupported_answer"]


def test_refusal_has_no_fake_perfect_citation_score(rows):
    method = evaluate(rows, lambda _: {"action": "abstain", "value": "", "citations": []}, "abstain")
    assert method["summary"]["citation_exact_rate"] is None
    assert method["summary"]["unsupported_answer_rate"] is None
    assert method["summary"]["over_refusal_rate"] == 1


def test_counterfactual_values_and_invariance(rows):
    g = {r["variant"]: r for r in rows if r["group_id"] == "g0000"}
    assert g["supported"]["target"]["value"] != g["value_changed"]["target"]["value"]
    assert g["supported"]["target"] == g["distractor"]["target"] == g["reordered"]["target"]
    assert g["supported"]["observation"]["documents"] == list(reversed(g["reordered"]["observation"]["documents"]))


def test_real_retrieval_loop_and_budget(rows):
    row = next(r for r in rows if r["variant"] == "missing_search")
    result = run_episode(row["observation"], row["gold"]["catalogue"], contract_rule)
    assert result["status"] == "completed"
    assert result["searches"] == 1
    assert result["decision"]["action"] == "answer"
    assert len(result["trace"]) == 2
    assert row["observation"]["budget"] == 1  # Search variant starts with one retrieval.
    malicious = lambda _: {"action": "search", "value": "", "citations": []}
    result = run_episode(row["observation"], [], malicious)
    assert result["status"] == "budget_violation"
    assert result["searches"] == 1


def test_dataset_existing_output_preserved(tmp_path):
    save_dataset(tmp_path, train_groups=1, dev_groups=1, test_groups=1)
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path, train_groups=1, dev_groups=1, test_groups=1)


def test_model_policy_receives_only_messages(rows):
    class Spy:
        def predict_batch(self, prompts):
            assert all(set(m) == {"role", "content"} for p in prompts for m in p)
            return [{"raw": '{"action":"abstain","value":"","citations":[]}'} for _ in prompts]
    result = evaluate(rows[:2], Spy(), "spy", "model")
    assert result["kind"] == "model"
    assert result["summary"]["n"] == 2
