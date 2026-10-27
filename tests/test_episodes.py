import importlib.util
import json
from copy import deepcopy
from pathlib import Path

from evidence_gate.data import canonical, generate
from evidence_gate.evaluation import contract_rule

spec = importlib.util.spec_from_file_location("run_diagnostics", Path(__file__).resolve().parents[1] / "scripts/run_diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class PublicRulePolicy:
    def predict(self, messages):
        return {"raw": canonical(contract_rule(json.loads(messages[-1]["content"])))}


def rows():
    return {r["variant"]: r for r in generate(seed=42, train_groups=1, dev_groups=1, test_groups=1) if r["split"] == "test"}


def test_episode_searches_and_validates_newly_observed_evidence():
    row = rows()["missing_search"]
    result = diagnostics.score_episode(row, PublicRulePolicy())
    assert result["success"] and result["searches"] == 1
    assert [e["decision"]["action"] for e in result["trace"]] == ["search", "answer"]
    assert result["trace"][-1]["observation"]["budget"] == 0


def test_lucky_hidden_catalogue_answer_is_not_success():
    row = rows()["missing_search"]
    hidden = {**row["observation"], "documents": row["gold"]["catalogue"]}
    class Guess:
        def predict(self, messages):
            return {"raw": canonical(contract_rule(hidden))}
    result = diagnostics.score_episode(row, Guess())
    assert result["status"] == "completed" and result["decision"]["action"] == "answer"
    assert not result["success"]  # The answer was never visible to this policy.


def test_zero_budget_refusal_is_success_but_invalid_output_is_not():
    row = rows()["missing_stop"]
    assert diagnostics.score_episode(row, PublicRulePolicy())["success"]
    class Invalid:
        def predict(self, messages):
            return {"raw": "I think it is thirty days."}
    result = diagnostics.score_episode(row, Invalid())
    assert result["status"] == "invalid_output" and not result["success"]


def test_correct_final_answer_does_not_erase_an_unnecessary_search():
    row = deepcopy(rows()["supported"])
    row["observation"]["budget"] = 1

    class SearchThenAnswer(PublicRulePolicy):
        calls = 0

        def predict(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"raw": canonical({"action": "search", "value": "", "citations": []})}
            return super().predict(messages)

    result = diagnostics.score_episode(row, SearchThenAnswer())
    assert result["status"] == "completed"
    assert result["decision"]["action"] == "answer"
    assert [event["correct"] for event in result["trace"]] == [False, True]
    assert not result["success"]


def test_search_without_budget_cannot_count_as_episode_success():
    class AlwaysSearch:
        def predict(self, messages):
            return {"raw": canonical({"action": "search", "value": "", "citations": []})}

    result = diagnostics.score_episode(rows()["missing_stop"], AlwaysSearch())
    assert result["status"] == "budget_violation"
    assert result["searches"] == 0
    assert result["trace"][0]["expected"]["action"] == "abstain"
    assert not result["success"]


def test_policy_only_observes_catalogue_results_after_search():
    row = rows()["missing_search"]
    before = deepcopy(row)

    class Observe(PublicRulePolicy):
        def __init__(self):
            self.observations = []

        def predict(self, messages):
            observation = json.loads(messages[-1]["content"])
            assert set(observation) == {"question", "as_of", "budget", "documents"}
            self.observations.append(observation)
            return super().predict(messages)

    policy = Observe()
    result = diagnostics.score_episode(row, policy)
    assert result["success"]
    assert policy.observations[0] == row["observation"]
    assert policy.observations[1]["documents"] != policy.observations[0]["documents"]
    assert {doc["id"] for doc in policy.observations[1]["documents"]} == set(result["trace"][0]["retrieved_ids"])
    assert policy.observations[1]["budget"] == 0
    assert row == before  # Episode execution and grading do not rewrite gold or input.
