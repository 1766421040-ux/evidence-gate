"""Published JSON metrics must audit exactly on Python 3.10 and 3.12+."""
import json

from evidence_gate.evaluation import ACTIONS, score, summarize


def test_macro_f1_matches_released_float_across_interpreters():
    # A nontrivial confusion matrix exposes the Python 3.12 sum(float) change.
    # Expected value is the released JSON number, not rounded for comparison.
    matrix = ((10, 37, 0, 33), (4, 96, 0, 0), (0, 0, 20, 0), (2, 0, 0, 78))
    cases = []

    def output(action):
        return {"action": action, "value": "30 days" if action == "answer" else "",
                "citations": ["official"] if action == "answer" else []}

    for expected, counts in zip(ACTIONS, matrix):
        for predicted, count in zip(ACTIONS, counts):
            for _ in range(count):
                row = {"id": str(len(cases)), "group_id": "portability-fixture",
                       "variant": "fixture", "observation": {}, "target": output(expected)}
                cases.append(score(row, json.dumps(output(predicted)), latency_ms=0.25))
    summary = summarize(cases)
    assert summary["action_macro_f1"] == 0.7122803986997132
    assert summary["success_rate"] == 204 / 280
    assert summary["mean_latency_ms"] == 0.25
