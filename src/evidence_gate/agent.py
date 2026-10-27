"""Bounded, read-only retrieval loop with a public observation boundary."""
from __future__ import annotations

import copy
import re
import time

from .data import canonical, messages_for
from .evaluation import parse_prediction


def retrieve(question, catalogue, limit=4):
    """Deterministic lexical search; no vectors or learned reranker are claimed."""
    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored = []
    for document in catalogue:
        tokens = set(re.findall(r"[a-z0-9]+", document["text"].lower()))
        scored.append((len(terms & tokens), document["id"], document))
    return [copy.deepcopy(d) for score, _, d in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit] if score]


def run_episode(observation, catalogue, policy, *, kind="rule", max_steps=4):
    if not 1 <= max_steps <= 12:
        raise ValueError("max_steps must be in [1,12]")
    state = copy.deepcopy(observation)
    if not isinstance(state["budget"], int) or not 0 <= state["budget"] <= 5:
        raise ValueError("budget must be an integer in [0,5]")
    trace = []
    searches = 0
    for step in range(max_steps):
        public = copy.deepcopy(state)
        start = time.perf_counter()
        raw = policy.predict(messages_for(public))["raw"] if kind == "model" else canonical(policy(public))
        event = {"step": step, "observation": public, "raw": raw,
                 "latency_ms": (time.perf_counter() - start) * 1000}
        trace.append(event)
        try:
            parsed = parse_prediction(raw)
        except ValueError as exc:
            event["error"] = str(exc)
            return {"status": "invalid_output", "decision": None, "searches": searches, "trace": trace}
        event["decision"] = parsed
        if parsed["action"] != "search":
            return {"status": "completed", "decision": parsed, "searches": searches, "trace": trace}
        if state["budget"] == 0:
            return {"status": "budget_violation", "decision": parsed, "searches": searches, "trace": trace}
        searches += 1
        state["budget"] -= 1
        state["documents"] = retrieve(state["question"], catalogue)
        event["retrieved_ids"] = [d["id"] for d in state["documents"]]
    return {"status": "step_limit", "decision": None, "searches": searches, "trace": trace}
