"""Graph state shared by LangGraph nodes (issue #18)."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from scyibl.config.models import AppConfig, ModelRef
from scyibl.escalation import EscalationDecision
from scyibl.llm.base import Usage
from scyibl.schema import ClassificationResult, LLMClassification


@dataclass(frozen=True)
class TraceStep:
    node: str
    model: str | None
    latency_ms: float
    usage: Usage
    note: str = ""


class GraphState(TypedDict, total=False):
    # input
    request_id: str
    message: str
    # router
    text: str
    # classify / adjudicate
    first_pass: LLMClassification
    decision: EscalationDecision
    adjudication: LLMClassification
    # assemble
    result: ClassificationResult
    # accumulated across nodes (reducers allow future parallel branches, e.g. safety)
    trace: Annotated[list[TraceStep], operator.add]


def total_usage(trace: list[TraceStep]) -> Usage:
    total = Usage()
    for step in trace:
        total += step.usage
    return total


def estimate_cost(app: AppConfig, trace: list[TraceStep]) -> float | None:
    """USD cost of the chat calls in ``trace``; None when a model has no configured price."""
    cost = 0.0
    for step in trace:
        if step.model is None or step.usage.calls == 0:
            continue
        price = app.price_for(ModelRef.parse(step.model))
        if price is None:
            return None
        cost += step.usage.input_tokens * price.input_per_mtok / 1_000_000
        cost += step.usage.output_tokens * price.output_per_mtok / 1_000_000
    return round(cost, 6)
