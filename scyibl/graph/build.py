"""LangGraph orchestration (ADR-001, issue #18). Spike version.

    START -> router -> classify --(should_escalate)--> adjudicate -> assemble -> END
                                 \\----------------(confident)----> assemble

Nodes are thin wrappers around plain agent functions. Retrieval (#6) and the parallel safety
branch (#8) are added in later issues.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from scyibl.agents.classifier import classify
from scyibl.config.models import AppConfig
from scyibl.escalation import confidence, should_escalate
from scyibl.graph.state import GraphState, TraceStep, estimate_cost, total_usage
from scyibl.llm.base import LLMClient, Usage
from scyibl.schema import ClassificationResult

_WHITESPACE = re.compile(r"\s+")


class InputError(ValueError):
    """The message cannot be processed (empty or too long)."""


def build_graph(app: AppConfig, client: LLMClient) -> Any:
    """Compile the graph. Returns a LangGraph ``CompiledStateGraph``."""

    async def router(state: GraphState) -> dict[str, Any]:
        start = time.perf_counter()
        text = _WHITESPACE.sub(" ", state.get("message", "")).strip()
        if not text:
            raise InputError("message is empty")
        if len(text) > app.graph.max_message_chars:
            raise InputError(f"message longer than {app.graph.max_message_chars} characters")
        step = TraceStep("router", None, _ms(start), Usage(), note="rule-based normalisation")
        return {"text": text, "trace": [step]}

    async def classify_node(state: GraphState) -> dict[str, Any]:
        agent = app.agents["classifier"]
        run = await classify(client, agent, state["text"])
        step = TraceStep(
            "classify", run.model, run.latency_ms, run.usage, f"attempts={run.attempts}"
        )
        trace = [*state.get("trace", []), step]
        decision = should_escalate(
            run.output,
            app.escalation,
            depth=0,
            cost_usd=estimate_cost(app, trace),
            adjudicator_enabled=app.agents["adjudicator"].enabled,
        )
        return {"first_pass": run.output, "decision": decision, "trace": [step]}

    def route_after_classify(state: GraphState) -> Literal["adjudicate", "assemble"]:
        return "adjudicate" if state["decision"].escalate else "assemble"

    async def adjudicate_node(state: GraphState) -> dict[str, Any]:
        agent = app.agents["adjudicator"]
        run = await classify(client, agent, state["text"], previous=state["first_pass"])
        step = TraceStep(
            "adjudicate", run.model, run.latency_ms, run.usage, f"attempts={run.attempts}"
        )
        return {"adjudication": run.output, "trace": [step]}

    async def assemble(state: GraphState) -> dict[str, Any]:
        trace = state.get("trace", [])
        final = state.get("adjudication") or state["first_pass"]
        usage = total_usage(trace)
        labels = ClassificationResult.label_view(final)
        result = ClassificationResult(
            request_id=state["request_id"],
            labels=labels,
            predicted=[label for label, score in labels.items() if score.score >= 0.5],
            confidence=confidence(final),
            escalated=state.get("adjudication") is not None,
            escalation_reasons=state["decision"].reasons(),
            models=[s.model for s in trace if s.model],
            latency_ms=round(sum(s.latency_ms for s in trace), 1),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=estimate_cost(app, trace),
        )
        return {"result": result}

    builder = StateGraph(GraphState)
    builder.add_node("router", router)
    builder.add_node("classify", classify_node)
    builder.add_node("adjudicate", adjudicate_node)
    builder.add_node("assemble", assemble)
    builder.add_edge(START, "router")
    builder.add_edge("router", "classify")
    builder.add_conditional_edges(
        "classify", route_after_classify, {"adjudicate": "adjudicate", "assemble": "assemble"}
    )
    builder.add_edge("adjudicate", "assemble")
    builder.add_edge("assemble", END)

    checkpointer = InMemorySaver() if app.graph.checkpointer == "memory" else None
    return builder.compile(checkpointer=checkpointer)


async def run_graph(
    graph: Any, app: AppConfig, message: str, *, request_id: str | None = None
) -> ClassificationResult:
    """Run one message through the compiled graph and return the public result."""
    rid = request_id or uuid.uuid4().hex
    config = {"recursion_limit": app.graph.recursion_limit, "configurable": {"thread_id": rid}}
    state = await graph.ainvoke({"request_id": rid, "message": message}, config=config)
    result: ClassificationResult = state["result"]
    return result


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
