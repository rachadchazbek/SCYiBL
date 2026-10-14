"""Provider-agnostic LLM interface used by every agent.

Agents depend only on this module, never on a vendor SDK or on LangChain types, so they can be
tested with :class:`scyibl.llm.fake.FakeLLMClient` and a provider can be swapped by adding an
adapter (see ``docs/adr/0001-agent-framework.md``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from scyibl.config.models import ModelRef

EmbedInputType = Literal["search_document", "search_query", "classification", "clustering"]


class LLMError(Exception):
    """A provider call failed or returned something unusable."""


@dataclass(frozen=True)
class Usage:
    """Token and unit counts, summable across calls (``a + b``)."""

    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    search_units: int = 0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embed_tokens=self.embed_tokens + other.embed_tokens,
            search_units=self.search_units + other.search_units,
            calls=self.calls + other.calls,
        )


@dataclass(frozen=True)
class ChatResult:
    data: dict[str, Any]
    text: str
    usage: Usage
    latency_ms: float
    model: str


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]] = field(repr=False)
    usage: Usage
    latency_ms: float
    model: str


@dataclass(frozen=True)
class RerankResult:
    ranking: list[tuple[int, float]]
    """(index in the input documents, relevance score), best first."""
    usage: Usage
    latency_ms: float
    model: str


@runtime_checkable
class LLMClient(Protocol):
    async def chat_json(
        self,
        *,
        model: ModelRef,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 30.0,
        thinking_budget: int | None = None,
    ) -> ChatResult:
        """Chat completion constrained to ``schema``; returns the parsed JSON object."""
        ...

    async def embed(
        self, *, model: ModelRef, texts: Sequence[str], input_type: EmbedInputType
    ) -> EmbedResult: ...

    async def rerank(
        self, *, model: ModelRef, query: str, documents: Sequence[str], top_n: int
    ) -> RerankResult: ...


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply that must be a JSON object, tolerating Markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model did not return valid JSON: {exc.msg} at char {exc.pos}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"model returned JSON {type(data).__name__}, expected an object")
    return data
