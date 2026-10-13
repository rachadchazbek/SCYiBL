"""Deterministic in-memory :class:`LLMClient` for tests and ``--dry-run``. No network."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from scyibl.config.models import ModelRef
from scyibl.llm.base import (
    ChatResult,
    EmbedInputType,
    EmbedResult,
    LLMError,
    RerankResult,
    Usage,
    parse_json_object,
)


@dataclass(frozen=True)
class ChatCall:
    model: ModelRef
    system: str
    user: str
    schema: dict[str, Any]


ChatResponder = Callable[[ChatCall], dict[str, Any] | str]


def _words(text: str) -> set[str]:
    return {w.strip(".,!?;:").lower() for w in text.split() if len(w) > 2}


class FakeLLMClient:
    """Returns whatever ``responder`` produces for each chat call and records every call.

    ``responder`` may return a dict (serialised to JSON) or a raw string (to test invalid
    output). Raising :class:`LLMError` inside it simulates a provider failure.
    """

    def __init__(self, responder: ChatResponder, *, dim: int = 16) -> None:
        self._responder = responder
        self._dim = dim
        self.chat_calls: list[ChatCall] = []

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
        call = ChatCall(model=model, system=system, user=user, schema=schema)
        self.chat_calls.append(call)
        reply = self._responder(call)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        usage = Usage(
            input_tokens=len((system + user).split()), output_tokens=len(text.split()), calls=1
        )
        return ChatResult(
            data=parse_json_object(text), text=text, usage=usage, latency_ms=1.0, model=str(model)
        )

    async def embed(
        self, *, model: ModelRef, texts: Sequence[str], input_type: EmbedInputType
    ) -> EmbedResult:
        vectors = []
        for text in texts:
            vec = [0.0] * self._dim
            for word in _words(text):
                bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self._dim
                vec[bucket] += 1.0
            vectors.append(vec)
        usage = Usage(embed_tokens=sum(len(t.split()) for t in texts), calls=1)
        return EmbedResult(vectors=vectors, usage=usage, latency_ms=1.0, model=str(model))

    async def rerank(
        self, *, model: ModelRef, query: str, documents: Sequence[str], top_n: int
    ) -> RerankResult:
        if top_n < 1:
            raise LLMError("top_n must be >= 1")
        q = _words(query)
        scored = [(i, len(q & _words(d)) / (len(q) or 1)) for i, d in enumerate(documents)]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return RerankResult(
            ranking=scored[:top_n],
            usage=Usage(search_units=1, calls=1),
            latency_ms=1.0,
            model=str(model),
        )
