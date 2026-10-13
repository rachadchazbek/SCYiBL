"""Cohere adapter (official SDK, API v2) for :class:`scyibl.llm.base.LLMClient`."""

from __future__ import annotations

import time
from collections.abc import Sequence
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


def _get(obj: Any, *path: str, default: Any = None) -> Any:
    """Read nested attributes or keys without failing on SDK shape differences."""
    for name in path:
        if obj is None:
            return default
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return default if obj is None else obj


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class CohereClient:
    """Thin async wrapper. ``sdk_client`` can be injected for tests."""

    provider = "cohere"

    def __init__(
        self, api_key: str, *, base_url: str | None = None, sdk_client: Any | None = None
    ) -> None:
        if sdk_client is None:
            import cohere  # imported lazily so the package imports without the SDK

            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            sdk_client = cohere.AsyncClientV2(**kwargs)
        self._sdk = sdk_client

    def _check(self, model: ModelRef) -> str:
        if model.provider != self.provider:
            raise LLMError(f"CohereClient cannot serve model {model}")
        return model.name

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
        name = self._check(model)
        kwargs: dict[str, Any] = {
            "model": name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object", "schema": schema},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "request_options": {"timeout_in_seconds": max(1, int(timeout_s))},
        }
        if thinking_budget is not None:
            kwargs["thinking"] = {"type": "enabled", "token_budget": thinking_budget}

        start = time.perf_counter()
        try:
            response = await self._sdk.chat(**kwargs)
        except Exception as exc:  # SDK raises many types (HTTP, timeout, rate limit)
            raise LLMError(f"cohere chat ({name}) failed: {type(exc).__name__}: {exc}") from exc
        latency = _ms(start)

        content = _get(response, "message", "content", default=[]) or []
        text = "".join(
            _get(part, "text", default="") for part in content if _get(part, "type") == "text"
        )
        if not text:
            reason = _get(response, "finish_reason", default="unknown")
            raise LLMError(f"cohere chat ({name}) returned no text (finish_reason={reason})")

        usage = Usage(
            input_tokens=_int(
                _get(response, "usage", "tokens", "input_tokens")
                or _get(response, "usage", "billed_units", "input_tokens")
            ),
            output_tokens=_int(
                _get(response, "usage", "tokens", "output_tokens")
                or _get(response, "usage", "billed_units", "output_tokens")
            ),
            calls=1,
        )
        return ChatResult(
            data=parse_json_object(text),
            text=text,
            usage=usage,
            latency_ms=latency,
            model=str(model),
        )

    async def embed(
        self, *, model: ModelRef, texts: Sequence[str], input_type: EmbedInputType
    ) -> EmbedResult:
        name = self._check(model)
        start = time.perf_counter()
        try:
            response = await self._sdk.embed(
                model=name, texts=list(texts), input_type=input_type, embedding_types=["float"]
            )
        except Exception as exc:
            raise LLMError(f"cohere embed ({name}) failed: {type(exc).__name__}: {exc}") from exc
        latency = _ms(start)

        embeddings = _get(response, "embeddings")
        vectors = _get(embeddings, "float_") or _get(embeddings, "float")
        if not vectors or len(vectors) != len(texts):
            raise LLMError(f"cohere embed ({name}) returned {len(vectors or [])} vectors")
        usage = Usage(
            embed_tokens=_int(_get(response, "meta", "billed_units", "input_tokens")), calls=1
        )
        return EmbedResult(
            vectors=[list(map(float, v)) for v in vectors],
            usage=usage,
            latency_ms=latency,
            model=str(model),
        )

    async def rerank(
        self, *, model: ModelRef, query: str, documents: Sequence[str], top_n: int
    ) -> RerankResult:
        name = self._check(model)
        start = time.perf_counter()
        try:
            response = await self._sdk.rerank(
                model=name, query=query, documents=list(documents), top_n=top_n
            )
        except Exception as exc:
            raise LLMError(f"cohere rerank ({name}) failed: {type(exc).__name__}: {exc}") from exc
        latency = _ms(start)

        ranking = [
            (_int(_get(r, "index")), float(_get(r, "relevance_score", default=0.0)))
            for r in _get(response, "results", default=[]) or []
        ]
        usage = Usage(
            search_units=_int(_get(response, "meta", "billed_units", "search_units")), calls=1
        )
        return RerankResult(ranking=ranking, usage=usage, latency_ms=latency, model=str(model))
