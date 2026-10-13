"""Build the LLM client for a runtime config: one adapter per provider in use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scyibl.config.models import ModelRef
from scyibl.config.settings import ConfigError, RuntimeConfig
from scyibl.llm.base import (
    ChatResult,
    EmbedInputType,
    EmbedResult,
    LLMClient,
    LLMError,
    RerankResult,
)

SUPPORTED_PROVIDERS = ("cohere",)


class ProviderRouter:
    """Dispatches each call to the adapter of the model's provider."""

    def __init__(self, clients: Mapping[str, LLMClient]) -> None:
        self._clients = dict(clients)

    def _for(self, model: ModelRef) -> LLMClient:
        try:
            return self._clients[model.provider]
        except KeyError:
            raise LLMError(f"no client configured for provider '{model.provider}'") from None

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
        return await self._for(model).chat_json(
            model=model,
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            thinking_budget=thinking_budget,
        )

    async def embed(
        self, *, model: ModelRef, texts: Sequence[str], input_type: EmbedInputType
    ) -> EmbedResult:
        return await self._for(model).embed(model=model, texts=texts, input_type=input_type)

    async def rerank(
        self, *, model: ModelRef, query: str, documents: Sequence[str], top_n: int
    ) -> RerankResult:
        return await self._for(model).rerank(
            model=model, query=query, documents=documents, top_n=top_n
        )


def build_llm_client(cfg: RuntimeConfig) -> ProviderRouter:
    clients: dict[str, LLMClient] = {}
    unsupported = sorted(set(cfg.app.providers_in_use()) - set(SUPPORTED_PROVIDERS))
    if unsupported:
        raise ConfigError(
            [f"no adapter for provider(s) {unsupported}; supported: {list(SUPPORTED_PROVIDERS)}"]
        )
    for provider in cfg.app.providers_in_use():
        key = cfg.provider_api_key(provider).get_secret_value()
        if provider == "cohere":
            from scyibl.llm.cohere_client import CohereClient

            clients[provider] = CohereClient(key, base_url=cfg.app.providers[provider].base_url)
    return ProviderRouter(clients)
