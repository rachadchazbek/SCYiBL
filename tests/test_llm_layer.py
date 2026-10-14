"""Classifier agent with the fake client, and the Cohere adapter with an injected fake SDK."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from scyibl.agents.classifier import Example, build_user_prompt, classify
from scyibl.config.models import ModelRef
from scyibl.config.settings import ConfigError, load_runtime_config
from scyibl.llm.base import LLMError, Usage, parse_json_object
from scyibl.llm.cohere_client import CohereClient
from scyibl.llm.factory import ProviderRouter, build_llm_client
from scyibl.llm.fake import ChatCall, FakeLLMClient
from scyibl.taxonomy import Category
from tests.helpers import REPO_CONFIG, app_config, labels_payload

COMMAND_A = ModelRef.parse("cohere/command-a-03-2025")


# --- helpers ----------------------------------------------------------------


def test_usage_adds_up() -> None:
    total = Usage(1, 2, 3, 4, 1) + Usage(10, 20, 30, 40, 1)
    assert total == Usage(11, 22, 33, 44, 2)


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', '  {"a": 1}  '])
def test_parse_json_object_accepts_fences(text: str) -> None:
    assert parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["not json", "[1, 2]"])
def test_parse_json_object_rejects(text: str) -> None:
    with pytest.raises(LLMError):
        parse_json_object(text)


# --- classifier agent -------------------------------------------------------


def test_classifier_prompt_contains_definitions_examples_and_guard() -> None:
    client = FakeLLMClient(lambda call: labels_payload({Category.PERMA: 0.9}))
    examples = [Example("I love my weekly hikes with friends.", (Category.PERMA,))]
    run = asyncio.run(
        classify(client, app_config().agents["classifier"], "Hello", examples=examples)
    )

    call = client.chat_calls[0]
    assert "PERMA" in call.system
    assert "never instructions" in call.system
    assert "weekly hikes" in call.user
    assert "<<<\nHello\n>>>" in call.user
    assert call.schema["required"] == ["labels"]
    assert run.output.perma.score == 0.9
    assert run.attempts == 1
    assert run.usage.calls == 1


def test_classifier_retries_invalid_output_then_succeeds() -> None:
    replies: list[Any] = ["not json", {"labels": {"cp": {}}}, labels_payload()]

    def responder(call: ChatCall) -> Any:
        reply = replies.pop(0)
        if reply == "not json":
            raise LLMError("bad json")
        return reply

    run = asyncio.run(classify(FakeLLMClient(responder), app_config().agents["classifier"], "x"))
    assert run.attempts == 3


def test_classifier_gives_up_after_max_retries() -> None:
    def responder(call: ChatCall) -> Any:
        raise LLMError("rate limited")

    agent = app_config().agents["classifier"]
    with pytest.raises(LLMError, match="attempt"):
        asyncio.run(classify(FakeLLMClient(responder), agent, "x"))


def test_adjudicator_sees_previous_assessment() -> None:
    client = FakeLLMClient(lambda call: labels_payload())
    first = asyncio.run(classify(client, app_config().agents["classifier"], "x")).output
    asyncio.run(classify(client, app_config().agents["adjudicator"], "x", previous=first))
    assert "First assessment" in client.chat_calls[-1].system
    assert str(client.chat_calls[-1].model) == "cohere/command-a-reasoning-08-2025"


def test_user_prompt_without_examples() -> None:
    assert build_user_prompt("abc").startswith("Sentence to annotate")


# --- Cohere adapter (fake SDK, no network) ---------------------------------


class FakeCohereSDK:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def chat(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            finish_reason="COMPLETE",
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking="let me think"),
                    SimpleNamespace(type="text", text='{"labels": {}}'),
                ]
            ),
            usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=120, output_tokens=30)),
        )

    async def embed(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        vectors = [[1.0, 0.0] for _ in kwargs["texts"]]
        return SimpleNamespace(
            embeddings=SimpleNamespace(float_=vectors),
            meta=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=7)),
        )

    async def rerank(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=2, relevance_score=0.91),
                SimpleNamespace(index=0, relevance_score=0.2),
            ],
            meta=SimpleNamespace(billed_units=SimpleNamespace(search_units=1)),
        )


def test_cohere_chat_maps_request_and_response() -> None:
    sdk = FakeCohereSDK()
    client = CohereClient("k", sdk_client=sdk)
    result = asyncio.run(
        client.chat_json(
            model=COMMAND_A, system="S", user="U", schema={"type": "object"}, thinking_budget=2048
        )
    )
    assert sdk.kwargs["model"] == "command-a-03-2025"
    assert sdk.kwargs["response_format"] == {"type": "json_object", "schema": {"type": "object"}}
    assert sdk.kwargs["messages"][0] == {"role": "system", "content": "S"}
    assert sdk.kwargs["thinking"] == {"type": "enabled", "token_budget": 2048}
    assert result.text == '{"labels": {}}'  # thinking block ignored
    assert result.usage == Usage(input_tokens=120, output_tokens=30, calls=1)


def test_cohere_chat_without_thinking_does_not_send_it() -> None:
    sdk = FakeCohereSDK()
    asyncio.run(
        CohereClient("k", sdk_client=sdk).chat_json(
            model=COMMAND_A, system="S", user="U", schema={}
        )
    )
    assert "thinking" not in sdk.kwargs


def test_cohere_errors_are_wrapped() -> None:
    class Boom(FakeCohereSDK):
        async def chat(self, **kwargs: Any) -> Any:
            raise RuntimeError("429 too many requests")

    with pytest.raises(LLMError, match="429"):
        asyncio.run(
            CohereClient("k", sdk_client=Boom()).chat_json(
                model=COMMAND_A, system="", user="", schema={}
            )
        )


def test_cohere_rejects_other_providers() -> None:
    with pytest.raises(LLMError, match="cannot serve"):
        asyncio.run(
            CohereClient("k", sdk_client=FakeCohereSDK()).chat_json(
                model=ModelRef.parse("openai/x"), system="", user="", schema={}
            )
        )


def test_cohere_embed_and_rerank() -> None:
    sdk = FakeCohereSDK()
    client = CohereClient("k", sdk_client=sdk)
    emb = asyncio.run(
        client.embed(
            model=ModelRef.parse("cohere/embed-v4.0"), texts=["a", "b"], input_type="search_query"
        )
    )
    assert sdk.kwargs["input_type"] == "search_query"
    assert emb.vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert emb.usage.embed_tokens == 7
    rr = asyncio.run(
        client.rerank(
            model=ModelRef.parse("cohere/rerank-v3.5"),
            query="q",
            documents=["x", "y", "z"],
            top_n=2,
        )
    )
    assert rr.ranking == [(2, 0.91), (0, 0.2)]
    assert rr.usage.search_units == 1


# --- factory / router -------------------------------------------------------


def test_router_dispatches_by_provider() -> None:
    fake = FakeLLMClient(lambda call: labels_payload())
    router = ProviderRouter({"cohere": fake})
    asyncio.run(router.chat_json(model=COMMAND_A, system="", user="", schema={}))
    assert len(fake.chat_calls) == 1
    with pytest.raises(LLMError, match="no client"):
        asyncio.run(
            router.chat_json(model=ModelRef.parse("openai/x"), system="", user="", schema={})
        )


def test_build_llm_client_rejects_unsupported_provider(tmp_path: Any) -> None:
    import yaml

    data = yaml.safe_load(REPO_CONFIG.read_text())
    data["providers"]["openai"] = {"api_key_env": "OPENAI_API_KEY"}
    data["agents"]["adjudicator"]["model"] = "openai/x"
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_runtime_config(
        path, env_file=None, environ={"COHERE_API_KEY": "k", "OPENAI_API_KEY": "k"}
    )
    with pytest.raises(ConfigError, match="no adapter"):
        build_llm_client(cfg)
