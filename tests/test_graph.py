"""End-to-end LangGraph runs with the fake LLM (no network)."""

from __future__ import annotations

import asyncio
import importlib.util
from typing import Any

import pytest

pytest.importorskip("langgraph")

from scyibl.graph.build import InputError, build_graph, run_graph
from scyibl.llm.fake import ChatCall, FakeLLMClient
from scyibl.taxonomy import Category
from tests.helpers import REPO_ROOT, app_config, labels_payload


def _run(app: Any, client: FakeLLMClient, message: str) -> Any:
    return asyncio.run(run_graph(build_graph(app, client), app, message, request_id="t-1"))


def test_confident_path_skips_adjudicator() -> None:
    client = FakeLLMClient(lambda call: labels_payload({Category.PERMA: 0.95}, default=0.02))
    result = _run(app_config(), client, "  I feel   grateful today. ")
    assert result.escalated is False
    assert result.predicted == ["PERMA"]
    assert result.models == ["cohere/command-a-03-2025"]
    assert len(client.chat_calls) == 1
    assert "<<<\nI feel grateful today.\n>>>" in client.chat_calls[0].user  # router normalised
    assert result.request_id == "t-1"
    assert result.cost_usd is None  # no pricing configured


def test_uncertain_path_escalates_to_adjudicator() -> None:
    def responder(call: ChatCall) -> dict[str, Any]:
        if "First assessment" in call.system:
            return labels_payload({Category.COPING: 0.9}, default=0.03)
        return labels_payload({Category.COPING: 0.5})

    client = FakeLLMClient(responder)
    result = _run(app_config(), client, "I take a walk when stressed.")
    assert result.escalated is True
    assert result.predicted == ["Coping"]
    assert result.models == ["cohere/command-a-03-2025", "cohere/command-a-reasoning-08-2025"]
    assert any("ambiguous" in r for r in result.escalation_reasons)


def test_max_depth_zero_blocks_escalation() -> None:
    client = FakeLLMClient(lambda call: labels_payload({Category.COPING: 0.5}))
    result = _run(app_config(max_depth=0), client, "x")
    assert result.escalated is False
    assert result.escalation_reasons[-1].startswith("not escalated: max_depth")
    assert len(client.chat_calls) == 1


def test_cost_is_computed_when_pricing_configured() -> None:
    from scyibl.config.models import ModelPrice

    app = app_config()
    app = app.model_copy(
        update={
            "pricing": {
                "cohere/command-a-03-2025": ModelPrice(input_per_mtok=1_000_000, output_per_mtok=0)
            }
        }
    )
    client = FakeLLMClient(lambda call: labels_payload({Category.PERMA: 0.95}, default=0.02))
    result = _run(app, client, "x")
    assert result.cost_usd == result.input_tokens  # $1 per token in this fake price


@pytest.mark.parametrize("message", ["   ", "x" * 2001])
def test_router_rejects_bad_input(message: str) -> None:
    client = FakeLLMClient(lambda call: labels_payload())
    with pytest.raises(InputError):
        _run(app_config(), client, message)
    assert client.chat_calls == []


def test_memory_checkpointer_mode_runs() -> None:
    app = app_config()
    app = app.model_copy(update={"graph": app.graph.model_copy(update={"checkpointer": "memory"})})
    client = FakeLLMClient(lambda call: labels_payload({Category.PERMA: 0.95}, default=0.02))
    assert _run(app, client, "x").predicted == ["PERMA"]


def test_spike_script_dry_run(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    spec = importlib.util.spec_from_file_location(
        "spike", REPO_ROOT / "scripts" / "spike_cohere.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--dry-run", "--out", str(tmp_path)]) == 0
    reports = list(tmp_path.glob("spike-dry-run-*.json"))
    assert len(reports) == 1
    import json

    report = json.loads(reports[0].read_text())
    assert report["summary"]["succeeded"] == len(module.SENTENCES)
    assert report["summary"]["escalated"] >= 1
    assert report["retrieval"]["embed_ok"]
    assert report["retrieval"]["rerank_ok"]
