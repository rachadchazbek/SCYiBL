from __future__ import annotations

import json

import pytest

from scyibl.escalation import confidence, should_escalate
from scyibl.schema import ClassificationResult, LLMClassification, llm_output_schema
from scyibl.taxonomy import CATEGORIES, Category
from tests.helpers import app_config, labels_payload

FORBIDDEN_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "additionalProperties",
    "anyOf",
    "allOf",
    "oneOf",
    "$ref",
    "$defs",
    "minLength",
    "maxLength",
}


def _walk(node: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if isinstance(node, dict):
        out.append(node)
        for value in node.values():
            out.extend(_walk(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_walk(value))
    return out


# --- taxonomy & schema ------------------------------------------------------


def test_taxonomy_matches_original_dataset_columns() -> None:
    assert [CATEGORIES[c].label for c in Category] == [
        "CP",
        "Constructivism",
        "Coping",
        "ED",
        "ER/SR",
        "Empowerment",
        "Exposure/Habit",
        "PERMA",
    ]


def test_llm_schema_respects_cohere_constraints() -> None:
    schema = llm_output_schema()
    assert schema["type"] == "object"
    for node in _walk(schema):
        assert not FORBIDDEN_SCHEMA_KEYS & node.keys(), node
        if node.get("type") == "object":
            assert node.get("required"), f"object without required fields: {node}"
    assert set(schema["properties"]["labels"]["required"]) == {c.value for c in Category}
    json.dumps(schema)  # serialisable


def test_parse_wrapped_and_flat_payloads() -> None:
    payload = labels_payload({Category.PERMA: 0.9})
    wrapped = LLMClassification.from_llm(payload)
    flat = LLMClassification.from_llm(payload["labels"])
    assert wrapped == flat
    assert wrapped.perma.present is True


def test_scores_are_clamped_and_rationale_trimmed() -> None:
    payload = labels_payload()
    payload["labels"]["cp"].update(score=1.7, rationale="x" * 1000)
    payload["labels"]["ed"].update(score=-0.2)
    parsed = LLMClassification.from_llm(payload)
    assert parsed.cp.score == 1.0
    assert len(parsed.cp.rationale) == 300
    assert parsed.ed.score == 0.0


def test_missing_category_is_rejected() -> None:
    payload = labels_payload()
    del payload["labels"]["perma"]
    with pytest.raises(ValueError, match="perma"):
        LLMClassification.from_llm(payload)


def test_label_view_uses_display_labels() -> None:
    parsed = LLMClassification.from_llm(labels_payload({Category.ER_SR: 0.8}))
    view = ClassificationResult.label_view(parsed)
    assert view["ER/SR"].score == 0.8
    assert "er_sr" not in view


# --- escalation -------------------------------------------------------------


def test_confidence() -> None:
    assert confidence(LLMClassification.from_llm(labels_payload(default=0.0))) == 1.0
    assert confidence(LLMClassification.from_llm(labels_payload({Category.CP: 0.5}))) == 0.0


def test_confident_output_is_not_escalated() -> None:
    out = LLMClassification.from_llm(labels_payload({Category.PERMA: 0.95}, default=0.02))
    decision = should_escalate(out, app_config().escalation, depth=0, cost_usd=None)
    assert decision.escalate is False
    assert decision.reasons() == []


def test_low_confidence_and_ambiguity_trigger_escalation() -> None:
    out = LLMClassification.from_llm(labels_payload({Category.COPING: 0.45}))
    decision = should_escalate(out, app_config().escalation, depth=0, cost_usd=None)
    assert decision.escalate is True
    joined = " ".join(decision.triggers)
    assert "confidence" in joined
    assert "Coping" in joined


def test_inconsistent_present_flag_triggers_escalation() -> None:
    payload = labels_payload({Category.PERMA: 0.95}, default=0.02)
    payload["labels"]["perma"]["present"] = False
    decision = should_escalate(
        LLMClassification.from_llm(payload), app_config().escalation, depth=0, cost_usd=None
    )
    assert decision.escalate is True
    assert any("disagree" in t for t in decision.triggers)


@pytest.mark.parametrize(
    ("kwargs", "blocked"),
    [
        ({"depth": 2, "cost_usd": None}, "max_depth"),
        ({"depth": 0, "cost_usd": 5.0}, "cost budget"),
        ({"depth": 0, "cost_usd": None, "adjudicator_enabled": False}, "adjudicator disabled"),
    ],
)
def test_guards_block_escalation(kwargs: dict[str, object], blocked: str) -> None:
    out = LLMClassification.from_llm(labels_payload({Category.COPING: 0.5}))
    decision = should_escalate(out, app_config().escalation, **kwargs)  # type: ignore[arg-type]
    assert decision.escalate is False
    assert decision.blocked_by is not None
    assert blocked in decision.blocked_by
    assert decision.reasons()[-1].startswith("not escalated")
