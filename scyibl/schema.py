"""Output schemas (groundwork for issue #5).

Two layers:

* :class:`LLMClassification` is what the model returns. Its JSON Schema
  (:func:`llm_output_schema`) is written by hand to respect Cohere's structured-output limits:
  no ``minimum``/``maximum``, no ``additionalProperties``, no ``anyOf``/``$ref``, and every
  object has ``required`` fields.
* :class:`ClassificationResult` is the public result returned by the graph and the API.
  Range checks happen here, in Pydantic, not in the schema sent to the model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scyibl.taxonomy import CATEGORIES, Category

SCHEMA_VERSION = "0.1"
RATIONALE_MAX_CHARS = 300


class LabelScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    present: bool
    score: float = Field(description="Probability in [0, 1] that the category applies")
    rationale: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> float:
        score = float(value)
        if score != score:  # NaN
            raise ValueError("score is NaN")
        return min(1.0, max(0.0, score))

    @field_validator("rationale", mode="before")
    @classmethod
    def _trim_rationale(cls, value: Any) -> str:
        return str(value or "").strip()[:RATIONALE_MAX_CHARS]


class LLMClassification(BaseModel):
    """Model output: one :class:`LabelScore` per category."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    cp: LabelScore
    constructivism: LabelScore
    coping: LabelScore
    ed: LabelScore
    er_sr: LabelScore
    empowerment: LabelScore
    exposure_habit: LabelScore
    perma: LabelScore

    @classmethod
    def from_llm(cls, data: dict[str, Any]) -> LLMClassification:
        """Accept both ``{"labels": {...}}`` (our schema) and a flat object."""
        labels = data.get("labels", data)
        return cls.model_validate(labels)

    def scores(self) -> dict[Category, LabelScore]:
        return {category: getattr(self, category.value) for category in Category}


def llm_output_schema() -> dict[str, Any]:
    label = {
        "type": "object",
        "properties": {
            "present": {"type": "boolean"},
            "score": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["present", "score", "rationale"],
    }
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "object",
                "properties": {category.value: label for category in Category},
                "required": [category.value for category in Category],
            }
        },
        "required": ["labels"],
    }


class ClassificationResult(BaseModel):
    """Public result (provisional until #5 is finalised; safety is added in #8)."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    request_id: str
    labels: dict[str, LabelScore] = Field(description="Keyed by display label, e.g. 'ER/SR'")
    predicted: list[str] = Field(description="Display labels with score >= 0.5")
    confidence: float
    escalated: bool
    escalation_reasons: list[str]
    models: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    @staticmethod
    def label_view(classification: LLMClassification) -> dict[str, LabelScore]:
        return {CATEGORIES[c].label: s for c, s in classification.scores().items()}
