"""Typed schema for ``config/agents.yaml``.

This file describes *behaviour* (which model each agent uses, escalation thresholds, ...).
It never contains secrets: those are referenced by environment-variable name and resolved
in :mod:`scyibl.config.settings`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REQUIRED_AGENTS: frozenset[str] = frozenset({"router", "classifier", "adjudicator", "safety"})
RiskLevel = Literal["low", "elevated", "urgent"]


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so typos in the YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRef(_Strict):
    """A ``provider/model`` reference, e.g. ``cohere/command-a-03-2025``."""

    provider: str
    name: str

    @classmethod
    def parse(cls, value: str) -> ModelRef:
        provider, sep, name = value.partition("/")
        if not sep or not provider or not name:
            raise ValueError(f"model must look like 'provider/model-name', got {value!r}")
        return cls(provider=provider.strip().lower(), name=name.strip())

    def __str__(self) -> str:
        return f"{self.provider}/{self.name}"


def _to_model_ref(value: object) -> object:
    return ModelRef.parse(value) if isinstance(value, str) else value


class ProviderConfig(_Strict):
    api_key_env: str = Field(description="Name of the environment variable holding the API key")
    base_url: str | None = None


class AgentConfig(_Strict):
    model: ModelRef
    enabled: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    few_shot_k: int | None = Field(default=None, ge=0, le=50)
    thinking_budget: int | None = Field(
        default=None, ge=0, description="Reasoning-token budget for reasoning models (Cohere)"
    )

    _parse_model = field_validator("model", mode="before")(_to_model_ref)


class RetrievalConfig(_Strict):
    embed_model: ModelRef
    rerank_model: ModelRef | None = None
    candidates: int = Field(default=30, gt=0, description="Neighbours fetched before re-ranking")

    _parse_models = field_validator("embed_model", "rerank_model", mode="before")(_to_model_ref)


class EscalationConfig(_Strict):
    min_confidence: float = Field(ge=0.0, le=1.0)
    ambiguity_band: tuple[float, float]
    max_depth: int = Field(default=2, ge=0, le=5)
    max_cost_usd_per_request: float = Field(gt=0)

    @field_validator("ambiguity_band")
    @classmethod
    def _band_is_ordered(cls, band: tuple[float, float]) -> tuple[float, float]:
        low, high = band
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"ambiguity_band must satisfy 0 <= low < high <= 1, got {band}")
        return band


class DatasetConfig(_Strict):
    repo_id: str
    revision: str = "main"
    token_env: str = "HF_TOKEN"  # noqa: S105 - env-var *name*, not a secret
    required: bool = Field(
        default=False,
        description="When true, the token must be present at startup (needed by retrieval/eval)",
    )


class ClientsConfig(_Strict):
    api_keys_file: str = "secrets/client_keys.json"
    rate_limit_per_minute: int = Field(default=60, gt=0)


class EmailAlertConfig(_Strict):
    enabled: bool = False
    to_env: str = "ALERT_EMAIL_TO"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    username_env: str = "ALERT_SMTP_USERNAME"
    password_env: str = "ALERT_SMTP_PASSWORD"  # noqa: S105 - env-var *name*, not a secret


class SheetAlertConfig(_Strict):
    enabled: bool = False
    sheet_id_env: str = "ALERT_SHEET_ID"
    credentials_file_env: str = "GOOGLE_APPLICATION_CREDENTIALS"


class AlertsConfig(_Strict):
    min_risk_level: RiskLevel = "elevated"
    email: EmailAlertConfig = EmailAlertConfig()
    google_sheet: SheetAlertConfig = SheetAlertConfig()


class GraphConfig(_Strict):
    checkpointer: Literal["none", "memory"] = Field(
        default="none",
        description="'memory' keeps graph state in RAM per thread; nothing is persisted to disk",
    )
    recursion_limit: int = Field(default=25, gt=0, le=200)
    max_message_chars: int = Field(default=2000, gt=0)


class ModelPrice(_Strict):
    """USD per million tokens. Used for the per-request cost budget."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class AppConfig(_Strict):
    providers: dict[str, ProviderConfig]
    agents: dict[str, AgentConfig]
    retrieval: RetrievalConfig
    escalation: EscalationConfig
    dataset: DatasetConfig
    clients: ClientsConfig = ClientsConfig()
    alerts: AlertsConfig = AlertsConfig()
    graph: GraphConfig = GraphConfig()
    pricing: dict[str, ModelPrice] = Field(
        default_factory=dict, description="Keyed by 'provider/model'; empty = cost unknown"
    )

    @field_validator("pricing")
    @classmethod
    def _pricing_keys_are_model_refs(cls, value: dict[str, ModelPrice]) -> dict[str, ModelPrice]:
        return {str(ModelRef.parse(key)): price for key, price in value.items()}

    @field_validator("providers")
    @classmethod
    def _lowercase_providers(cls, value: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        return {name.lower(): cfg for name, cfg in value.items()}

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        missing = REQUIRED_AGENTS - self.agents.keys()
        if missing:
            raise ValueError(f"missing required agents: {sorted(missing)}")
        for agent in ("router", "classifier"):
            if not self.agents[agent].enabled:
                raise ValueError(f"agent '{agent}' cannot be disabled")
        unknown = {
            f"{where} -> {ref}"
            for where, ref in self.model_refs()
            if ref.provider not in self.providers
        }
        if unknown:
            raise ValueError(
                f"undeclared provider(s) referenced: {sorted(unknown)}; "
                f"declared providers: {sorted(self.providers)}"
            )
        return self

    def model_refs(self) -> list[tuple[str, ModelRef]]:
        """Every model reference that is active in this configuration."""
        refs = [(f"agents.{n}.model", a.model) for n, a in self.agents.items() if a.enabled]
        refs.append(("retrieval.embed_model", self.retrieval.embed_model))
        if self.retrieval.rerank_model is not None:
            refs.append(("retrieval.rerank_model", self.retrieval.rerank_model))
        return refs

    def price_for(self, model: ModelRef) -> ModelPrice | None:
        return self.pricing.get(str(model))

    def providers_in_use(self) -> set[str]:
        return {ref.provider for _, ref in self.model_refs()}
