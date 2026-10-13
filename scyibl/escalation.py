"""Escalation policy (groundwork for #7): a pure function behind the conditional edge."""

from __future__ import annotations

from dataclasses import dataclass

from scyibl.config.models import EscalationConfig
from scyibl.schema import LLMClassification
from scyibl.taxonomy import CATEGORIES


def confidence(classification: LLMClassification) -> float:
    """1.0 when every score is at 0 or 1, 0.0 when any score sits at 0.5."""
    return round(min(abs(s.score - 0.5) * 2 for s in classification.scores().values()), 4)


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    triggers: tuple[str, ...]
    """Why escalation was wanted (empty when the first pass is confident)."""
    blocked_by: str | None = None
    """Set when escalation was wanted but a guard (depth, budget) prevented it."""

    def reasons(self) -> list[str]:
        out = list(self.triggers)
        if self.blocked_by:
            out.append(f"not escalated: {self.blocked_by}")
        return out


def should_escalate(
    classification: LLMClassification,
    cfg: EscalationConfig,
    *,
    depth: int,
    cost_usd: float | None,
    adjudicator_enabled: bool = True,
) -> EscalationDecision:
    triggers: list[str] = []
    conf = confidence(classification)
    if conf < cfg.min_confidence:
        triggers.append(f"confidence {conf:.2f} < {cfg.min_confidence}")

    low, high = cfg.ambiguity_band
    ambiguous = [
        CATEGORIES[c].label for c, s in classification.scores().items() if low <= s.score <= high
    ]
    if ambiguous:
        triggers.append(f"ambiguous labels in [{low}, {high}]: {', '.join(ambiguous)}")

    inconsistent = [
        CATEGORIES[c].label
        for c, s in classification.scores().items()
        if s.present != (s.score >= 0.5)
    ]
    if inconsistent:
        triggers.append(f"present/score disagree: {', '.join(inconsistent)}")

    if not triggers:
        return EscalationDecision(escalate=False, triggers=())
    if not adjudicator_enabled:
        return EscalationDecision(False, tuple(triggers), "adjudicator disabled")
    if depth >= cfg.max_depth:
        return EscalationDecision(False, tuple(triggers), f"max_depth {cfg.max_depth} reached")
    if cost_usd is not None and cost_usd >= cfg.max_cost_usd_per_request:
        return EscalationDecision(
            False, tuple(triggers), f"cost budget ${cfg.max_cost_usd_per_request} reached"
        )
    return EscalationDecision(escalate=True, triggers=tuple(triggers))
