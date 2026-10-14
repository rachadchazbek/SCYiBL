from __future__ import annotations

from pathlib import Path
from typing import Any

from scyibl.config import load_app_config
from scyibl.config.models import AppConfig
from scyibl.taxonomy import Category

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO_ROOT / "config" / "agents.yaml"


def app_config(**escalation: Any) -> AppConfig:
    app = load_app_config(REPO_CONFIG)
    if escalation:
        esc = app.escalation.model_copy(update=escalation)
        app = app.model_copy(update={"escalation": esc})
    return app


def labels_payload(
    scores: dict[Category, float] | None = None, default: float = 0.05
) -> dict[str, Any]:
    scores = scores or {}
    return {
        "labels": {
            c.value: {
                "present": scores.get(c, default) >= 0.5,
                "score": scores.get(c, default),
                "rationale": f"because {c.value}",
            }
            for c in Category
        }
    }
