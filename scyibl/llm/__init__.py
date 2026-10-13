"""Provider-agnostic LLM layer (ADR-001, issue #1)."""

from scyibl.llm.base import (
    ChatResult,
    EmbedResult,
    LLMClient,
    LLMError,
    RerankResult,
    Usage,
)
from scyibl.llm.fake import FakeLLMClient

__all__ = [
    "ChatResult",
    "EmbedResult",
    "FakeLLMClient",
    "LLMClient",
    "LLMError",
    "RerankResult",
    "Usage",
]
