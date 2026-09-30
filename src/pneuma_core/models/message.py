"""Unified message interface: MessageInput, MessageOutput, ToolCall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pneuma_core.models.change_record import ChangeRecord
from pneuma_core.models.diagnostic import DiagnosticInfo
from pneuma_core.models.emotion import EmotionalState


@dataclass(frozen=True)
class SystemMessage:
    """System-level message for error/warning/info visibility.

    These messages are separate from character responses and used to
    communicate internal system status (e.g., memory search failures,
    LLM errors) to the adapter layer.
    """

    type: str  # "warning", "info", "error"
    message: str  # Human-readable message
    component: str  # "memory_search", "emotion", "todo", "llm", etc.


@dataclass(frozen=True)
class StructuredResponse:
    """Parsed structured response from LLM (speech/thought/action)."""

    speech: str | None = None
    thought: str | None = None
    action: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """External tool invocation request."""

    name: str
    arguments: dict = field(default_factory=dict)


_VALID_SENDER_TYPES = frozenset({"human", "character", "system"})


@dataclass(frozen=True)
class MessageInput:
    """Unified input from any message source."""

    content: str
    sender_id: str
    sender_name: str
    sender_type: Literal["human", "character", "system"]
    channel: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("content must not be empty")
        if self.sender_type not in _VALID_SENDER_TYPES:
            raise ValueError(
                f"sender_type must be one of {_VALID_SENDER_TYPES}, got '{self.sender_type}'"
            )


@dataclass(frozen=True)
class MessageOutput:
    """Unified response output."""

    content: str
    emotion: EmotionalState
    thought: str | None = None
    action: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    internal_changes: list[ChangeRecord] = field(default_factory=list)
    diagnostic: DiagnosticInfo | None = None
    system_messages: list[SystemMessage] = field(default_factory=list)
    # 本轮联网检索命中的来源（空列表 = 本轮未联网或没有结果）
    web_sources: list[dict[str, str]] = field(default_factory=list)
