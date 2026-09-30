"""LLMAdapter Protocol and related models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def _validate_temperature(value: float) -> None:
    if value < 0.0 or value > 2.0:
        raise ValueError(f"temperature must be between 0.0 and 2.0, got {value}")


def _validate_max_tokens(value: int) -> None:
    if value <= 0:
        raise ValueError(f"max_tokens must be positive, got {value}")


@dataclass(frozen=True)
class LLMRequest:
    """Request to an LLM."""

    system_prompt: str
    messages: list[dict]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt_cached: str | None = None
    system_prompt_dynamic: str | None = None
    # 是否允许模型进行隐式推理（thinking）。
    #   None  = 由适配器默认策略决定
    #   False = 强制关闭思考（快，适合每轮对话/情绪评估）
    #   True  = 保留思考（慢但更深，适合会话结束的深度整合）
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        _validate_temperature(self.temperature)
        _validate_max_tokens(self.max_tokens)


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    usage: dict


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        _validate_temperature(self.temperature)
        _validate_max_tokens(self.max_tokens)


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM communication adapters."""

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
