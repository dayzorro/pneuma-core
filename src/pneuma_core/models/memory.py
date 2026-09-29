"""Memory data models: EpisodicMemory, SemanticMemory."""

from dataclasses import dataclass, field
from datetime import datetime


def _validate_range(value: float, name: str, min_val: float, max_val: float) -> float:
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


@dataclass(frozen=True)
class EpisodicMemory:
    """情节记忆：具体事件的记录。

    emotional_valence: -1.0〜1.0（不快〜愉悦）
    importance: 0.0〜1.0
    embedding: 1536 维向量（None = 尚未生成）
    """

    id: str
    character_id: str
    content: str
    timestamp: datetime
    emotional_valence: float
    importance: float
    conversation_id: str | None = None
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        _validate_range(self.emotional_valence, "emotional_valence", -1.0, 1.0)
        _validate_range(self.importance, "importance", 0.0, 1.0)


@dataclass(frozen=True)
class SemanticMemory:
    """语义记忆：泛化后的知识。

    confidence: 0.0〜1.0（随支撑它的事件数量而提高）
    """

    id: str
    character_id: str
    content: str
    confidence: float
    source_episode_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        _validate_range(self.confidence, "confidence", 0.0, 1.0)
