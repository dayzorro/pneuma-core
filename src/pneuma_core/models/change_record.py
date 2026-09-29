"""ChangeRecord: internal state change logging."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChangeRecord:
    """内部状态变化的记录。

    type: "emotion_updated", "memory_added", "goal_updated" 等
    before: 变更前的状态（None = 新增）
    after: 变更后的状态
    """

    id: str
    character_id: str
    type: str
    before: dict | None
    after: dict
    reason: str
    timestamp: datetime
