"""Character identity model."""

from dataclasses import dataclass

from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values


@dataclass(frozen=True)
class Character:
    """角色的身份标识。

    包含不变属性（性格、价值观）与自由描述（简介、外貌、口吻等）。
    personality_description / values_description 可来自 YAML 直写或 LLM 生成（Phase 1.5）。

    职业化设定（可选，用于「前台/客服」这类有固定岗位的角色）：
        role_title: 岗位名称，例如「华润万家 · 顾客服务前台」
        job_description: 岗位职责与服务范围
        service_rules: 服务守则（行为边界、话术要求、转交流程）
        emotional_expressiveness: 情绪外显系数，0.0〜1.0。1.0=情绪完全外显，
            越小情绪越往性格基线收敛（服务型岗位用来保持专业稳定）
    """

    id: str
    name: str
    personality: Personality
    values: Values
    profile: str | None = None
    appearance: str | None = None
    speaking_style: str | None = None
    background: str | None = None
    personality_description: str | None = None
    values_description: str | None = None
    role_title: str | None = None
    job_description: str | None = None
    service_rules: str | None = None
    emotional_expressiveness: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.emotional_expressiveness <= 1.0):
            raise ValueError(
                "emotional_expressiveness must be between 0.0 and 1.0, "
                f"got {self.emotional_expressiveness}"
            )

