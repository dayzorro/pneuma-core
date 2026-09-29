"""Character identity model."""

from dataclasses import dataclass

from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values


@dataclass(frozen=True)
class Character:
    """角色的身份标识。

    包含不变属性（性格、价值观）与自由描述（简介、外貌、口吻等）。
    personality_description / values_description 可来自 YAML 直写或 LLM 生成（Phase 1.5）。
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
