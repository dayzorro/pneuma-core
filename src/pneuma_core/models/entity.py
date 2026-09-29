"""EntityContext: unified context format for users and characters.

Issue #121: Entity 统一格式
让角色与用户能够以同一套接口处理。
角色特有的字段（personality、speaking_style）在用户侧为 None。

This is an ADDITIVE abstraction layer -- it wraps existing models
(Character, UserContext) without modifying them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pneuma_core.models.character import Character
    from pneuma_core.runtime.user_context import UserContext

from pneuma_core.models.goals import GoalTree
from pneuma_core.models.personality import Personality
from pneuma_core.models.relation import Relation

_VALID_ENTITY_TYPES = ("character", "user")


@dataclass
class EntityContext:
    """Unified context that both users and characters share.

    Fields that don't apply to an entity type are None.
    For example, users have no personality or speaking_style.
    """

    entity_id: str
    entity_type: str  # "character" | "user"

    # Identity layer (static)
    profile: str | None = None
    values_text: str | None = None
    background: str | None = None

    # Character-specific (None for users)
    personality: Personality | None = None
    speaking_style: str | None = None

    # Shared structures
    goals: GoalTree | None = None
    relations: list[Relation] | None = None
    glossary: str | None = None
    projects: dict[str, str] | None = None
    diary_summary: str | None = None

    def __post_init__(self) -> None:
        if self.entity_type not in _VALID_ENTITY_TYPES:
            raise ValueError(
                f"entity_type must be one of {_VALID_ENTITY_TYPES}, "
                f"got '{self.entity_type}'"
            )


def build_from_character(
    character: Character,
    *,
    goals: GoalTree | None = None,
    relations: list[Relation] | None = None,
    glossary: str | None = None,
    projects: dict[str, str] | None = None,
    diary_summary: str | None = None,
) -> EntityContext:
    """Build EntityContext from a Character dataclass.

    Maps Character fields to the unified EntityContext format:
    - character.profile -> profile
    - character.values_description -> values_text
    - character.background -> background
    - character.personality -> personality
    - character.speaking_style -> speaking_style
    """
    return EntityContext(
        entity_id=character.id,
        entity_type="character",
        profile=character.profile,
        values_text=character.values_description,
        background=character.background,
        personality=character.personality,
        speaking_style=character.speaking_style,
        goals=goals,
        relations=relations,
        glossary=glossary,
        projects=projects,
        diary_summary=diary_summary,
    )


def build_from_user_context(
    user_context: UserContext,
    *,
    entity_id: str = "user",
    goals: GoalTree | None = None,
    relations: list[Relation] | None = None,
) -> EntityContext:
    """Build EntityContext from a UserContext dataclass.

    Maps UserContext fields to the unified EntityContext format:
    - user_context.identity -> profile
    - user_context.values -> values_text
    - user_context.core_experiences -> background
    - user_context.glossary -> glossary
    - user_context.projects -> projects
    - user_context.diary_summary -> diary_summary
    """
    return EntityContext(
        entity_id=entity_id,
        entity_type="user",
        profile=user_context.identity,
        values_text=user_context.values,
        background=user_context.core_experiences,
        personality=None,
        speaking_style=None,
        goals=goals,
        relations=relations,
        glossary=user_context.glossary,
        projects=user_context.projects if user_context.projects else {},
        diary_summary=user_context.diary_summary,
    )
