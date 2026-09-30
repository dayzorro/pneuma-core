"""PromptCache: static/dynamic section separation for LLM prompt caching."""

from __future__ import annotations

from dataclasses import dataclass

from pneuma_core.knowledge.models import KnowledgeHit
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.relation import Relation
from pneuma_core.runtime.prompt_builder import PromptBuilder, UserContextConfig
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import UserContextSearchResult


@dataclass
class CachedPrompt:
    """Prompt split into static and dynamic sections."""

    static_section: str
    dynamic_section: str

    @property
    def full_prompt(self) -> str:
        """Combine static and dynamic sections."""
        return f"{self.static_section}\n\n{self.dynamic_section}"


class PromptCache:
    """Caches static prompt sections, rebuilding only dynamic parts each turn.

    Static sections (cached):
        - Profile, Personality, Values, UserContext Tier 1, Speaking Style

    Dynamic sections (rebuilt each turn):
        - UserContext Tier 2+3, Memory, Goals, Emotional State
    """

    def __init__(self) -> None:
        self._builder = PromptBuilder()
        self._cached_static: str | None = None
        self._cached_character_id: str | None = None

    def build(
        self,
        character: Character,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
        user_context_config: UserContextConfig | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list[Relation] | None = None,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
        knowledge_hits: list[KnowledgeHit] | None = None,
    ) -> CachedPrompt:
        """Build a CachedPrompt with static/dynamic separation."""
        static = self._get_or_build_static(
            character,
            user_context=user_context,
            user_context_config=user_context_config,
            user_goal_tree=user_goal_tree,
            character_relations=character_relations,
        )
        dynamic = self._build_dynamic(
            emotional_state,
            goal_tree,
            memories,
            user_context=user_context,
            user_context_search_results=user_context_search_results,
            user_context_config=user_context_config,
            user_tasks=user_tasks,
            character_tasks=character_tasks,
            knowledge_hits=knowledge_hits,
        )
        return CachedPrompt(static_section=static, dynamic_section=dynamic)

    def invalidate(self) -> None:
        """Force cache invalidation."""
        self._cached_static = None
        self._cached_character_id = None

    def _get_or_build_static(
        self,
        character: Character,
        *,
        user_context: UserContext | None = None,
        user_context_config: UserContextConfig | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list[Relation] | None = None,
    ) -> str:
        """Return cached static section or rebuild if character changed."""
        if (
            self._cached_static is not None
            and self._cached_character_id == character.id
        ):
            return self._cached_static

        self._cached_static = self._builder.build_static_sections(
            character,
            user_context=user_context,
            user_context_config=user_context_config,
            user_goal_tree=user_goal_tree,
            character_relations=character_relations,
        )
        self._cached_character_id = character.id
        return self._cached_static

    def _build_dynamic(
        self,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
        user_context_config: UserContextConfig | None = None,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
        knowledge_hits: list[KnowledgeHit] | None = None,
    ) -> str:
        """Build dynamic sections (always rebuilt)."""
        return self._builder.build_dynamic_sections(
            emotional_state=emotional_state,
            goal_tree=goal_tree,
            memories=memories,
            user_context=user_context,
            user_context_search_results=user_context_search_results,
            user_context_config=user_context_config,
            user_tasks=user_tasks,
            character_tasks=character_tasks,
            knowledge_hits=knowledge_hits,
        )
