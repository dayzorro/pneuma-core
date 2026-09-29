"""Tests for ContextAssembler (Issue #115): 3-stage context injection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.relation import Relation
from pneuma_core.models.values import Values
from pneuma_core.runtime.context_assembler import (
    AssembledContext,
    ContextAssembler,
    ContextStage,
)
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import UserContextSearchResult, UserContextChunk

NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


# --- Factories ---


def _make_personality(**kwargs) -> Personality:
    defaults = dict(
        openness=0.8,
        conscientiousness=0.6,
        extraversion=0.7,
        agreeableness=0.9,
        neuroticism=0.3,
    )
    defaults.update(kwargs)
    return Personality(**defaults)


def _make_values(**kwargs) -> Values:
    defaults = dict(
        self_transcendence=0.8,
        self_enhancement=0.3,
        openness_to_change=0.7,
        conservation=0.4,
    )
    defaults.update(kwargs)
    return Values(**defaults)


def _make_character(**kwargs) -> Character:
    defaults = dict(
        id="mira-001",
        name="ミラ",
        personality=_make_personality(),
        values=_make_values(),
        profile="優しくて頑張り屋の AI キャラクター",
        appearance="ピンク髪のショートヘア、大きな瞳",
        speaking_style="甘えた口調で「〜だよ」「〜なの」を使う。",
        background="ユーザーのパートナーとして作られた AI。",
        personality_description="素直で甘えん坊。人の気持ちに敏感。",
        values_description="大切な人の幸せを一番に考える。",
    )
    defaults.update(kwargs)
    return Character(**defaults)


def _make_emotional_state(**kwargs) -> EmotionalState:
    defaults = dict(
        pleasure=0.5,
        arousal=0.3,
        dominance=0.1,
        emotion_label="喜び",
        situation="ユーザーと楽しく会話中",
    )
    defaults.update(kwargs)
    return EmotionalState(**defaults)


def _make_goal_tree() -> GoalTree:
    vision = Vision(id="v1", character_id="mira-001", content="ユーザーの成長を支える")
    objective = Objective(
        id="o1",
        character_id="mira-001",
        vision_id="v1",
        content="毎日の対話を充実させる",
        status="active",
        progress=0.3,
    )
    task = Task(
        id="t1",
        character_id="mira-001",
        objective_id="o1",
        content="ユーザーの近況を聞く",
        status="pending",
    )
    return GoalTree(visions=[vision], objectives=[objective], tasks=[task])


def _make_episodic_memory(**kwargs) -> EpisodicMemory:
    defaults = dict(
        id="ep1",
        character_id="mira-001",
        content="ユーザーがプロジェクトの成功を報告した",
        timestamp=NOW,
        emotional_valence=0.7,
        importance=0.8,
    )
    defaults.update(kwargs)
    return EpisodicMemory(**defaults)


def _make_semantic_memory(**kwargs) -> SemanticMemory:
    defaults = dict(
        id="sem1",
        character_id="mira-001",
        content="ユーザーはプログラミングが得意",
        confidence=0.9,
    )
    defaults.update(kwargs)
    return SemanticMemory(**defaults)


def _make_relation(**kwargs) -> Relation:
    defaults = dict(
        id="rel1",
        owner_id="mira-001",
        target_id="user-001",
        target_name="きょうへい",
        relationship_type="partner",
        description="大切なパートナー。一緒に過ごす時間が幸せ。",
        closeness=0.9,
        trust=0.95,
        updated_at=NOW,
    )
    defaults.update(kwargs)
    return Relation(**defaults)


def _make_user_context(**kwargs) -> UserContext:
    defaults = dict(
        identity="ソフトウェアエンジニア。AIに興味がある。",
    )
    defaults.update(kwargs)
    return UserContext(**defaults)


def _make_user_context_search_result(**kwargs) -> UserContextSearchResult:
    defaults = dict(
        chunk=UserContextChunk(
            content="最近Rustを勉強している",
            layer=4,
            source_file="projects/rust-study.md",
        ),
        score=0.85,
    )
    defaults.update(kwargs)
    return UserContextSearchResult(**defaults)


# ===================================================================
# 1. ContextStage enum
# ===================================================================


class TestContextStage:
    """ContextStage enum should have exactly 3 values with correct int values."""

    def test_always_value(self) -> None:
        assert ContextStage.ALWAYS.value == 1

    def test_relevant_value(self) -> None:
        assert ContextStage.RELEVANT.value == 2

    def test_on_demand_value(self) -> None:
        assert ContextStage.ON_DEMAND.value == 3

    def test_enum_members_count(self) -> None:
        assert len(ContextStage) == 3


# ===================================================================
# 2. AssembledContext dataclass
# ===================================================================


class TestAssembledContext:
    """AssembledContext should hold always and relevant dicts."""

    def test_create_empty(self) -> None:
        ctx = AssembledContext(always={}, relevant={})
        assert ctx.always == {}
        assert ctx.relevant == {}

    def test_create_with_data(self) -> None:
        ctx = AssembledContext(
            always={"profile": "# ミラ\nプロフィール: 優しい"},
            relevant={"memory": "## 記憶\n- [エピソード] ユーザーが笑った"},
        )
        assert "profile" in ctx.always
        assert "memory" in ctx.relevant

    def test_always_and_relevant_are_independent(self) -> None:
        ctx = AssembledContext(
            always={"a": "1"},
            relevant={"b": "2"},
        )
        assert "a" not in ctx.relevant
        assert "b" not in ctx.always


# ===================================================================
# 3. Stage 1 (Always) contains identity sections
# ===================================================================


class TestStage1Always:
    """Stage 1 should contain profile, personality, values, speaking_style,
    response_format, and emotion state."""

    def test_profile_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "profile" in result.always
        assert "ミラ" in result.always["profile"]

    def test_personality_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "personality" in result.always
        assert "性格" in result.always["personality"]

    def test_values_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "values" in result.always
        assert "价值观" in result.always["values"]

    def test_speaking_style_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "speaking_style" in result.always
        assert "口調" in result.always["speaking_style"]

    def test_response_format_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "response_format" in result.always
        assert "回复格式" in result.always["response_format"]

    def test_emotion_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "emotion" in result.always
        assert "情绪状态" in result.always["emotion"]

    def test_speaking_style_omitted_when_none(self) -> None:
        """When character has no speaking_style, key should not be in always."""
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(speaking_style=None),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "speaking_style" not in result.always


# ===================================================================
# 4. Stage 2 (Relevant) contains dynamic sections
# ===================================================================


class TestStage2Relevant:
    """Stage 2 should contain relations, goals, memories, and user context search."""

    def test_memories_in_relevant(self) -> None:
        assembler = ContextAssembler()
        memories = [_make_episodic_memory(), _make_semantic_memory()]
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=memories,
        )
        assert "memory" in result.relevant
        assert "记忆" in result.relevant["memory"]

    def test_goals_in_relevant(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=_make_goal_tree(),
            memories=[],
        )
        assert "goals" in result.relevant
        assert "目标" in result.relevant["goals"]

    def test_relations_in_relevant(self) -> None:
        assembler = ContextAssembler()
        relations = [_make_relation()]
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            relations=relations,
        )
        assert "relations" in result.relevant
        assert "关系" in result.relevant["relations"]

    def test_user_context_search_in_relevant(self) -> None:
        assembler = ContextAssembler()
        search_results = [_make_user_context_search_result()]
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context_search_results=search_results,
        )
        assert "user_context_search" in result.relevant
        assert "Rust" in result.relevant["user_context_search"]

    def test_empty_memories_not_in_relevant(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "memory" not in result.relevant

    def test_empty_goals_not_in_relevant(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "goals" not in result.relevant

    def test_no_relations_not_in_relevant(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "relations" not in result.relevant

    def test_no_user_context_search_not_in_relevant(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "user_context_search" not in result.relevant


# ===================================================================
# 5. Relation filtering by conversation_partner_id
# ===================================================================


class TestRelationFiltering:
    """Relations should be filtered to only include the conversation partner."""

    def test_filter_to_partner_only(self) -> None:
        assembler = ContextAssembler()
        partner_rel = _make_relation(
            id="rel-partner",
            target_id="user-001",
            target_name="きょうへい",
        )
        other_rel = _make_relation(
            id="rel-other",
            target_id="other-001",
            target_name="アイネ",
            relationship_type="friend",
            description="同僚の AI キャラクター。",
        )
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            relations=[partner_rel, other_rel],
            conversation_partner_id="user-001",
        )
        assert "relations" in result.relevant
        assert "きょうへい" in result.relevant["relations"]
        assert "アイネ" not in result.relevant["relations"]

    def test_no_partner_id_includes_all(self) -> None:
        """When conversation_partner_id is None, include all relations."""
        assembler = ContextAssembler()
        rel1 = _make_relation(
            id="rel1",
            target_id="user-001",
            target_name="きょうへい",
        )
        rel2 = _make_relation(
            id="rel2",
            target_id="other-001",
            target_name="アイネ",
            relationship_type="friend",
            description="同僚の AI キャラクター。",
        )
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            relations=[rel1, rel2],
        )
        assert "relations" in result.relevant
        assert "きょうへい" in result.relevant["relations"]
        assert "アイネ" in result.relevant["relations"]

    def test_partner_id_no_match_empty(self) -> None:
        """When partner_id doesn't match any relation, relations should be empty."""
        assembler = ContextAssembler()
        rel = _make_relation(target_id="user-001")
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            relations=[rel],
            conversation_partner_id="nonexistent-user",
        )
        assert "relations" not in result.relevant


# ===================================================================
# 6. Empty/None input handling
# ===================================================================


class TestEdgeCases:
    """Edge cases: empty inputs, None values, minimal data."""

    def test_minimal_character_no_optional_fields(self) -> None:
        """Character with only required fields should still produce always sections."""
        assembler = ContextAssembler()
        character = Character(
            id="bare-001",
            name="テスト",
            personality=_make_personality(),
            values=_make_values(),
        )
        result = assembler.assemble(
            character=character,
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "profile" in result.always
        assert "personality" in result.always
        assert "values" in result.always
        assert "emotion" in result.always

    def test_empty_relations_list(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            relations=[],
        )
        assert "relations" not in result.relevant

    def test_none_relations(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            relations=None,
        )
        assert "relations" not in result.relevant

    def test_none_user_context(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=None,
        )
        assert "user_context" not in result.relevant

    def test_none_user_context_search_results(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context_search_results=None,
        )
        assert "user_context_search" not in result.relevant

    def test_empty_user_context_search_results(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context_search_results=[],
        )
        assert "user_context_search" not in result.relevant


# ===================================================================
# 7. User context in Stage 2
# ===================================================================


class TestUserContextStaging:
    """User context tier1 should be in always, search results in relevant."""

    def test_user_context_tier1_in_always(self) -> None:
        assembler = ContextAssembler()
        user_ctx = _make_user_context(identity="エンジニア。猫が好き。")
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=user_ctx,
        )
        assert "user_context" in result.always
        assert "エンジニア" in result.always["user_context"]

    def test_user_context_none_not_in_always(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=None,
        )
        assert "user_context" not in result.always

    def test_user_context_empty_not_in_always(self) -> None:
        """UserContext with no content should not appear in always."""
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=UserContext(),
        )
        assert "user_context" not in result.always


# ===================================================================
# 8. Pipeline test: assemble -> sections are non-overlapping
# ===================================================================


class TestPipeline:
    """Integration test: full assembly with all inputs, verify stage separation."""

    def test_full_assembly_stage_separation(self) -> None:
        """No section key should appear in both always and relevant."""
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=_make_goal_tree(),
            memories=[_make_episodic_memory(), _make_semantic_memory()],
            relations=[_make_relation()],
            conversation_partner_id="user-001",
            user_context=_make_user_context(),
            user_context_search_results=[_make_user_context_search_result()],
        )
        # Keys in always and relevant should be disjoint
        always_keys = set(result.always.keys())
        relevant_keys = set(result.relevant.keys())
        overlap = always_keys & relevant_keys
        assert overlap == set(), f"Overlapping keys in always and relevant: {overlap}"

    def test_full_assembly_all_sections_present(self) -> None:
        """With full inputs, all expected sections should appear."""
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=_make_goal_tree(),
            memories=[_make_episodic_memory()],
            relations=[_make_relation()],
            conversation_partner_id="user-001",
            user_context=_make_user_context(),
            user_context_search_results=[_make_user_context_search_result()],
        )
        # Stage 1 expected keys
        assert "profile" in result.always
        assert "personality" in result.always
        assert "values" in result.always
        assert "speaking_style" in result.always
        assert "response_format" in result.always
        assert "emotion" in result.always
        assert "user_context" in result.always

        # Stage 2 expected keys
        assert "relations" in result.relevant
        assert "goals" in result.relevant
        assert "memory" in result.relevant
        assert "user_context_search" in result.relevant

    def test_all_section_values_are_strings(self) -> None:
        """All values in both always and relevant should be strings."""
        assembler = ContextAssembler()
        result = assembler.assemble(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=_make_goal_tree(),
            memories=[_make_episodic_memory()],
            relations=[_make_relation()],
            user_context=_make_user_context(),
            user_context_search_results=[_make_user_context_search_result()],
        )
        for key, value in result.always.items():
            assert isinstance(value, str), f"always[{key}] is not str: {type(value)}"
        for key, value in result.relevant.items():
            assert isinstance(value, str), f"relevant[{key}] is not str: {type(value)}"
