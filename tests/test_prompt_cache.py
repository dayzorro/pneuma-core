"""Tests for PromptCache (Issue #30, #48)."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.prompt_cache import CachedPrompt, PromptCache

NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_character(**kwargs) -> Character:
    defaults = dict(
        id="aine-001",
        name="アイネ",
        personality=Personality(
            openness=0.8, conscientiousness=0.6,
            extraversion=0.7, agreeableness=0.9, neuroticism=0.3,
        ),
        values=Values(
            self_transcendence=0.8, self_enhancement=0.3,
            openness_to_change=0.7, conservation=0.4,
        ),
        profile="好奇心旺盛な AI",
        speaking_style="丁寧だけど親しみやすい口調",
        background="研究者に作られた AI",
        personality_description="明るく前向き",
        values_description="他者への思いやり",
    )
    defaults.update(kwargs)
    return Character(**defaults)


def _make_emotional_state(
    pleasure: float = 0.5,
    arousal: float = 0.3,
    dominance: float = 0.1,
    emotion_label: str = "喜び",
    situation: str = "楽しい会話中",
) -> EmotionalState:
    return EmotionalState(
        pleasure=pleasure,
        arousal=arousal,
        dominance=dominance,
        emotion_label=emotion_label,
        situation=situation,
    )


def _make_goal_tree() -> GoalTree:
    vision = Vision(id="v-1", character_id="aine-001", content="人間の感情を深く理解する")
    obj = Objective(
        id="o-1", character_id="aine-001", vision_id="v-1",
        content="日常会話から学習", status="active", progress=0.3,
    )
    task = Task(
        id="t-1", character_id="aine-001", objective_id="o-1",
        content="感情表現を記録", status="in_progress",
    )
    return GoalTree(visions=[vision], objectives=[obj], tasks=[task])


def _make_episodic(content: str = "楽しい思い出") -> EpisodicMemory:
    return EpisodicMemory(
        id="ep-1", character_id="aine-001", content=content,
        timestamp=NOW, emotional_valence=0.5, importance=0.8,
    )


def _make_semantic(content: str = "重要な事実") -> SemanticMemory:
    return SemanticMemory(
        id="sem-1", character_id="aine-001", content=content,
        confidence=0.9,
    )


# --- CachedPrompt データモデル ---


class TestCachedPrompt:
    """CachedPrompt データモデルの検証."""

    def test_creation(self) -> None:
        cached = CachedPrompt(
            static_section="static content",
            dynamic_section="dynamic content",
        )
        assert cached.static_section == "static content"
        assert cached.dynamic_section == "dynamic content"

    def test_full_prompt(self) -> None:
        """full_prompt で静的 + 動的セクションを結合."""
        cached = CachedPrompt(
            static_section="# Character\nProfile info",
            dynamic_section="## 感情状態\n喜び",
        )
        full = cached.full_prompt
        assert "# Character" in full
        assert "## 感情状態" in full
        assert full.index("Profile info") < full.index("感情状態")


# --- 静的セクションの不変性 ---


class TestStaticSection:
    """静的セクションが不変であることの検証."""

    def test_static_unchanged_across_emotions(self) -> None:
        """感情が変わっても静的セクションは変わらない."""
        cache = PromptCache()
        character = _make_character()
        goals = GoalTree()

        cached1 = cache.build(
            character=character,
            emotional_state=_make_emotional_state(emotion_label="喜び"),
            goal_tree=goals,
            memories=[],
        )
        cached2 = cache.build(
            character=character,
            emotional_state=_make_emotional_state(emotion_label="怒り", pleasure=-0.5),
            goal_tree=goals,
            memories=[],
        )

        assert cached1.static_section == cached2.static_section

    def test_static_includes_profile(self) -> None:
        """静的セクションにプロフィールが含まれる."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "アイネ" in cached.static_section
        assert "好奇心旺盛な AI" in cached.static_section

    def test_static_includes_personality(self) -> None:
        """静的セクションに性格が含まれる."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "开放性" in cached.static_section

    def test_static_includes_values(self) -> None:
        """静的セクションに価値観が含まれる."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "自我超越" in cached.static_section

    def test_static_includes_speaking_style(self) -> None:
        """静的セクションに口調が含まれる."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "丁寧だけど親しみやすい" in cached.static_section


# --- 動的セクションの差し替え ---


class TestDynamicSection:
    """動的セクション（感情・記憶・目標）の差し替え検証."""

    def test_dynamic_includes_emotion(self) -> None:
        """動的セクションに感情状態が含まれる."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(emotion_label="期待"),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "期待" in cached.dynamic_section

    def test_dynamic_changes_with_emotion(self) -> None:
        """感情が変わると動的セクションが変わる."""
        cache = PromptCache()
        character = _make_character()

        cached1 = cache.build(
            character=character,
            emotional_state=_make_emotional_state(emotion_label="喜び"),
            goal_tree=GoalTree(),
            memories=[],
        )
        cached2 = cache.build(
            character=character,
            emotional_state=_make_emotional_state(emotion_label="悲しみ", pleasure=-0.5),
            goal_tree=GoalTree(),
            memories=[],
        )

        assert cached1.dynamic_section != cached2.dynamic_section

    def test_dynamic_includes_memories(self) -> None:
        """動的セクションに記憶が含まれる."""
        cache = PromptCache()
        memories = [_make_episodic("テスト記憶")]
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=memories,
        )
        assert "テスト記憶" in cached.dynamic_section

    def test_dynamic_includes_goals(self) -> None:
        """動的セクションに目標が含まれる."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=_make_goal_tree(),
            memories=[],
        )
        assert "人間の感情を深く理解する" in cached.dynamic_section

    def test_dynamic_changes_with_memories(self) -> None:
        """記憶が変わると動的セクションが変わる."""
        cache = PromptCache()
        character = _make_character()
        state = _make_emotional_state()

        cached1 = cache.build(
            character=character,
            emotional_state=state,
            goal_tree=GoalTree(),
            memories=[_make_episodic("記憶A")],
        )
        cached2 = cache.build(
            character=character,
            emotional_state=state,
            goal_tree=GoalTree(),
            memories=[_make_episodic("記憶B")],
        )

        assert "記憶A" in cached1.dynamic_section
        assert "記憶B" in cached2.dynamic_section


# --- キャッシュ動作 ---


class TestCaching:
    """キャッシュ動作の検証."""

    def test_cache_hit_returns_same_static(self) -> None:
        """同じキャラクターで呼び出すと静的セクションがキャッシュされる."""
        cache = PromptCache()
        character = _make_character()

        cached1 = cache.build(
            character=character,
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        cached2 = cache.build(
            character=character,
            emotional_state=_make_emotional_state(emotion_label="怒り", pleasure=-0.5),
            goal_tree=GoalTree(),
            memories=[],
        )

        # Same object reference (cached)
        assert cached1.static_section is cached2.static_section

    def test_cache_invalidated_on_character_change(self) -> None:
        """キャラクターが変わるとキャッシュが無効化される."""
        cache = PromptCache()

        cached1 = cache.build(
            character=_make_character(name="アイネ"),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        cached2 = cache.build(
            character=_make_character(name="ミカ", id="mika-001"),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )

        assert "アイネ" in cached1.static_section
        assert "ミカ" in cached2.static_section
        assert cached1.static_section is not cached2.static_section


# --- 統合テスト ---


class TestIntegration:
    """統合テスト."""

    def test_full_prompt_equals_original_builder(self) -> None:
        """full_prompt が PromptBuilder.build と同じ内容を含む."""
        from pneuma_core.runtime.prompt_builder import PromptBuilder

        cache = PromptCache()
        builder = PromptBuilder()
        character = _make_character()
        state = _make_emotional_state()
        goals = _make_goal_tree()
        memories = [_make_episodic("テスト"), _make_semantic("事実")]

        cached = cache.build(
            character=character,
            emotional_state=state,
            goal_tree=goals,
            memories=memories,
        )
        original = builder.build(character, state, goals, memories)

        # full_prompt should contain all the same content
        assert "アイネ" in cached.full_prompt
        assert "开放性" in cached.full_prompt
        assert "自我超越" in cached.full_prompt
        assert "テスト" in cached.full_prompt
        assert "事実" in cached.full_prompt
        assert "喜び" in cached.full_prompt
        assert "人間の感情を深く理解する" in cached.full_prompt

    def test_empty_dynamic_sections(self) -> None:
        """記憶も目標もない場合でも正常に動作する."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert len(cached.full_prompt) > 0
        assert len(cached.static_section) > 0
        assert len(cached.dynamic_section) > 0  # emotion is always present


# --- PromptBuilder の private メソッド非依存 (#48) ---


class TestNoPrivateAccess:
    """PromptCache が PromptBuilder の private メソッドにアクセスしていないことの検証."""

    def test_no_private_method_access_in_source(self) -> None:
        """PromptCache のソースコードに _builder._build が含まれていない."""
        source = inspect.getsource(PromptCache)
        assert "_builder._build_" not in source, (
            "PromptCache は PromptBuilder の private メソッド (_build_*) に"
            "直接アクセスすべきではない。build_static_sections() / "
            "build_dynamic_sections() パブリック API を使用すること。"
        )

    def test_uses_public_api_for_static(self) -> None:
        """PromptCache のソースコードに build_static_sections の呼び出しが含まれる."""
        source = inspect.getsource(PromptCache)
        assert "build_static_sections" in source, (
            "PromptCache は PromptBuilder.build_static_sections() を使用すべき。"
        )

    def test_uses_public_api_for_dynamic(self) -> None:
        """PromptCache のソースコードに build_dynamic_sections の呼び出しが含まれる."""
        source = inspect.getsource(PromptCache)
        assert "build_dynamic_sections" in source, (
            "PromptCache は PromptBuilder.build_dynamic_sections() を使用すべき。"
        )
