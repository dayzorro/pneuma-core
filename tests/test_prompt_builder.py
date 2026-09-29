"""Tests for PromptBuilder (Issue #27)."""

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
from pneuma_core.runtime.prompt_builder import PromptBuilder

NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_personality(
    openness: float = 0.8,
    conscientiousness: float = 0.6,
    extraversion: float = 0.7,
    agreeableness: float = 0.9,
    neuroticism: float = 0.3,
) -> Personality:
    return Personality(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
    )


def _make_values(
    self_transcendence: float = 0.8,
    self_enhancement: float = 0.3,
    openness_to_change: float = 0.7,
    conservation: float = 0.4,
) -> Values:
    return Values(
        self_transcendence=self_transcendence,
        self_enhancement=self_enhancement,
        openness_to_change=openness_to_change,
        conservation=conservation,
    )


def _make_character(**kwargs) -> Character:
    defaults = dict(
        id="aine-001",
        name="アイネ",
        personality=_make_personality(),
        values=_make_values(),
        profile="好奇心旺盛な AI キャラクター",
        appearance="銀髪のロングヘア、青い瞳",
        speaking_style="丁寧だけど親しみやすい口調。「〜だよ」「〜だね」を使う。",
        background="研究者に作られた AI。人間の感情を理解したいと思っている。",
        personality_description="明るく前向きで、新しいことが大好き。",
        values_description="他者への思いやりを大切にしている。",
    )
    defaults.update(kwargs)
    return Character(**defaults)


def _make_emotional_state(
    pleasure: float = 0.5,
    arousal: float = 0.3,
    dominance: float = 0.1,
    emotion_label: str = "喜び",
    situation: str = "ユーザーと楽しく会話中",
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
        id="o-1",
        character_id="aine-001",
        vision_id="v-1",
        content="日常会話から感情パターンを学習する",
        status="active",
        progress=0.3,
    )
    task = Task(
        id="t-1",
        character_id="aine-001",
        objective_id="o-1",
        content="ユーザーの感情表現を記録する",
        status="in_progress",
    )
    return GoalTree(visions=[vision], objectives=[obj], tasks=[task])


def _make_episodic(
    id: str = "ep-1",
    content: str = "ユーザーと初めて出会った",
    importance: float = 0.8,
) -> EpisodicMemory:
    return EpisodicMemory(
        id=id,
        character_id="aine-001",
        content=content,
        timestamp=NOW,
        emotional_valence=0.5,
        importance=importance,
    )


def _make_semantic(
    id: str = "sem-1",
    content: str = "ユーザーはプログラミングが好き",
    confidence: float = 0.9,
) -> SemanticMemory:
    return SemanticMemory(
        id=id,
        character_id="aine-001",
        content=content,
        confidence=confidence,
    )


# --- プロフィールセクション ---


class TestProfileSection:
    """キャラクター基本情報セクションの検証."""

    def test_includes_name(self) -> None:
        builder = PromptBuilder()
        character = _make_character()
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "アイネ" in prompt

    def test_includes_profile(self) -> None:
        builder = PromptBuilder()
        character = _make_character()
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "好奇心旺盛な AI キャラクター" in prompt

    def test_includes_appearance(self) -> None:
        builder = PromptBuilder()
        character = _make_character()
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "銀髪のロングヘア" in prompt

    def test_includes_background(self) -> None:
        builder = PromptBuilder()
        character = _make_character()
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "研究者に作られた AI" in prompt

    def test_none_fields_omitted(self) -> None:
        """None のフィールドはプロンプトに含まれない."""
        builder = PromptBuilder()
        character = _make_character(profile=None, appearance=None, background=None)
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "アイネ" in prompt
        # None フィールドが "None" という文字列として出力されない
        assert "None" not in prompt


# --- 性格セクション ---


class TestPersonalitySection:
    """性格 (Big Five) セクションの検証."""

    def test_high_openness_described(self) -> None:
        builder = PromptBuilder()
        character = _make_character(personality=_make_personality(openness=0.9))
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "开放性" in prompt

    def test_low_neuroticism_described(self) -> None:
        builder = PromptBuilder()
        character = _make_character(personality=_make_personality(neuroticism=0.2))
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "神经质" in prompt

    def test_personality_description_included(self) -> None:
        builder = PromptBuilder()
        character = _make_character(personality_description="明るく前向き")
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "明るく前向き" in prompt

    def test_no_personality_description(self) -> None:
        """personality_description が None でもエラーにならない."""
        builder = PromptBuilder()
        character = _make_character(personality_description=None)
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "性格" in prompt or "Personality" in prompt


# --- 価値観セクション ---


class TestValuesSection:
    """価値観 (Schwartz) セクションの検証."""

    def test_high_self_transcendence_described(self) -> None:
        builder = PromptBuilder()
        character = _make_character(values=_make_values(self_transcendence=0.9))
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "自我超越" in prompt

    def test_values_description_included(self) -> None:
        builder = PromptBuilder()
        character = _make_character(values_description="思いやり重視")
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "思いやり重視" in prompt

    def test_no_values_description(self) -> None:
        """values_description が None でもエラーにならない."""
        builder = PromptBuilder()
        character = _make_character(values_description=None)
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert len(prompt) > 0


# --- 記憶セクション ---


class TestMemorySection:
    """記憶セクションの検証."""

    def test_episodic_memory_included(self) -> None:
        builder = PromptBuilder()
        memories = [_make_episodic(content="初めての出会い")]
        prompt = builder.build(_make_character(), _make_emotional_state(), GoalTree(), memories)
        assert "初めての出会い" in prompt

    def test_semantic_memory_included(self) -> None:
        builder = PromptBuilder()
        memories = [_make_semantic(content="ユーザーは猫が好き")]
        prompt = builder.build(_make_character(), _make_emotional_state(), GoalTree(), memories)
        assert "ユーザーは猫が好き" in prompt

    def test_mixed_memories(self) -> None:
        """エピソードと意味記憶の混合."""
        builder = PromptBuilder()
        memories = [
            _make_episodic(id="ep-1", content="一緒に散歩した"),
            _make_semantic(id="sem-1", content="ユーザーは朝型"),
        ]
        prompt = builder.build(_make_character(), _make_emotional_state(), GoalTree(), memories)
        assert "一緒に散歩した" in prompt
        assert "ユーザーは朝型" in prompt

    def test_empty_memories(self) -> None:
        """記憶なしでもエラーにならない."""
        builder = PromptBuilder()
        prompt = builder.build(_make_character(), _make_emotional_state(), GoalTree(), [])
        assert len(prompt) > 0


# --- 目標セクション ---


class TestGoalsSection:
    """目標セクションの検証."""

    def test_vision_included(self) -> None:
        builder = PromptBuilder()
        goal_tree = _make_goal_tree()
        prompt = builder.build(_make_character(), _make_emotional_state(), goal_tree, [])
        assert "人間の感情を深く理解する" in prompt

    def test_objective_included(self) -> None:
        builder = PromptBuilder()
        goal_tree = _make_goal_tree()
        prompt = builder.build(_make_character(), _make_emotional_state(), goal_tree, [])
        assert "日常会話から感情パターンを学習する" in prompt

    def test_task_included(self) -> None:
        builder = PromptBuilder()
        goal_tree = _make_goal_tree()
        prompt = builder.build(_make_character(), _make_emotional_state(), goal_tree, [])
        assert "ユーザーの感情表現を記録する" in prompt

    def test_empty_goal_tree(self) -> None:
        """目標なしでもエラーにならない."""
        builder = PromptBuilder()
        prompt = builder.build(_make_character(), _make_emotional_state(), GoalTree(), [])
        assert len(prompt) > 0


# --- 感情状態セクション ---


class TestStateSection:
    """感情状態セクションの検証."""

    def test_emotion_label_included(self) -> None:
        builder = PromptBuilder()
        state = _make_emotional_state(emotion_label="期待")
        prompt = builder.build(_make_character(), state, GoalTree(), [])
        assert "期待" in prompt

    def test_situation_included(self) -> None:
        builder = PromptBuilder()
        state = _make_emotional_state(situation="新しいプロジェクトを始めた")
        prompt = builder.build(_make_character(), state, GoalTree(), [])
        assert "新しいプロジェクトを始めた" in prompt


# --- 口調セクション ---


class TestSpeakingStyleSection:
    """口調セクションの検証."""

    def test_speaking_style_included(self) -> None:
        builder = PromptBuilder()
        character = _make_character(speaking_style="丁寧語を使う")
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert "丁寧語を使う" in prompt

    def test_no_speaking_style(self) -> None:
        """speaking_style が None でもエラーにならない."""
        builder = PromptBuilder()
        character = _make_character(speaking_style=None)
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        assert len(prompt) > 0


# --- 統合テスト ---


class TestIntegration:
    """統合テスト."""

    def test_full_prompt_contains_all_sections(self) -> None:
        """全5コンテキストがプロンプトに含まれる."""
        builder = PromptBuilder()
        character = _make_character()
        state = _make_emotional_state()
        goals = _make_goal_tree()
        memories = [
            _make_episodic(content="楽しい思い出"),
            _make_semantic(content="重要な事実"),
        ]

        prompt = builder.build(character, state, goals, memories)

        # 基本情報
        assert "アイネ" in prompt
        # 性格
        assert "开放性" in prompt
        # 価値観
        assert "自我超越" in prompt
        # 記憶
        assert "楽しい思い出" in prompt
        assert "重要な事実" in prompt
        # 目標
        assert "人間の感情を深く理解する" in prompt
        # 感情状態
        assert "喜び" in prompt
        # 口調
        assert "丁寧だけど親しみやすい" in prompt

    def test_section_order(self) -> None:
        """セクションが正しい順序で出力される."""
        builder = PromptBuilder()
        character = _make_character()
        state = _make_emotional_state()
        goals = _make_goal_tree()
        memories = [_make_episodic(content="テスト記憶")]

        prompt = builder.build(character, state, goals, memories)

        # プロフィール → 性格 → 価値観 → 記憶 → 目標 → 感情 → 口調の順
        profile_pos = prompt.find("アイネ")
        personality_pos = prompt.find("开放性")
        memory_pos = prompt.find("テスト記憶")
        goal_pos = prompt.find("人間の感情を深く理解する")
        state_pos = prompt.find("喜び")
        style_pos = prompt.find("丁寧だけど親しみやすい")

        assert profile_pos < personality_pos
        assert personality_pos < memory_pos
        assert memory_pos < goal_pos
        assert goal_pos < state_pos
        assert state_pos < style_pos

    def test_minimal_character(self) -> None:
        """最小構成でもプロンプト生成可能."""
        builder = PromptBuilder()
        character = Character(
            id="min-001",
            name="ミニマル",
            personality=Personality(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            ),
            values=Values(
                self_transcendence=0.5,
                self_enhancement=0.5,
                openness_to_change=0.5,
                conservation=0.5,
            ),
        )
        state = _make_emotional_state()
        prompt = builder.build(character, state, GoalTree(), [])
        assert "ミニマル" in prompt
        assert len(prompt) > 50

    def test_prompt_is_string(self) -> None:
        """build() は str を返す."""
        builder = PromptBuilder()
        result = builder.build(
            _make_character(), _make_emotional_state(), GoalTree(), []
        )
        assert isinstance(result, str)


# --- パブリック API: build_static_sections / build_dynamic_sections (#48) ---


class TestBuildStaticSections:
    """build_static_sections パブリック API の検証."""

    def test_method_exists(self) -> None:
        """build_static_sections メソッドが存在する."""
        builder = PromptBuilder()
        assert hasattr(builder, "build_static_sections")
        assert callable(builder.build_static_sections)

    def test_returns_string(self) -> None:
        """build_static_sections は str を返す."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert isinstance(result, str)

    def test_includes_profile(self) -> None:
        """静的セクションにプロフィールが含まれる."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert "アイネ" in result
        assert "好奇心旺盛な AI キャラクター" in result

    def test_includes_personality(self) -> None:
        """静的セクションに性格が含まれる."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert "开放性" in result

    def test_includes_values(self) -> None:
        """静的セクションに価値観が含まれる."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert "自我超越" in result

    def test_includes_speaking_style(self) -> None:
        """静的セクションに口調が含まれる."""
        builder = PromptBuilder()
        result = builder.build_static_sections(
            _make_character(speaking_style="丁寧語を使う")
        )
        assert "丁寧語を使う" in result

    def test_no_dynamic_content(self) -> None:
        """静的セクションに動的コンテンツ（感情・記憶・目標）が含まれない."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert "現在の感情状態" not in result
        assert "記憶" not in result
        assert "目標" not in result


class TestBuildDynamicSections:
    """build_dynamic_sections パブリック API の検証."""

    def test_method_exists(self) -> None:
        """build_dynamic_sections メソッドが存在する."""
        builder = PromptBuilder()
        assert hasattr(builder, "build_dynamic_sections")
        assert callable(builder.build_dynamic_sections)

    def test_returns_string(self) -> None:
        """build_dynamic_sections は str を返す."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert isinstance(result, str)

    def test_includes_emotion(self) -> None:
        """動的セクションに感情状態が含まれる."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(emotion_label="期待"),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "期待" in result

    def test_includes_memories(self) -> None:
        """動的セクションに記憶が含まれる."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[_make_episodic(content="テスト記憶")],
        )
        assert "テスト記憶" in result

    def test_includes_goals(self) -> None:
        """動的セクションに目標が含まれる."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=_make_goal_tree(),
            memories=[],
        )
        assert "人間の感情を深く理解する" in result

    def test_no_static_content(self) -> None:
        """動的セクションに静的コンテンツ（プロフィール・性格・価値観・口調）が含まれない."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "性格" not in result
        assert "価値観" not in result
        assert "口調" not in result


# --- 応答フォーマットセクション (#86) ---


class TestResponseFormatSection:
    """応答フォーマットセクション（構造化出力指示）の検証 (#86)."""

    def test_build_response_format_section_returns_content(self) -> None:
        """_build_response_format_section() が内容を返す."""
        builder = PromptBuilder()
        result = builder._build_response_format_section()
        assert len(result) > 0
        assert "回复格式" in result

    def test_response_format_contains_json_instruction(self) -> None:
        """応答フォーマットセクションに JSON 形式指示が含まれる."""
        builder = PromptBuilder()
        result = builder._build_response_format_section()
        assert "speech" in result
        assert "thought" in result
        assert "action" in result
        assert "JSON" in result

    def test_response_format_is_last_section_in_build(self) -> None:
        """応答フォーマットセクションが build() 出力の最後のセクションである."""
        builder = PromptBuilder()
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
        )
        # 応答フォーマットは最後のセクション
        assert prompt.rstrip().endswith("不要输出 JSON 以外的任何文本")

    def test_response_format_in_build_static_sections(self) -> None:
        """応答フォーマットセクションが build_static_sections() に含まれる."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert "回复格式" in result
        assert "speech" in result

    def test_response_format_not_in_build_dynamic_sections(self) -> None:
        """応答フォーマットセクションが build_dynamic_sections() に含まれない."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert "応答フォーマット" not in result

    def test_response_format_after_speaking_style_in_build(self) -> None:
        """応答フォーマットが口調セクションの後に来る."""
        builder = PromptBuilder()
        prompt = builder.build(
            _make_character(speaking_style="丁寧語を使う"),
            _make_emotional_state(),
            GoalTree(),
            [],
        )
        style_pos = prompt.find("丁寧語を使う")
        format_pos = prompt.find("回复格式")
        assert style_pos < format_pos


# --- 自然言語変換テスト (#120) ---


import re


def _has_raw_numeric(text: str) -> bool:
    """テキスト内に生の数値パターンが含まれるか検出する.

    検出対象:
    - PAD形式: (+0.5), (-0.3) など符号付き小数
    - Big Five / Schwartz 形式: (0.8), (0.3) など括弧内小数
    - "高い (0.8)" "低い (0.2)" のようなラベル+数値パターン
    ただし、JSON フォーマットセクション内の例示は除外する。
    """
    # 応答フォーマットセクションを除外してチェック
    format_section_start = text.find("## 応答フォーマット")
    # 目標セクションの進捗率（30%）は除外（これは自然言語として許容）
    check_text = text[:format_section_start] if format_section_start >= 0 else text

    # PAD 数値: +0.5, -0.3 のような符号付き小数
    if re.search(r'[+-]\d+\.\d+', check_text):
        return True
    # 括弧内の小数: (0.8), (0.3) など
    if re.search(r'\(\d+\.\d+\)', check_text):
        return True
    return False


class TestNaturalLanguageConversion:
    """数値→自然言語変換の検証 (#120)."""

    # --- PAD 感情状態の自然言語変換 ---

    def test_pad_no_raw_numbers_in_state_section(self) -> None:
        """感情状態セクションに生の PAD 数値が含まれない."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.5, arousal=0.3, dominance=0.1)
        prompt = builder.build(_make_character(), state, GoalTree(), [])
        # "PAD: 快感(+0.5) 覚醒(+0.3) 支配(+0.1)" のような生数値がないこと
        state_section_start = prompt.find("## 現在の感情状態")
        state_section = prompt[state_section_start:]
        assert not re.search(r'[+-]\d+\.\d+', state_section)

    def test_pad_strong_positive_pleasure(self) -> None:
        """強い正の快感値が自然な表現に変換される."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.8, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert "非常" in section or "强烈" in section
        assert not re.search(r'[+-]\d+\.\d+', section)

    def test_pad_slight_positive_pleasure(self) -> None:
        """わずかな正の快感値が自然な表現に変換される."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.2, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert "略微" in section or "稍" in section
        assert not re.search(r'[+-]\d+\.\d+', section)

    def test_pad_strong_negative_pleasure(self) -> None:
        """強い負の快感値が自然な表現に変換される."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=-0.8, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert "非常" in section or "强烈" in section
        assert not re.search(r'[+-]\d+\.\d+', section)

    def test_pad_neutral_values(self) -> None:
        """PAD が全て中立（0.0）の場合、穏やかな状態として表現される."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.0, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert not re.search(r'[+-]\d+\.\d+', section)

    def test_pad_all_dimensions_described(self) -> None:
        """PAD の3次元すべてが自然言語で記述される."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.5, arousal=-0.3, dominance=0.4)
        section = builder._build_state_section(state)
        # 感情ラベルと状況は残る
        assert state.emotion_label in section
        assert state.situation in section
        assert not re.search(r'[+-]\d+\.\d+', section)

    # --- Big Five 性格の自然言語変換 ---

    def test_bigfive_no_raw_numbers(self) -> None:
        """性格セクションに生の Big Five 数値が含まれない."""
        builder = PromptBuilder()
        character = _make_character(
            personality=_make_personality(
                openness=0.9,
                conscientiousness=0.5,
                extraversion=0.7,
                agreeableness=0.2,
                neuroticism=0.4,
            )
        )
        prompt = builder.build(character, _make_emotional_state(), GoalTree(), [])
        personality_start = prompt.find("## 性格")
        values_start = prompt.find("## 価値観")
        personality_section = prompt[personality_start:values_start]
        assert not re.search(r'\(\d+\.\d+\)', personality_section)

    def test_bigfive_high_trait_description_only(self) -> None:
        """高い特性は説明文のみで数値なし."""
        builder = PromptBuilder()
        character = _make_character(
            personality=_make_personality(openness=0.9)
        )
        section = builder._build_personality_section(character)
        assert "开放性" in section
        # 説明文がある
        assert "对新经验" in section or "开放" in section
        # 数値がない
        assert "(0.9)" not in section
        assert "0.9" not in section

    def test_bigfive_low_trait_description_only(self) -> None:
        """低い特性は説明文のみで数値なし."""
        builder = PromptBuilder()
        character = _make_character(
            personality=_make_personality(neuroticism=0.2)
        )
        section = builder._build_personality_section(character)
        assert "神经质" in section
        # 説明文がある
        assert "稳定" in section or "冷静" in section
        # 数値がない
        assert "(0.2)" not in section
        assert "0.2" not in section

    def test_bigfive_mid_trait_has_description(self) -> None:
        """中程度の特性にも説明文がある（_TRAIT_MID_DESC）."""
        builder = PromptBuilder()
        character = _make_character(
            personality=_make_personality(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            )
        )
        section = builder._build_personality_section(character)
        # 各特性に説明文がある（数値だけの行がない）
        for trait_label in ["开放性", "尽责性", "外向性", "宜人性", "神经质"]:
            line = [l for l in section.split("\n") if trait_label in l]
            assert len(line) == 1
            # その行に数値が含まれない
            assert not re.search(r'\(\d+\.\d+\)', line[0])
            # 説明文（"—" の後）が存在する
            assert "—" in line[0] or ":" in line[0]

    def test_bigfive_no_numeric_labels(self) -> None:
        """性格セクションに「高い (0.8)」「低い (0.2)」のようなラベル+数値がない."""
        builder = PromptBuilder()
        character = _make_character()
        section = builder._build_personality_section(character)
        assert not re.search(r'高い\s*\(\d', section)
        assert not re.search(r'低い\s*\(\d', section)
        assert not re.search(r'中程度\s*\(\d', section)

    # --- Schwartz Values の自然言語変換 ---

    def test_values_no_raw_numbers(self) -> None:
        """価値観セクションに生の Schwartz 数値が含まれない."""
        builder = PromptBuilder()
        character = _make_character(
            values=_make_values(
                self_transcendence=0.8,
                self_enhancement=0.3,
                openness_to_change=0.7,
                conservation=0.4,
            )
        )
        section = builder._build_values_section(character)
        assert not re.search(r'\(\d+\.\d+\)', section)

    def test_values_important_has_description(self) -> None:
        """重視する価値観に説明文がある."""
        builder = PromptBuilder()
        character = _make_character(
            values=_make_values(self_transcendence=0.9)
        )
        section = builder._build_values_section(character)
        assert "自我超越" in section
        assert "他人" in section or "幸福" in section or "善" in section
        assert "(0.9)" not in section

    def test_values_unimportant_also_has_description(self) -> None:
        """重視しない価値観にも説明文がある."""
        builder = PromptBuilder()
        character = _make_character(
            values=_make_values(self_enhancement=0.2)
        )
        section = builder._build_values_section(character)
        # self_enhancement の行に説明がある
        se_lines = [l for l in section.split("\n") if "自我增强" in l]
        assert len(se_lines) == 1
        # 説明文（"—" の後）がある
        assert "—" in se_lines[0]
        # 数値がない
        assert "(0.2)" not in se_lines[0]

    def test_values_no_numeric_labels(self) -> None:
        """価値観セクションに「重視 (0.8)」のようなラベル+数値がない."""
        builder = PromptBuilder()
        character = _make_character()
        section = builder._build_values_section(character)
        assert not re.search(r'重視\s*\(\d', section)
        assert not re.search(r'\(\d+\.\d+\)', section)

    # --- 統合テスト: プロンプト全体に生の数値が含まれない ---

    def test_full_prompt_no_raw_numbers(self) -> None:
        """プロンプト全体（応答フォーマット除く）に生の数値が含まれない."""
        builder = PromptBuilder()
        character = _make_character(
            personality=_make_personality(
                openness=0.9,
                conscientiousness=0.5,
                extraversion=0.2,
                agreeableness=0.8,
                neuroticism=0.4,
            ),
            values=_make_values(
                self_transcendence=0.9,
                self_enhancement=0.2,
                openness_to_change=0.6,
                conservation=0.3,
            ),
        )
        state = _make_emotional_state(pleasure=0.7, arousal=-0.4, dominance=0.2)
        prompt = builder.build(character, state, _make_goal_tree(), [])
        assert not _has_raw_numeric(prompt)

    def test_build_static_sections_no_raw_numbers(self) -> None:
        """build_static_sections の出力にも生の数値が含まれない."""
        builder = PromptBuilder()
        character = _make_character()
        result = builder.build_static_sections(character)
        assert not _has_raw_numeric(result)

    def test_build_dynamic_sections_no_raw_numbers(self) -> None:
        """build_dynamic_sections の出力にも生の数値が含まれない."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.6, arousal=-0.5, dominance=0.3)
        result = builder.build_dynamic_sections(state, GoalTree(), [])
        assert not _has_raw_numeric(result)

    # --- PAD 4段階強度レベルの検証 ---

    def test_pad_intensity_very_strong(self) -> None:
        """PAD |値| >= 0.7 は「とても」レベル."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.8, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert "非常" in section

    def test_pad_intensity_moderate(self) -> None:
        """PAD |値| 0.4-0.69 は「やや」レベル."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.5, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert "有些" in section

    def test_pad_intensity_slight(self) -> None:
        """PAD |値| 0.1-0.39 は「わずかに」レベル."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.2, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        assert "略微" in section

    def test_pad_intensity_neutral(self) -> None:
        """PAD |値| < 0.1 は中立（強度表現なし）."""
        builder = PromptBuilder()
        state = _make_emotional_state(pleasure=0.05, arousal=0.0, dominance=0.0)
        section = builder._build_state_section(state)
        # 中立の場合、快・不快に関する記述がないか、中立であることが示される
        assert "とても" not in section or "中立" in section


# --- 関係性セクション (#116): description テキスト主体化 ---


def _make_relation(
    target_name: str = "きょうへい",
    relationship_type: str = "partner",
    description: str = "何でも話せるけどたまに遠慮する。一緒にいるのが自然。",
    closeness: float = 0.9,
    trust: float = 0.8,
) -> Relation:
    return Relation(
        id="rel-test",
        owner_id="mira",
        target_id="user",
        target_name=target_name,
        relationship_type=relationship_type,
        description=description,
        closeness=closeness,
        trust=trust,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestRelationsSection:
    """関係性セクションの検証 (#116): description テキスト主体化."""

    def test_description_is_primary_content(self) -> None:
        """description がプロンプトのプライマリコンテンツとして含まれる."""
        builder = PromptBuilder()
        relations = [_make_relation(description="何でも話せるけどたまに遠慮する。一緒にいるのが自然。")]
        result = builder._build_relations_section(relations)
        assert "何でも話せるけどたまに遠慮する。一緒にいるのが自然。" in result

    def test_closeness_numbers_not_in_output(self) -> None:
        """closeness/trust の数値がプロンプトに含まれない."""
        builder = PromptBuilder()
        relations = [_make_relation(closeness=0.9, trust=0.8)]
        result = builder._build_relations_section(relations)
        assert "0.9" not in result
        assert "0.8" not in result

    def test_closeness_labels_not_in_output(self) -> None:
        """closeness ラベル（とても親しい/親しい/知り合い）がプロンプトに含まれない."""
        builder = PromptBuilder()
        relations = [_make_relation(closeness=0.9)]
        result = builder._build_relations_section(relations)
        assert "とても親しい" not in result
        assert "知り合い" not in result

    def test_relationship_type_appears(self) -> None:
        """relationship_type ラベルがプロンプトに含まれる."""
        builder = PromptBuilder()
        relations = [_make_relation(relationship_type="partner")]
        result = builder._build_relations_section(relations)
        assert "partner" in result

    def test_target_name_appears(self) -> None:
        """target_name がプロンプトに含まれる."""
        builder = PromptBuilder()
        relations = [_make_relation(target_name="きょうへい")]
        result = builder._build_relations_section(relations)
        assert "きょうへい" in result

    def test_empty_description_handled(self) -> None:
        """description が空文字列でもエラーにならない."""
        builder = PromptBuilder()
        relations = [_make_relation(description="")]
        result = builder._build_relations_section(relations)
        assert "きょうへい" in result
        assert "partner" in result

    def test_multiple_relations_formatted(self) -> None:
        """複数の関係性が正しくフォーマットされる."""
        builder = PromptBuilder()
        relations = [
            _make_relation(
                target_name="きょうへい",
                relationship_type="partner",
                description="何でも話せる。",
            ),
            _make_relation(
                target_name="あかね",
                relationship_type="friend",
                description="よく一緒に遊ぶ。",
            ),
        ]
        result = builder._build_relations_section(relations)
        assert "きょうへい" in result
        assert "あかね" in result
        assert "何でも話せる。" in result
        assert "よく一緒に遊ぶ。" in result

    def test_section_header_is_relations(self) -> None:
        """セクションヘッダーが '## 関係性' である."""
        builder = PromptBuilder()
        relations = [_make_relation()]
        result = builder._build_relations_section(relations)
        assert "## 关系" in result

    def test_none_returns_empty(self) -> None:
        """None が渡された場合は空文字列を返す."""
        builder = PromptBuilder()
        result = builder._build_relations_section(None)
        assert result == ""

    def test_empty_list_returns_empty(self) -> None:
        """空リストが渡された場合は空文字列を返す."""
        builder = PromptBuilder()
        result = builder._build_relations_section([])
        assert result == ""

    def test_output_format_matches_expected(self) -> None:
        """出力フォーマットが期待通りである."""
        builder = PromptBuilder()
        relations = [
            _make_relation(
                target_name="きょうへい",
                relationship_type="partner",
                description="何でも話せるけどたまに遠慮する。一緒にいるのが自然。",
            ),
        ]
        result = builder._build_relations_section(relations)
        expected_line = "- きょうへい（partner）: 何でも話せるけどたまに遠慮する。一緒にいるのが自然。"
        assert expected_line in result


# --- タスクプロンプト注入 (#124) ---


class TestTasksSection:
    """タスクセクションの検証 (#124)."""

    def test_user_tasks_only(self) -> None:
        """ユーザータスクのみ."""
        builder = PromptBuilder()
        tasks = [{"title": "レポートを書く", "priority": 5, "tags": []}]
        result = builder._build_tasks_section(user_tasks=tasks)
        assert "## 当前任务" in result
        assert "### 用户的任务" in result
        assert "レポートを書く" in result

    def test_character_tasks_only(self) -> None:
        """キャラクタータスクのみ."""
        builder = PromptBuilder()
        tasks = [{"title": "体調を気にかける", "priority": 1, "tags": []}]
        result = builder._build_tasks_section(character_tasks=tasks)
        assert "### 角色的任务" in result
        assert "体調を気にかける" in result

    def test_both_tasks(self) -> None:
        """ユーザーとキャラクター両方のタスク."""
        builder = PromptBuilder()
        result = builder._build_tasks_section(
            user_tasks=[{"title": "買い物", "priority": 5, "tags": []}],
            character_tasks=[{"title": "励ます", "priority": 1, "tags": []}],
        )
        assert "### 用户的任务" in result
        assert "### 角色的任务" in result

    def test_empty_returns_empty(self) -> None:
        """タスクなしで空文字列."""
        builder = PromptBuilder()
        assert builder._build_tasks_section() == ""
        assert builder._build_tasks_section(user_tasks=[], character_tasks=[]) == ""

    def test_priority_labels(self) -> None:
        """優先度でラベルが正しく表示."""
        builder = PromptBuilder()
        tasks = [
            {"title": "重要タスク", "priority": 5, "tags": []},
            {"title": "普通タスク", "priority": 1, "tags": []},
            {"title": "ラベルなし", "priority": 0, "tags": []},
        ]
        result = builder._build_tasks_section(user_tasks=tasks)
        assert "[重要] 重要タスク" in result
        assert "[普通] 普通タスク" in result
        assert "- ラベルなし" in result

    def test_tasks_in_dynamic_sections(self) -> None:
        """タスクは動的セクションに含まれる."""
        builder = PromptBuilder()
        tasks = [{"title": "テスト", "priority": 5, "tags": []}]
        result = builder.build_dynamic_sections(
            _make_emotional_state(), GoalTree(), [],
            user_tasks=tasks,
        )
        assert "当前任务" in result

    def test_tasks_not_in_static_sections(self) -> None:
        """タスクは静的セクションに含まれない."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert "現在のタスク" not in result


# --- RAG Tier 3 文字数上限 (#128) ---

from pneuma_core.runtime.prompt_builder import UserContextConfig
from pneuma_core.runtime.user_context import UserContext, UserContextChunk
from pneuma_core.runtime.user_context_search import UserContextSearchResult


class TestTier3CharLimit:
    """RAG Tier 3 検索結果の文字数上限 (#128)."""

    def _make_search_result(self, content: str, score: float = 0.9) -> UserContextSearchResult:
        chunk = UserContextChunk(content=content, layer=6, source_file="test.md")
        return UserContextSearchResult(chunk=chunk, score=score)

    def test_results_within_limit_all_included(self) -> None:
        """上限以内の結果はすべて含まれる."""
        builder = PromptBuilder()
        results = [
            self._make_search_result("短い結果A"),
            self._make_search_result("短い結果B"),
        ]
        output = builder._build_user_context_tier3_section(results)
        assert "短い結果A" in output
        assert "短い結果B" in output

    def test_results_exceeding_limit_are_truncated(self) -> None:
        """上限を超える結果は切り捨てられる."""
        builder = PromptBuilder()
        # 1つ 800文字の結果を3つ → 合計2400文字 > デフォルト2000文字
        long_content = "あ" * 800
        results = [
            self._make_search_result(long_content + "A", score=0.9),
            self._make_search_result(long_content + "B", score=0.8),
            self._make_search_result(long_content + "C", score=0.7),
        ]
        output = builder._build_user_context_tier3_section(results)
        # 最初の2つは入る(801+801=1602 < 2000)が3つ目は超過(2403 > 2000)
        assert long_content + "A" in output
        assert long_content + "B" in output
        assert long_content + "C" not in output

    def test_custom_config_limit(self) -> None:
        """カスタム上限値が適用される."""
        builder = PromptBuilder()
        config = UserContextConfig(tier3_max_chars=500)
        results = [
            self._make_search_result("あ" * 300, score=0.9),
            self._make_search_result("い" * 300, score=0.8),
        ]
        output = builder._build_user_context_tier3_section(results, config=config)
        # 最初の300文字は入るが、2つ目を加えると600文字で500超過
        assert "あ" * 300 in output
        assert "い" * 300 not in output

    def test_single_large_result_included(self) -> None:
        """1つ目の結果が上限を超えていてもその1つは含まれる."""
        builder = PromptBuilder()
        config = UserContextConfig(tier3_max_chars=100)
        results = [self._make_search_result("あ" * 500)]
        output = builder._build_user_context_tier3_section(results, config=config)
        # 少なくとも1つ目は含まれる（完全に空にはしない）
        assert "あ" * 500 in output

    def test_empty_results_returns_empty(self) -> None:
        """空の結果は空文字列."""
        builder = PromptBuilder()
        assert builder._build_user_context_tier3_section([]) == ""
        assert builder._build_user_context_tier3_section(None) == ""


# --- 日記サマリー鮮度フィルター (#128) ---

from datetime import date


class TestDiarySummaryFreshness:
    """日記サマリーの鮮度フィルター (#128)."""

    _SAMPLE_SUMMARY = (
        "# 日記サマリー\n\n"
        "## 2021-01\n\n"
        "- 2021年1月の内容\n\n"
        "## 2021-04\n\n"
        "- 2021年4月の内容\n\n"
        "## 2025-09\n\n"
        "- 2025年9月の内容\n\n"
        "## 2025-12\n\n"
        "- 2025年12月の内容\n\n"
        "## 2026-01\n\n"
        "- 2026年1月の内容\n\n"
        "## 2026-02\n\n"
        "- 2026年2月の内容\n"
    )

    def test_filter_removes_old_entries(self) -> None:
        """_filter_diary_summary が古いエントリを除去する."""
        builder = PromptBuilder()
        result = builder._filter_diary_summary(
            self._SAMPLE_SUMMARY, max_months=6, reference_date=date(2026, 3, 5),
        )
        assert "2021年1月" not in result
        assert "2021年4月" not in result

    def test_filter_keeps_recent_entries(self) -> None:
        """_filter_diary_summary が直近のエントリを保持する."""
        builder = PromptBuilder()
        result = builder._filter_diary_summary(
            self._SAMPLE_SUMMARY, max_months=6, reference_date=date(2026, 3, 5),
        )
        assert "2025年9月" in result
        assert "2025年12月" in result
        assert "2026年1月" in result
        assert "2026年2月" in result

    def test_tier1_section_excludes_old_summary(self) -> None:
        """_build_user_context_tier1_section が古いサマリーを含まない."""
        builder = PromptBuilder()
        ctx = UserContext(diary_summary=self._SAMPLE_SUMMARY)
        config = UserContextConfig(diary_summary_months=6)
        result = builder._build_user_context_tier1_section(
            user_context=ctx, config=config, reference_date=date(2026, 3, 5),
        )
        assert "2021年1月" not in result
        assert "2021年4月" not in result
        # 直近のエントリは含まれる（truncation によりすべてとは限らない）
        assert "2025年9月" in result

    def test_default_months_filters_old(self) -> None:
        """デフォルトの diary_summary_months でも古いエントリがフィルタされる."""
        builder = PromptBuilder()
        result = builder._filter_diary_summary(
            self._SAMPLE_SUMMARY, reference_date=date(2026, 3, 5),
        )
        assert "2021年1月" not in result

    def test_no_summary_returns_normally(self) -> None:
        """日記サマリーがない場合は正常に空を返す."""
        builder = PromptBuilder()
        ctx = UserContext()
        result = builder._build_user_context_tier1_section(user_context=ctx)
        assert result == ""

    def test_no_date_headers_keeps_all(self) -> None:
        """日付ヘッダーがない場合はそのまま表示."""
        builder = PromptBuilder()
        summary = "# 日記サマリー\n\nただのテキスト。日付ヘッダーなし。"
        result = builder._filter_diary_summary(
            summary, max_months=6, reference_date=date(2026, 3, 5),
        )
        assert "ただのテキスト" in result
