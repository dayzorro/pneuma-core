"""Tests for PAD 3D emotion model (Issue #3)."""

import math

import pytest

from pneuma_core.models.emotion import EmotionalState, Mood
from pneuma_core.models.personality import Personality
from pneuma_core.emotion.baseline import personality_to_pad_baseline
from pneuma_core.emotion.pad_mapping import pad_to_emotion_label
from pneuma_core.emotion.decay import exponential_decay


# --- EmotionalState dataclass tests ---


class TestEmotionalState:
    """EmotionalState の基本テスト."""

    def test_create_valid_state(self) -> None:
        state = EmotionalState(
            pleasure=0.5,
            arousal=0.3,
            dominance=-0.2,
            emotion_label="喜び（高揚）",
            situation="楽しい会話",
        )
        assert state.pleasure == 0.5
        assert state.arousal == 0.3
        assert state.dominance == -0.2
        assert state.emotion_label == "喜び（高揚）"
        assert state.situation == "楽しい会話"

    def test_create_neutral_state(self) -> None:
        state = EmotionalState(
            pleasure=0.0,
            arousal=0.0,
            dominance=0.0,
            emotion_label="中立",
            situation="初めての会話",
        )
        assert state.pleasure == 0.0
        assert state.emotion_label == "中立"

    def test_create_extreme_values(self) -> None:
        state = EmotionalState(
            pleasure=-1.0,
            arousal=1.0,
            dominance=-1.0,
            emotion_label="恐怖",
            situation="危険な状況",
        )
        assert state.pleasure == -1.0
        assert state.arousal == 1.0
        assert state.dominance == -1.0

    def test_frozen(self) -> None:
        state = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="テスト",
        )
        with pytest.raises(AttributeError):
            state.pleasure = 0.5  # type: ignore[misc]

    def test_pleasure_too_high(self) -> None:
        with pytest.raises(ValueError, match="pleasure"):
            EmotionalState(
                pleasure=1.1, arousal=0.0, dominance=0.0,
                emotion_label="", situation="",
            )

    def test_pleasure_too_low(self) -> None:
        with pytest.raises(ValueError, match="pleasure"):
            EmotionalState(
                pleasure=-1.1, arousal=0.0, dominance=0.0,
                emotion_label="", situation="",
            )

    def test_arousal_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="arousal"):
            EmotionalState(
                pleasure=0.0, arousal=2.0, dominance=0.0,
                emotion_label="", situation="",
            )

    def test_dominance_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dominance"):
            EmotionalState(
                pleasure=0.0, arousal=0.0, dominance=-1.5,
                emotion_label="", situation="",
            )


# --- PAD → Emotion Label mapping tests ---


class TestPadToEmotionLabel:
    """8オクタント + 中立マッピングのテスト."""

    def test_joy_elevated(self) -> None:
        """P+ A+ D+ → 喜び（高揚）"""
        assert pad_to_emotion_label(0.5, 0.5, 0.5) == "喜悦（高涨）"

    def test_moved(self) -> None:
        """P+ A+ D- → 感動"""
        assert pad_to_emotion_label(0.5, 0.5, -0.5) == "感动"

    def test_composure(self) -> None:
        """P+ A- D+ → 余裕"""
        assert pad_to_emotion_label(0.5, -0.5, 0.5) == "从容"

    def test_serenity(self) -> None:
        """P+ A- D- → 安らぎ"""
        assert pad_to_emotion_label(0.5, -0.5, -0.5) == "安详"

    def test_anger(self) -> None:
        """P- A+ D+ → 怒り"""
        assert pad_to_emotion_label(-0.5, 0.5, 0.5) == "愤怒"

    def test_fear(self) -> None:
        """P- A+ D- → 恐怖"""
        assert pad_to_emotion_label(-0.5, 0.5, -0.5) == "恐惧"

    def test_ennui(self) -> None:
        """P- A- D+ → 倦怠"""
        assert pad_to_emotion_label(-0.5, -0.5, 0.5) == "倦怠"

    def test_despair(self) -> None:
        """P- A- D- → 絶望"""
        assert pad_to_emotion_label(-0.5, -0.5, -0.5) == "绝望"

    def test_neutral(self) -> None:
        """中間値 → 中立"""
        assert pad_to_emotion_label(0.0, 0.0, 0.0) == "中立"

    def test_neutral_within_threshold(self) -> None:
        """しきい値内は中立."""
        assert pad_to_emotion_label(0.1, -0.1, 0.05) == "中立"

    def test_boundary_positive(self) -> None:
        """しきい値ちょうどは正の象限に入る."""
        # デフォルトしきい値は Issue の 0.3 想定
        # P=0.3 は P>0 の象限
        label = pad_to_emotion_label(0.3, 0.3, 0.3)
        assert label == "喜悦（高涨）"

    def test_mixed_threshold(self) -> None:
        """一部の軸だけ中間の場合."""
        # P+ で A と D が中間 → pleasure が正で arousal/dominance が neutral
        # 最も近い象限にフォールバック
        # P+ で A,D が中間(正扱い) → P+ A+ D+ = 喜び（高揚）
        label = pad_to_emotion_label(0.8, 0.0, 0.0)
        assert label == "喜悦（高涨）"


# --- Big Five → PAD Baseline tests ---


class TestPersonalityToPadBaseline:
    """Big Five → PAD ベースライン変換のテスト."""

    def test_aine_baseline(self) -> None:
        """アイネの Big Five 値からの変換.

        O=0.8, C=0.5, E=0.3, A=0.6, N=0.7
        P = 0.21×0.3 + 0.59×0.6 + 0.19×(1-0.7) = 0.063 + 0.354 + 0.057 = 0.474
        A = 0.15×0.8 + 0.30×(1-0.6) + 0.57×0.7 = 0.12 + 0.12 + 0.399 = 0.639
        D = 0.25×0.8 + 0.17×0.5 + 0.60×0.3 - 0.32×0.6 = 0.2 + 0.085 + 0.18 - 0.192 = 0.273
        """
        aine = Personality(
            openness=0.8,
            conscientiousness=0.5,
            extraversion=0.3,
            agreeableness=0.6,
            neuroticism=0.7,
        )
        p, a, d = personality_to_pad_baseline(aine)
        assert p == pytest.approx(0.474, abs=0.001)
        assert a == pytest.approx(0.639, abs=0.001)
        assert d == pytest.approx(0.273, abs=0.001)

    def test_all_zero(self) -> None:
        """全て0の場合."""
        p_all_zero = Personality(
            openness=0.0, conscientiousness=0.0, extraversion=0.0,
            agreeableness=0.0, neuroticism=0.0,
        )
        p, a, d = personality_to_pad_baseline(p_all_zero)
        # P = 0.21×0 + 0.59×0 + 0.19×(1-0) = 0.19
        # A = 0.15×0 + 0.30×(1-0) + 0.57×0 = 0.30
        # D = 0.25×0 + 0.17×0 + 0.60×0 - 0.32×0 = 0.0
        assert p == pytest.approx(0.19, abs=0.001)
        assert a == pytest.approx(0.30, abs=0.001)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_all_one(self) -> None:
        """全て1.0の場合."""
        p_all_one = Personality(
            openness=1.0, conscientiousness=1.0, extraversion=1.0,
            agreeableness=1.0, neuroticism=1.0,
        )
        p, a, d = personality_to_pad_baseline(p_all_one)
        # P = 0.21×1 + 0.59×1 + 0.19×(1-1) = 0.80
        # A = 0.15×1 + 0.30×(1-1) + 0.57×1 = 0.72
        # D = 0.25×1 + 0.17×1 + 0.60×1 - 0.32×1 = 0.70
        assert p == pytest.approx(0.80, abs=0.001)
        assert a == pytest.approx(0.72, abs=0.001)
        assert d == pytest.approx(0.70, abs=0.001)

    def test_high_extraversion_increases_pleasure(self) -> None:
        """外向性が高いと Pleasure が高くなる."""
        low_e = Personality(
            openness=0.5, conscientiousness=0.5, extraversion=0.1,
            agreeableness=0.5, neuroticism=0.5,
        )
        high_e = Personality(
            openness=0.5, conscientiousness=0.5, extraversion=0.9,
            agreeableness=0.5, neuroticism=0.5,
        )
        p_low, _, _ = personality_to_pad_baseline(low_e)
        p_high, _, _ = personality_to_pad_baseline(high_e)
        assert p_high > p_low

    def test_high_neuroticism_increases_arousal(self) -> None:
        """神経症傾向が高いと Arousal が高くなる."""
        low_n = Personality(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.1,
        )
        high_n = Personality(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.9,
        )
        _, a_low, _ = personality_to_pad_baseline(low_n)
        _, a_high, _ = personality_to_pad_baseline(high_n)
        assert a_high > a_low

    def test_baseline_clamped_to_pad_range(self) -> None:
        """極端な値でも [-1.0, 1.0] にクランプされる."""
        extreme = Personality(
            openness=1.0, conscientiousness=1.0, extraversion=1.0,
            agreeableness=1.0, neuroticism=1.0,
        )
        p, a, d = personality_to_pad_baseline(extreme)
        assert -1.0 <= p <= 1.0
        assert -1.0 <= a <= 1.0
        assert -1.0 <= d <= 1.0


# --- Exponential decay tests ---


class TestExponentialDecay:
    """感情の指数減衰テスト."""

    def test_no_time_elapsed(self) -> None:
        """時間0ではベースラインに向かわない."""
        result = exponential_decay(
            current=0.8, baseline=0.0, elapsed_seconds=0.0, half_life=300.0
        )
        assert result == pytest.approx(0.8, abs=0.001)

    def test_at_half_life(self) -> None:
        """半減期ではベースラインとの差が半分になる."""
        result = exponential_decay(
            current=1.0, baseline=0.0, elapsed_seconds=300.0, half_life=300.0
        )
        assert result == pytest.approx(0.5, abs=0.001)

    def test_double_half_life(self) -> None:
        """半減期2倍では差が1/4になる."""
        result = exponential_decay(
            current=1.0, baseline=0.0, elapsed_seconds=600.0, half_life=300.0
        )
        assert result == pytest.approx(0.25, abs=0.001)

    def test_negative_to_baseline(self) -> None:
        """負の値もベースラインに向かう."""
        result = exponential_decay(
            current=-0.8, baseline=0.0, elapsed_seconds=300.0, half_life=300.0
        )
        assert result == pytest.approx(-0.4, abs=0.001)

    def test_nonzero_baseline(self) -> None:
        """ベースラインが0でない場合."""
        result = exponential_decay(
            current=1.0, baseline=0.5, elapsed_seconds=300.0, half_life=300.0
        )
        # diff = 1.0 - 0.5 = 0.5, half-life → 0.25 remaining → 0.5 + 0.25 = 0.75
        assert result == pytest.approx(0.75, abs=0.001)

    def test_large_time_approaches_baseline(self) -> None:
        """十分な時間経過でベースラインに収束."""
        result = exponential_decay(
            current=1.0, baseline=0.2, elapsed_seconds=10000.0, half_life=300.0
        )
        assert result == pytest.approx(0.2, abs=0.01)

    def test_already_at_baseline(self) -> None:
        """すでにベースラインにいる場合は変化しない."""
        result = exponential_decay(
            current=0.3, baseline=0.3, elapsed_seconds=300.0, half_life=300.0
        )
        assert result == pytest.approx(0.3, abs=0.001)

    def test_zero_half_life_raises(self) -> None:
        """half_life=0 は ValueError."""
        with pytest.raises(ValueError, match="half_life"):
            exponential_decay(current=1.0, baseline=0.0, elapsed_seconds=1.0, half_life=0.0)

    def test_negative_half_life_raises(self) -> None:
        """負の half_life は ValueError."""
        with pytest.raises(ValueError, match="half_life"):
            exponential_decay(current=1.0, baseline=0.0, elapsed_seconds=1.0, half_life=-1.0)


# --- Mood dataclass tests ---


class TestMood:
    """Mood（中期感情）のテスト."""

    def test_create_mood(self) -> None:
        mood = Mood(pleasure=0.3, arousal=0.1, dominance=-0.1)
        assert mood.pleasure == 0.3
        assert mood.arousal == 0.1
        assert mood.dominance == -0.1

    def test_mood_validation(self) -> None:
        with pytest.raises(ValueError, match="pleasure"):
            Mood(pleasure=1.5, arousal=0.0, dominance=0.0)

    def test_mood_frozen(self) -> None:
        mood = Mood(pleasure=0.0, arousal=0.0, dominance=0.0)
        with pytest.raises(AttributeError):
            mood.pleasure = 0.5  # type: ignore[misc]

    def test_update_with_emotion(self) -> None:
        """感情を反映して新しい Mood を返す."""
        mood = Mood(pleasure=0.0, arousal=0.0, dominance=0.0)
        state = EmotionalState(
            pleasure=1.0, arousal=0.5, dominance=0.3,
            emotion_label="喜び（高揚）", situation="嬉しいニュース",
        )
        # alpha=0.3 (デフォルト) で移動平均
        new_mood = mood.update(state, alpha=0.3)
        assert new_mood.pleasure == pytest.approx(0.3, abs=0.001)
        assert new_mood.arousal == pytest.approx(0.15, abs=0.001)
        assert new_mood.dominance == pytest.approx(0.09, abs=0.001)

    def test_update_preserves_momentum(self) -> None:
        """既存の Mood に感情が加算される."""
        mood = Mood(pleasure=0.5, arousal=0.5, dominance=0.5)
        state = EmotionalState(
            pleasure=1.0, arousal=1.0, dominance=1.0,
            emotion_label="喜び（高揚）", situation="最高の瞬間",
        )
        new_mood = mood.update(state, alpha=0.3)
        # new = (1-0.3)*0.5 + 0.3*1.0 = 0.35 + 0.3 = 0.65
        assert new_mood.pleasure == pytest.approx(0.65, abs=0.001)
