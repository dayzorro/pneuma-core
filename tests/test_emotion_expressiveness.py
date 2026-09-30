"""Tests for 情绪收敛（emotional_expressiveness）——服务型岗位的「收敛为服务式」。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pneuma_core.emotion import personality_to_pad_baseline
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.personality import Personality
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.runtime.emotion_engine import EmotionEngine, EmotionResult

PERSONALITY = Personality(
    openness=0.55,
    conscientiousness=0.85,
    extraversion=0.72,
    agreeableness=0.9,
    neuroticism=0.15,
)

CURRENT = EmotionalState(
    pleasure=0.45, arousal=0.15, dominance=0.25,
    emotion_label="亲切", situation="在服务台值班",
)


def _llm(pleasure: float, arousal: float, dominance: float) -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=AsyncMock(
            content=json.dumps(
                {
                    "pleasure": pleasure,
                    "arousal": arousal,
                    "dominance": dominance,
                    "emotion_label": "烦躁",
                    "situation": "顾客催得很急",
                },
                ensure_ascii=False,
            )
        )
    )
    return llm


class TestEstimateDamping:
    @pytest.mark.asyncio
    async def test_full_expressiveness_keeps_raw_values(self) -> None:
        engine = EmotionEngine(llm=_llm(-0.8, 0.7, -0.5))

        state = await engine.estimate(PERSONALITY, [{"role": "user", "content": "x"}])

        assert state.pleasure == pytest.approx(-0.8)
        assert state.arousal == pytest.approx(0.7)
        assert state.dominance == pytest.approx(-0.5)

    @pytest.mark.asyncio
    async def test_low_expressiveness_pulls_towards_baseline(self) -> None:
        engine = EmotionEngine(llm=_llm(-0.8, 0.7, -0.5))
        base_p, base_a, base_d = personality_to_pad_baseline(PERSONALITY)

        state = await engine.estimate(
            PERSONALITY, [{"role": "user", "content": "x"}], expressiveness=0.45
        )

        assert state.pleasure == pytest.approx(base_p + (-0.8 - base_p) * 0.45)
        assert state.arousal == pytest.approx(base_a + (0.7 - base_a) * 0.45)
        assert state.dominance == pytest.approx(base_d + (-0.5 - base_d) * 0.45)

    @pytest.mark.asyncio
    async def test_damping_reduces_deviation_from_baseline(self) -> None:
        raw = await EmotionEngine(llm=_llm(-0.8, 0.7, -0.5)).estimate(
            PERSONALITY, [{"role": "user", "content": "x"}]
        )
        damped = await EmotionEngine(llm=_llm(-0.8, 0.7, -0.5)).estimate(
            PERSONALITY, [{"role": "user", "content": "x"}], expressiveness=0.45
        )
        base_p, _, _ = personality_to_pad_baseline(PERSONALITY)

        assert abs(damped.pleasure - base_p) < abs(raw.pleasure - base_p)

    @pytest.mark.asyncio
    async def test_zero_expressiveness_sits_exactly_on_baseline(self) -> None:
        base_p, base_a, base_d = personality_to_pad_baseline(PERSONALITY)

        state = await EmotionEngine(llm=_llm(-0.8, 0.7, -0.5)).estimate(
            PERSONALITY, [{"role": "user", "content": "x"}], expressiveness=0.0
        )

        assert state.pleasure == pytest.approx(base_p)
        assert state.arousal == pytest.approx(base_a)
        assert state.dominance == pytest.approx(base_d)

    @pytest.mark.asyncio
    async def test_label_and_situation_are_preserved(self) -> None:
        state = await EmotionEngine(llm=_llm(-0.8, 0.7, -0.5)).estimate(
            PERSONALITY, [{"role": "user", "content": "x"}], expressiveness=0.45
        )

        assert state.emotion_label == "烦躁"
        assert state.situation == "顾客催得很急"

    @pytest.mark.asyncio
    async def test_damped_values_stay_in_range(self) -> None:
        state = await EmotionEngine(llm=_llm(-1.0, 1.0, -1.0)).estimate(
            PERSONALITY, [{"role": "user", "content": "x"}], expressiveness=0.9
        )

        assert -1.0 <= state.pleasure <= 1.0
        assert -1.0 <= state.arousal <= 1.0
        assert -1.0 <= state.dominance <= 1.0


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_forwards_expressiveness(self) -> None:
        engine = EmotionEngine(llm=_llm(-0.8, 0.7, -0.5))
        base_p, _, _ = personality_to_pad_baseline(PERSONALITY)

        result = await engine.evaluate(
            personality=PERSONALITY,
            messages=[{"role": "user", "content": "x"}],
            turn_count=1,
            current_state=CURRENT,
            expressiveness=0.45,
        )

        assert isinstance(result, EmotionResult)
        assert result.trigger_type == "triggered"
        assert result.state.pleasure == pytest.approx(base_p + (-0.8 - base_p) * 0.45)


class _SpyEmotionEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def evaluate(self, **kwargs) -> EmotionResult:
        self.calls.append(kwargs)
        return EmotionResult(state=CURRENT, trigger_type="triggered")


class _StubStorage:
    def __init__(self) -> None:
        self.saved: list[EmotionalState] = []

    async def save_emotional_state(self, character_id: str, state) -> None:
        self.saved.append(state)


class TestEngineWiring:
    @pytest.mark.asyncio
    async def test_engine_passes_character_expressiveness(self) -> None:
        """RuntimeEngine 必须把角色的 emotional_expressiveness 传给情绪引擎。"""
        storage = _StubStorage()
        engine = RuntimeEngine(
            character_id="frontdesk-001",
            storage=storage,
            llm=AsyncMock(),
            embedding_service=AsyncMock(),
            memory_store=AsyncMock(),
        )
        spy = _SpyEmotionEngine()
        engine._emotion_engine = spy  # type: ignore[assignment]

        await engine._async_emotion_evaluation(
            personality=PERSONALITY,
            messages=[],
            current_emotion=CURRENT,
            expressiveness=0.45,
        )

        assert spy.calls[0]["expressiveness"] == pytest.approx(0.45)
        assert storage.saved == [CURRENT]
