"""EmotionEngine：基于 LLM 的 PAD 情绪估计与生命周期管理。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pneuma_core.emotion.baseline import personality_to_pad_baseline
from pneuma_core.emotion.decay import exponential_decay
from pneuma_core.emotion.pad_mapping import pad_to_emotion_label
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.personality import Personality

logger = logging.getLogger(__name__)

_MD_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)

_SYSTEM_PROMPT_BASE = """\
你是一位对话情绪分析专家。
请阅读以下对话历史，用 PAD 模型推断角色当前的情绪状态。

## 角色的性格特质（Big Five）
- 开放性 (Openness): {openness}
- 尽责性 (Conscientiousness): {conscientiousness}
- 外向性 (Extraversion): {extraversion}
- 宜人性 (Agreeableness): {agreeableness}
- 神经质 (Neuroticism): {neuroticism}

请结合角色的性格特质来推断情绪状态。

请严格按照以下 JSON 格式回答，所有文本字段必须使用简体中文：
{{
  "pleasure": <-1.0〜1.0>,
  "arousal": <-1.0〜1.0>,
  "dominance": <-1.0〜1.0>,
  "emotion_label": "<情绪标签，2〜4 个汉字的简体中文，例如：喜悦、不安、平静、感动>",
  "situation": "<用一句简体中文描述当前状况>"
}}
"""


def _build_system_prompt(personality: Personality) -> str:
    """构建包含性格信息的 system prompt。"""
    return _SYSTEM_PROMPT_BASE.format(
        openness=personality.openness,
        conscientiousness=personality.conscientiousness,
        extraversion=personality.extraversion,
        agreeableness=personality.agreeableness,
        neuroticism=personality.neuroticism,
    )

NEUTRAL_EMOTION = EmotionalState(
    pleasure=0.0, arousal=0.0, dominance=0.0,
    emotion_label="中立", situation="",
)


_MAX_LABEL_LEN = 50
_MAX_SITUATION_LEN = 200


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sanitize_text(text: str, max_len: int) -> str:
    """移除控制字符并截断。"""
    cleaned = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return cleaned[:max_len]


@dataclass(frozen=True)
class EmotionConfig:
    """EmotionEngine 的配置。"""

    recent_messages_limit: int = 10
    decay_half_life: float = 3600.0


@dataclass(frozen=True)
class EmotionResult:
    """情绪评估结果，附带触发信息。"""

    state: EmotionalState
    trigger_type: str  # "triggered"
    reasons: list[str] = field(default_factory=list)


class EmotionEngine:
    """基于 LLM 的情绪估计，并带有基线衰减。"""

    def __init__(
        self,
        llm: LLMAdapter,
        config: EmotionConfig | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or EmotionConfig()
        self._model = model

    async def evaluate(
        self,
        personality: Personality,
        messages: list[dict],
        turn_count: int,
        current_state: EmotionalState,
    ) -> EmotionResult:
        """每轮都通过直接的 LLM 估计来评估情绪。

        每轮调用一次 estimate() 直接获得 PAD 值。
        不再使用 Tier 0/1 门控——相较此前的混合触发机制已简化。

        Returns:
            trigger_type="triggered" 的 EmotionResult。
        """
        state = await self.estimate(personality, messages)
        return EmotionResult(
            state=state,
            trigger_type="triggered",
        )

    async def estimate(
        self,
        personality: Personality,
        messages: list[dict],
    ) -> EmotionalState:
        """通过 LLM 从对话中估计情绪状态。

        任何错误（JSON 格式错误、缺字段、LLM 异常）都返回中立状态。
        """
        truncated = messages[-self._config.recent_messages_limit :]

        request = LLMRequest(
            system_prompt=_build_system_prompt(personality),
            messages=truncated,
            model=self._model,
            # 情绪评估在每轮请求的关键路径上，关闭思考以显著降低延迟
            enable_thinking=False,
            max_tokens=256,
        )

        try:
            response = await self._llm.generate(request)
        except Exception as e:
            logger.warning("LLM generate failed: %s: %s, returning neutral state", type(e).__name__, e)
            return NEUTRAL_EMOTION

        try:
            text = response.content.strip()
            md_match = _MD_CODE_BLOCK_RE.match(text)
            if md_match:
                text = md_match.group(1).strip()
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Malformed LLM response (%s), raw=%r, returning neutral state",
                e, response.content[:200] if response.content else "<empty>",
            )
            return NEUTRAL_EMOTION

        if not isinstance(data, dict):
            return NEUTRAL_EMOTION

        required = ("pleasure", "arousal", "dominance", "emotion_label", "situation")
        if not all(k in data for k in required):
            return NEUTRAL_EMOTION

        try:
            pleasure = _clamp(float(data["pleasure"]))
            arousal = _clamp(float(data["arousal"]))
            dominance = _clamp(float(data["dominance"]))
        except (ValueError, TypeError):
            return NEUTRAL_EMOTION

        return EmotionalState(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            emotion_label=_sanitize_text(str(data["emotion_label"]), _MAX_LABEL_LEN),
            situation=_sanitize_text(str(data["situation"]), _MAX_SITUATION_LEN),
        )

    def decay_towards_baseline(
        self,
        state: EmotionalState,
        personality: Personality,
        elapsed_seconds: float,
    ) -> EmotionalState:
        """让情绪状态向性格基线衰减。"""
        baseline = personality_to_pad_baseline(personality)

        pleasure = exponential_decay(
            state.pleasure, baseline[0], elapsed_seconds, self._config.decay_half_life,
        )
        arousal = exponential_decay(
            state.arousal, baseline[1], elapsed_seconds, self._config.decay_half_life,
        )
        dominance = exponential_decay(
            state.dominance, baseline[2], elapsed_seconds, self._config.decay_half_life,
        )

        label = pad_to_emotion_label(pleasure, arousal, dominance)

        return EmotionalState(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            emotion_label=label,
            situation=state.situation,
        )
