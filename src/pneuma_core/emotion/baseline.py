"""Big Five → PAD 基线转换。"""

from pneuma_core.models.personality import Personality


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def personality_to_pad_baseline(personality: Personality) -> tuple[float, float, float]:
    """由 Big Five 性格参数计算 PAD 基线。

    P_baseline = 0.21×E + 0.59×A + 0.19×(1-N)
    A_baseline = 0.15×O + 0.30×(1-A) + 0.57×N
    D_baseline = 0.25×O + 0.17×C + 0.60×E - 0.32×A

    返回值会被裁剪到 [-1.0, 1.0]。

    Returns:
        (pleasure, arousal, dominance) 元组
    """
    e = personality.extraversion
    a = personality.agreeableness
    n = personality.neuroticism
    o = personality.openness
    c = personality.conscientiousness

    pleasure = 0.21 * e + 0.59 * a + 0.19 * (1 - n)
    arousal = 0.15 * o + 0.30 * (1 - a) + 0.57 * n
    dominance = 0.25 * o + 0.17 * c + 0.60 * e - 0.32 * a

    return (_clamp(pleasure), _clamp(arousal), _clamp(dominance))
