"""PAD → 离情绪标签映射（8 象限 + 中立）。"""

_DEFAULT_THRESHOLD = 0.3

# 8 象限情绪标签表（Discussion #11）
_OCTANT_TABLE: list[tuple[bool, bool, bool, str]] = [
    # (P+, A+, D+) → 标签
    (True, True, True, "喜悦（高涨）"),
    (True, True, False, "感动"),
    (True, False, True, "从容"),
    (True, False, False, "安详"),
    (False, True, True, "愤怒"),
    (False, True, False, "恐惧"),
    (False, False, True, "倦怠"),
    (False, False, False, "绝望"),
]


def pad_to_emotion_label(
    pleasure: float,
    arousal: float,
    dominance: float,
    threshold: float = _DEFAULT_THRESHOLD,
) -> str:
    """由 PAD 值返回离情绪标签。

    各轴绝对值都小于 threshold 时视为「中间」，
    所有轴均为中间时返回「中立」。
    只有部分轴处于中间时，该轴按正（>= 0）处理，选取最接近的象限。
    """
    p_sign = pleasure >= 0 if abs(pleasure) >= threshold else None
    a_sign = arousal >= 0 if abs(arousal) >= threshold else None
    d_sign = dominance >= 0 if abs(dominance) >= threshold else None

    # 所有轴均为中间 → 中立
    if p_sign is None and a_sign is None and d_sign is None:
        return "中立"

    # 处于中间的轴默认按正（>= 0）处理
    p_pos = p_sign if p_sign is not None else (pleasure >= 0)
    a_pos = a_sign if a_sign is not None else (arousal >= 0)
    d_pos = d_sign if d_sign is not None else (dominance >= 0)

    for pos_p, pos_a, pos_d, label in _OCTANT_TABLE:
        if p_pos == pos_p and a_pos == pos_a and d_pos == pos_d:
            return label

    return "中立"
