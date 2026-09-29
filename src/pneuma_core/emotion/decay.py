"""情绪 → 基线回归的指数衰减。"""

import math


def exponential_decay(
    current: float,
    baseline: float,
    elapsed_seconds: float,
    half_life: float,
) -> float:
    """用指数衰减把情绪值向基线靠拢。

    decay = exp(-ln(2) / half_life * elapsed_seconds)
    result = baseline + (current - baseline) * decay

    Args:
        current: 当前情绪值
        baseline: 基线值
        elapsed_seconds: 经过的秒数
        half_life: 半衰期（秒）

    Returns:
        衰减后的情绪值
    """
    if half_life <= 0.0:
        raise ValueError(f"half_life must be positive, got {half_life}")

    if elapsed_seconds <= 0.0:
        return current

    decay_rate = math.log(2) / half_life
    decay = math.exp(-decay_rate * elapsed_seconds)

    return baseline + (current - baseline) * decay
