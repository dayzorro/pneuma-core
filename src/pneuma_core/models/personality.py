"""Big Five personality model."""

from dataclasses import dataclass

_HIGH_THRESHOLD = 0.7
_LOW_THRESHOLD = 0.3

_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def _validate_range(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    return value


@dataclass(frozen=True)
class Personality:
    """Big Five 性格模型。

    各参数范围均为 0.0〜1.0。
    阈值：低于 0.3=低，0.7 及以上=高
    """

    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def __post_init__(self) -> None:
        for trait in _TRAITS:
            _validate_range(getattr(self, trait), trait)

    def is_high(self, trait: str) -> bool:
        """判断指定特质是否为高（0.7 及以上）。"""
        return getattr(self, trait) >= _HIGH_THRESHOLD

    def is_low(self, trait: str) -> bool:
        """判断指定特质是否为低（低于 0.3）。"""
        return getattr(self, trait) < _LOW_THRESHOLD
