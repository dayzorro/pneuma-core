"""Schwartz values model."""

from dataclasses import dataclass

_IMPORTANCE_THRESHOLD = 0.6

_DIMENSIONS = ("self_transcendence", "self_enhancement", "openness_to_change", "conservation")


def _validate_range(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    return value


@dataclass(frozen=True)
class Values:
    """Schwartz 4 类价值观模型。

    各参数范围均为 0.0〜1.0。
    对立轴: self_transcendence ↔ self_enhancement, openness_to_change ↔ conservation
    阈值：0.6 及以上视为「重视」
    """

    self_transcendence: float
    self_enhancement: float
    openness_to_change: float
    conservation: float

    def __post_init__(self) -> None:
        for dim in _DIMENSIONS:
            _validate_range(getattr(self, dim), dim)

    def is_important(self, dimension: str) -> bool:
        """判断指定价值观是否被重视（0.6 及以上）。"""
        return getattr(self, dimension) >= _IMPORTANCE_THRESHOLD
