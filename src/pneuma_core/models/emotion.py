"""PAD 3D emotion model: EmotionalState, Mood."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_pad(value: float, name: str) -> float:
    if not (-1.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between -1.0 and 1.0, got {value}")
    return value


@dataclass(frozen=True)
class EmotionalState:
    """PAD 三维情绪状态。

    pleasure: -1.0〜1.0（不快〜愉悦）
    arousal: -1.0〜1.0（平静〜兴奋）
    dominance: -1.0〜1.0（顺从〜支配）
    emotion_label: 离散标签（用于展示）
    situation: 当前状况（一句话）
    """

    pleasure: float
    arousal: float
    dominance: float
    emotion_label: str
    situation: str

    def __post_init__(self) -> None:
        _validate_pad(self.pleasure, "pleasure")
        _validate_pad(self.arousal, "arousal")
        _validate_pad(self.dominance, "dominance")


@dataclass(frozen=True)
class Mood:
    """中期情绪（心情）：情绪的移动平均。

    与 PAD 相同的 -1.0〜1.0 范围。
    """

    pleasure: float
    arousal: float
    dominance: float

    def __post_init__(self) -> None:
        _validate_pad(self.pleasure, "pleasure")
        _validate_pad(self.arousal, "arousal")
        _validate_pad(self.dominance, "dominance")

    def update(self, state: EmotionalState, alpha: float = 0.3) -> Mood:
        """返回反映该情绪后的新 Mood（指数移动平均）。

        new_value = (1 - alpha) * current + alpha * emotion
        """
        return Mood(
            pleasure=(1 - alpha) * self.pleasure + alpha * state.pleasure,
            arousal=(1 - alpha) * self.arousal + alpha * state.arousal,
            dominance=(1 - alpha) * self.dominance + alpha * state.dominance,
        )
