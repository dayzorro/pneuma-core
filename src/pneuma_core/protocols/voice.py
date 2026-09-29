"""Voice Protocols: TTSAdapter and STTService."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSAdapter(Protocol):
    """把文本转换为音频二进制的适配器抽象接口。"""

    async def synthesize(
        self, text: str, *, emotion_tags: list[str] | None = None
    ) -> bytes:
        """把文本转换为音频二进制。"""
        ...


@runtime_checkable
class STTService(Protocol):
    """把音频二进制转换为文本的服务抽象接口。"""

    async def transcribe(self, audio: bytes, *, language: str = "zh") -> str:
        """把音频二进制转换为文本。"""
        ...
