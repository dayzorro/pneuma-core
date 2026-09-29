"""HistorySummarizer: compresses overflow conversation history via LLM summary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pneuma_core.llm.adapter import LLMRequest

if TYPE_CHECKING:
    from pneuma_core.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM_PROMPT = """\
你是一个对话历史摘要助手。
请简洁地总结以下对话历史。不要遗漏重要信息（姓名、话题、决定、情绪变化等），\
并尽量简短。摘要请使用简体中文。"""

_SUMMARIZE_WITH_PREVIOUS_PROMPT = """\
你是一个对话历史摘要助手。
下面有「此前的摘要」与「新的对话历史」。
请把它们整合成一段简洁的摘要。不要遗漏重要信息（姓名、话题、决定、情绪变化等），\
并尽量简短。摘要请使用简体中文。

【此前的摘要】
{previous_summary}"""


class HistorySummarizer:
    """Summarizes overflow conversation history using LLM.

    When conversation history exceeds the limit, the overflow messages
    are summarized. The summary text is returned separately from the
    trimmed messages so callers can include it in system_prompt.
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm
        self._current_summary: str | None = None

    @property
    def current_summary(self) -> str | None:
        """Return the current accumulated summary, or None if no summary exists."""
        return self._current_summary

    def should_summarize(self, history: list[dict], limit: int) -> bool:
        """Check if history exceeds the limit and needs summarization."""
        return len(history) > limit

    async def summarize(
        self, history: list[dict], limit: int
    ) -> str | None:
        """Summarize overflow messages (those beyond the limit).

        Args:
            history: Full conversation history.
            limit: Maximum number of recent messages to keep.

        Returns:
            Summary text, or None if summarization failed.
        """
        if not self.should_summarize(history, limit):
            return self._current_summary

        overflow = history[: len(history) - limit]
        overflow_text = self._format_messages(overflow)

        try:
            if self._current_summary is not None:
                system_prompt = _SUMMARIZE_WITH_PREVIOUS_PROMPT.format(
                    previous_summary=self._current_summary
                )
            else:
                system_prompt = _SUMMARIZE_SYSTEM_PROMPT

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"请总结以下对话:\n\n{overflow_text}",
                    }
                ],
                model="claude-haiku-4-5-20251001",
                temperature=0.3,
                max_tokens=512,
            )
            response = await self._llm.generate(request)
            self._current_summary = response.content
            return self._current_summary
        except Exception:
            logger.warning("History summarization failed, returning None")
            return None

    async def trim_with_summary(
        self, history: list[dict], limit: int
    ) -> tuple[str | None, list[dict]]:
        """Trim history with summarization.

        If history exceeds limit, summarize overflow and return
        (summary_text, recent_messages). If summarization fails,
        fall back to simple truncation with no summary.

        Args:
            history: Full conversation history.
            limit: Maximum number of recent messages to keep.

        Returns:
            Tuple of (summary_text or None, trimmed messages).
        """
        if not self.should_summarize(history, limit):
            return (self._current_summary, history)

        summary = await self.summarize(history, limit)
        recent = history[-limit:]

        return (summary, recent)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """Format messages for the summarization prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
