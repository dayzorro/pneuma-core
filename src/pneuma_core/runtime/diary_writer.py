"""DiaryWriter：自动生成角色的日记。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest

logger = logging.getLogger(__name__)

# JST（日本标准时间，UTC+9）
JST = timezone(timedelta(hours=9))


class DiaryWriter:
    """自动生成角色的日记。"""

    TRIGGER_KEYWORDS = ["晚安", "早安", "早上好", "我睡了"]

    def __init__(
        self,
        llm: LLMAdapter,
        logs_dir: Path,
        diary_dir: Path,
        character_name: str,
        character_profile: str,
        model: str = "claude-opus-4-6",
    ) -> None:
        self._llm = llm
        self._logs_dir = logs_dir
        self._diary_dir = diary_dir
        self._character_name = character_name
        self._character_profile = character_profile
        self._model = model
        self._generated_dates: set[str] = set()  # 每天最多一次

    def _get_effective_date(self) -> str:
        """返回生效日期。JST 0:00-3:59 视为前一天。"""
        now = datetime.now(JST)
        if now.hour < 4:
            now = now - timedelta(days=1)
        return now.strftime("%Y-%m-%d")

    def should_trigger(self, user_input: str) -> bool:
        """判断用户发言中是否包含触发关键词。"""
        return any(kw in user_input for kw in self.TRIGGER_KEYWORDS)

    async def maybe_generate(self, user_input: str) -> None:
        """满足触发条件时生成日记。"""
        if not self.should_trigger(user_input):
            return
        today = self._get_effective_date()
        if today in self._generated_dates:
            return  # 每天最多一次
        await self._generate_diary(today)
        self._generated_dates.add(today)

    async def _generate_diary(self, date_str: str) -> None:
        """生成日记并保存到 diary_dir/YYYY-MM-DD.md。"""
        # 1. 读取当天的对话日志
        log_path = self._logs_dir / f"{date_str}.md"
        log_content = ""
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")

        # 2. 让 LLM 写日记
        system_prompt = (
            f"你是「{self._character_name}」。"
            f"请基于以下角色设定，写下今天的日记。\n\n"
            f"{self._character_profile}\n\n"
            f"## 日记的写法\n"
            f"- 以第一人称书写（作为角色本人）\n"
            f"- 自由地写下今天发生的事、感受和想法\n"
            f"- 体现角色的口吻与性格\n"
            f"- 要读起来像一篇自然的日记（用成段的文字，而不是要点列表）\n"
            f"- 长度约 200〜400 字\n"
            f"- 不要 markdown 标题或装饰，只写正文\n"
            f"- 使用简体中文书写"
        )

        user_message = (
            f"今天是 {date_str}。请根据今天的对话日志写一篇日记。\n\n"
            f"## 今天的对话日志\n"
            f"{log_content if log_content else '（今天没有对话）'}"
        )

        response = await self._llm.generate(
            LLMRequest(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                model=self._model,
            )
        )

        # 3. 保存到文件
        self._diary_dir.mkdir(parents=True, exist_ok=True)
        diary_path = self._diary_dir / f"{date_str}.md"
        diary_path.write_text(response.content, encoding="utf-8")
