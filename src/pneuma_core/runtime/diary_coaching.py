"""DiaryCoaching：阅读用户日记，辅助进行教练式回应。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


# JST（日本标准时间，UTC+9）
JST = timezone(timedelta(hours=9))


class DiaryCoaching:
    """阅读用户日记，辅助进行教练式回应。"""

    TRIGGER_KEYWORDS = ["写了日记", "看下日记", "看看日记", "日记写好了"]

    def __init__(self, diary_dir: Path) -> None:
        self._diary_dir = diary_dir

    def should_trigger(self, user_input: str) -> bool:
        """判断用户发言中是否包含触发关键词。"""
        return any(kw in user_input for kw in self.TRIGGER_KEYWORDS)

    def get_diary_content(self, date_str: str | None = None) -> str | None:
        """获取日记内容。

        date_str 为 None 时先查找今天的日记，找不到则返回最新的日记。
        """
        if date_str:
            path = self._diary_dir / f"{date_str}.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
            return None

        # 查找今天的日记
        today = datetime.now(JST).strftime("%Y-%m-%d")
        today_path = self._diary_dir / f"{today}.md"
        if today_path.exists():
            return today_path.read_text(encoding="utf-8")

        # 找不到则查找最新的日记
        if self._diary_dir.exists():
            diary_files = sorted(self._diary_dir.glob("*.md"), reverse=True)
            if diary_files:
                return diary_files[0].read_text(encoding="utf-8")

        return None

    def build_coaching_context(self, user_input: str) -> str | None:
        """被触发时，把日记内容作为教练式上下文返回。

        由 RuntimeEngine 追加到传给 LLM 的消息中。
        """
        if not self.should_trigger(user_input):
            return None

        content = self.get_diary_content()
        if content is None:
            return None

        return (
            f"\n\n[用户的日记]\n{content}\n\n"
            f"[教练指南]\n"
            f"请阅读这篇日记，并按以下方针回应：\n"
            f"- 不要给建议，而是「提问」和「提供新的视角」\n"
            f"- 结合日记内容，自然地表达你的感受与觉察\n"
            f"- 语气轻松，不要有压迫感\n"
            f"- 保持角色的口吻\n"
            f"- 使用简体中文回应"
        )
