"""UserContextConsolidator: analyzes conversations and updates user context files.

Analyzes conversation history using LLM (Haiku) to detect context updates,
then applies them based on risk level:
  - LOW (projects/): auto-apply
  - MEDIUM (glossary): auto-apply + report summary
  - HIGH (identity, values, core_experiences): proposal only, pending user approval

File updates use section-level partial modification (not full file overwrite).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest

_VALID_UPDATE_TYPES = {"add", "modify", "replace"}

_SECTION_RE = re.compile(r"(?=^## )", re.MULTILINE)

ANALYSIS_SYSTEM_PROMPT = """\
你是一个分析用户与 AI 角色之间对话、并生成用户上下文更新建议的助手。

用户上下文按以下层次结构管理:
- identity.md: 基本资料（姓名、居住地、年龄等）
- values.md: 价值观、职业愿景
- glossary.md: 术语表（用户特有的用语或缩写）
- core_experiences.md: 核心经历（改变人生的经历）
- projects/*.md: 进行中的项目（健康管理、工作等）

请把从对话中检测到的信息，作为对相应文件的更新建议输出。

风险等级:
- "low": projects/ 下的状态更新（自动应用）
- "medium": 追加到 glossary（自动应用并上报）
- "high": identity、values、core_experiences 的修改（需要用户批准）

update_type:
- "add": 在既有文件末尾追加章节（不需要 section 字段）
- "modify": 替换指定章节的内容（用 section 字段指定章节标题）
- "replace": 替换整个文件（包括新建文件）

重要:
- 语音输入的错别字请根据上下文修正后再写入（例：「shibuya」→「涩谷」）
- 信息过于琐碎或无需更新时返回空数组
- section 字段请使用 "## 章节名" 的形式

输出为 JSON 数组，每个元素格式如下:
[
  {
    "target_file": "projects/health.md",
    "update_type": "modify",
    "content": "## 戒咖啡因\\n状态: 第 3 天\\n",
    "risk_level": "low",
    "reason": "用户报告已戒咖啡因 3 天",
    "section": "## 戒咖啡因"
  }
]

只输出 JSON 数组，不要任何说明文字。所有文本字段请使用简体中文。
"""


class ContextUpdateRisk(Enum):
    """Risk level for context updates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ContextUpdate:
    """A single proposed update to a user context file."""

    target_file: str
    update_type: str  # "add", "modify", "replace"
    content: str
    risk_level: ContextUpdateRisk
    reason: str
    section: str | None = None


@dataclass
class ConsolidationResult:
    """Result of user context consolidation."""

    applied: list[ContextUpdate] = field(default_factory=list)
    pending_approval: list[ContextUpdate] = field(default_factory=list)
    change_log: list[str] = field(default_factory=list)


class UserContextConsolidator:
    """Analyzes conversations and updates user context files.

    Pipeline:
        1. LLM analyzes conversation history
        2. Parse and validate update proposals
        3. Classify by risk level
        4. Apply LOW/MEDIUM updates to files
        5. Keep HIGH updates as pending proposals
        6. Record change log
    """

    def __init__(
        self,
        llm: LLMAdapter,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self._model = model

    async def consolidate(
        self,
        conversation_history: list[dict],
        user_context_dir: Path,
    ) -> ConsolidationResult:
        """Analyze conversation and update user context files.

        Args:
            conversation_history: List of conversation messages.
            user_context_dir: Path to user context directory.

        Returns:
            ConsolidationResult with applied updates, pending approvals,
            and change log entries.
        """
        result = ConsolidationResult()

        if not conversation_history:
            return result

        # 1. 用 LLM 分析对话
        raw_updates = await self._analyze_conversation(conversation_history)
        if not raw_updates:
            return result

        # 2. 校验与解析
        updates = self._parse_updates(raw_updates)
        if not updates:
            return result

        # 3. 按风险等级分别处理
        for update in updates:
            if update.risk_level == ContextUpdateRisk.HIGH:
                result.pending_approval.append(update)
                result.change_log.append(
                    f"[pending] {update.target_file}: {update.reason}"
                )
            else:
                # LOW / MEDIUM: 直接应用到文件
                self._apply_update(update, user_context_dir)
                result.applied.append(update)
                result.change_log.append(
                    f"{update.target_file}: {update.reason}"
                )

        # 4. 把变更日志写入文件
        if result.change_log:
            self._write_update_log(result.change_log, user_context_dir)

        return result

    async def _analyze_conversation(
        self, conversation_history: list[dict]
    ) -> list[dict]:
        """Use LLM to analyze conversation and propose updates."""
        request = LLMRequest(
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            messages=conversation_history,
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            response = await self.llm.generate(request)
        except Exception:
            return []

        return self._parse_json_response(response.content)

    @staticmethod
    def _parse_json_response(content: str) -> list[dict]:
        """Parse JSON response, handling markdown code block wrapping."""
        text = content.strip()

        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                return []
            return parsed
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _parse_updates(raw_updates: list[dict]) -> list[ContextUpdate]:
        """Validate and convert raw dicts to ContextUpdate objects."""
        updates: list[ContextUpdate] = []

        for raw in raw_updates:
            if not isinstance(raw, dict):
                continue

            # Check required fields
            required = {"target_file", "update_type", "content", "risk_level", "reason"}
            if not required.issubset(raw.keys()):
                continue

            # Validate update_type
            if raw["update_type"] not in _VALID_UPDATE_TYPES:
                continue

            # Validate risk_level
            try:
                risk = ContextUpdateRisk(raw["risk_level"])
            except ValueError:
                continue

            updates.append(
                ContextUpdate(
                    target_file=raw["target_file"],
                    update_type=raw["update_type"],
                    content=raw["content"],
                    risk_level=risk,
                    reason=raw["reason"],
                    section=raw.get("section"),
                )
            )

        return updates

    @staticmethod
    def _apply_update(update: ContextUpdate, base_dir: Path) -> None:
        """Apply a single update to the filesystem."""
        filepath = base_dir / update.target_file

        if update.update_type == "replace":
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(update.content, encoding="utf-8")
            return

        if update.update_type == "add":
            if filepath.exists():
                existing = filepath.read_text(encoding="utf-8")
                # Append new content with a blank line separator
                new_content = existing.rstrip() + "\n\n" + update.content
                filepath.write_text(new_content, encoding="utf-8")
            else:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(update.content, encoding="utf-8")
            return

        if update.update_type == "modify":
            if not filepath.exists():
                # If file doesn't exist, create it with the content
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(update.content, encoding="utf-8")
                return

            existing = filepath.read_text(encoding="utf-8")

            if update.section:
                new_content = _replace_section(
                    existing, update.section, update.content
                )
                filepath.write_text(new_content, encoding="utf-8")
            else:
                # No section specified: replace entire file
                filepath.write_text(update.content, encoding="utf-8")

    @staticmethod
    def _write_update_log(
        log_entries: list[str], base_dir: Path
    ) -> None:
        """Append change log entries to .update_log.md."""
        log_file = base_dir / ".update_log.md"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        entry = f"\n## {now}\n"
        for log in log_entries:
            entry += f"- {log}\n"

        if log_file.exists():
            existing = log_file.read_text(encoding="utf-8")
            log_file.write_text(existing + entry, encoding="utf-8")
        else:
            header = "# User Context Update Log\n"
            log_file.write_text(header + entry, encoding="utf-8")


def _replace_section(
    document: str, section_heading: str, new_content: str
) -> str:
    """替换 markdown 文档中的指定章节。

    查找与 section_heading 匹配的章节（例如 "## 戒咖啡因"），
    并用 new_content 替换它。文档的其余部分保持不变。

    若找不到该章节，则把新内容追加到末尾。
    """
    sections = _SECTION_RE.split(document)

    if not sections:
        return new_content

    # Find and replace the matching section
    found = False
    new_sections: list[str] = []

    for section in sections:
        stripped = section.strip()
        if stripped.startswith(section_heading.strip()):
            new_sections.append(new_content)
            found = True
        else:
            new_sections.append(section)

    if not found:
        # Section not found: append at end
        new_sections.append(new_content + "\n")

    # Reconstruct the document
    result_parts: list[str] = []
    for i, section in enumerate(new_sections):
        stripped = section.strip()
        if not stripped:
            continue
        if i == 0 and not stripped.startswith("##"):
            # First part is likely the title (# Title)
            result_parts.append(stripped)
        else:
            result_parts.append(stripped)

    return "\n\n".join(result_parts) + "\n"
