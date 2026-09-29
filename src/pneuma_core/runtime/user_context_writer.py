"""UserContextWriter：自动更新 UserContext 文件 (#136)。

在会话结束时，把 LLM 输出的 user_context_updates
真正应用到文件上，并执行 git commit。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 允许写入的文件名 / 前缀
_ALLOWED_FILES = {
    "identity.md",
    "values.md",
    "core_experiences.md",
    "glossary.md",
    "relationships.md",
}
_ALLOWED_PREFIXES = ("projects/",)


class UserContextWriter:
    """把更新应用到 UserContext 文件。

    接收 LLM 输出的 user_context_updates（list[dict]），
    对目标文件执行 rewrite / append，并执行 git commit。
    """

    def __init__(self, user_context_dir: Path | str) -> None:
        self._dir = Path(user_context_dir)

    async def apply(self, updates: list[dict]) -> int:
        """应用更新，返回成功件数。

        Args:
            updates: LLM 输出的 user_context_updates 列表。

        Returns:
            成功应用的件数。
        """
        if not updates:
            return 0

        applied = 0
        reasons: list[str] = []

        for update in updates:
            file_name = update.get("file", "")
            action = update.get("action", "")
            new_content = update.get("new_content", "")
            reason = update.get("reason", "")

            # 校验文件路径
            if not self._is_allowed_file(file_name):
                logger.warning("Skipping disallowed file: %s", file_name)
                continue

            file_path = self._dir / file_name

            # 检测路径穿越
            try:
                resolved = file_path.resolve()
                dir_resolved = self._dir.resolve()
                if not str(resolved).startswith(str(dir_resolved)):
                    logger.warning("Path traversal detected: %s", file_name)
                    continue
            except (OSError, ValueError):
                logger.warning("Invalid path: %s", file_name)
                continue

            if action == "rewrite":
                section_header = update.get("section", "")
                success = self._rewrite_section(file_path, section_header, new_content)
            elif action == "append":
                success = self._append(file_path, new_content)
            else:
                logger.warning("Unknown action: %s", action)
                continue

            if success:
                applied += 1
                if reason:
                    reasons.append(reason)

        # 若有更新被应用，则执行 Git commit
        if applied > 0:
            commit_msg = "session-end: " + "; ".join(reasons) if reasons else "session-end: update"
            self._git_commit(commit_msg)

        return applied

    def _is_allowed_file(self, file_name: str) -> bool:
        """判断文件名是否在允许列表中。"""
        if file_name in _ALLOWED_FILES:
            return True
        for prefix in _ALLOWED_PREFIXES:
            if file_name.startswith(prefix) and file_name.endswith(".md"):
                return True
        return False

    def _rewrite_section(
        self, file_path: Path, section_header: str, new_content: str,
    ) -> bool:
        """重写 markdown 的某个章节。

        若找不到该章节，则退化为 append。

        Algorithm:
        1. 读取文件
        2. 查找与 section_header 完全一致的行
        3. 取得章节层级（# 的数量）
        4. 到下一个同级或更高级标题、或 EOF 为止即为该章节范围
        5. 用「标题 + new_content」替换
        6. 写回文件
        """
        if not file_path.exists():
            logger.warning("File not found for rewrite: %s", file_path)
            return False

        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # 查找章节标题所在行
        header_line_idx = None
        for i, line in enumerate(lines):
            if line.strip() == section_header.strip():
                header_line_idx = i
                break

        if header_line_idx is None:
            # 找不到该章节 → 退化为 append
            logger.info(
                "Section '%s' not found in %s, falling back to append",
                section_header, file_path.name,
            )
            return self._append_section(file_path, section_header, new_content)

        # 取得章节层级
        level = self._header_level(section_header)

        # 查找章节结束位置
        end_line_idx = len(lines)
        for i in range(header_line_idx + 1, len(lines)):
            line_stripped = lines[i].strip()
            if line_stripped.startswith("#"):
                line_level = self._header_level(line_stripped)
                if line_level <= level:
                    end_line_idx = i
                    break

        # 构建新的章节内容
        new_section_lines = [section_header, "", new_content, ""]

        # 拼接前后部分
        before = lines[:header_line_idx]
        after = lines[end_line_idx:]

        new_lines = before + new_section_lines + after

        file_path.write_text("\n".join(new_lines), encoding="utf-8")
        return True

    def _append(self, file_path: Path, content: str) -> bool:
        """在文件末尾追加。文件不存在时则新建。"""
        try:
            # 父目录不存在则创建
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                existing = file_path.read_text(encoding="utf-8")
                # 末尾若无换行则补上
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                new_text = existing + "\n" + content + "\n"
            else:
                new_text = content + "\n"

            file_path.write_text(new_text, encoding="utf-8")
            return True
        except OSError:
            logger.warning("Failed to append to %s", file_path, exc_info=True)
            return False

    def _append_section(
        self, file_path: Path, section_header: str, content: str,
    ) -> bool:
        """带章节标题地追加到文件末尾。"""
        full_content = f"{section_header}\n\n{content}"
        return self._append(file_path, full_content)

    def _git_commit(self, message: str) -> None:
        """执行 git add + commit。"""
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=str(self._dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(self._dir),
                check=True,
                capture_output=True,
            )
            logger.info("Git commit: %s", message)
        except (subprocess.SubprocessError, OSError):
            logger.warning("Git commit failed", exc_info=True)

    @staticmethod
    def _header_level(header: str) -> int:
        """返回 markdown 标题的层级（# 的数量）。"""
        stripped = header.strip()
        level = 0
        for ch in stripped:
            if ch == "#":
                level += 1
            else:
                break
        return level
