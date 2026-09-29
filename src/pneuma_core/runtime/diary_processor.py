"""Diary processing pipeline: daily processing and bootstrap.

DiaryProcessor: processes new diary entries incrementally (daily trigger).
BootstrapProcessor: bulk-processes all past diaries to generate initial context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest

# Regex for YYYY-MM-DD.md filenames
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")

# State file to track last processed date
_STATE_FILE = ".diary_processor_state.json"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """\
你是一个阅读并总结日记的助手。
请用 1-2 行简洁地总结日记内容。

输出为如下 JSON 格式:
{
  "summary": "YYYY-MM-DD: 摘要内容"
}

只输出 JSON，不要任何说明文字。
摘要内容请使用简体中文。
"""

_EXTRACTION_SYSTEM_PROMPT = """\
你是一个从日记中抽取信息的助手。
请从日记中抽取以下 4 类信息。

1. relationships: 新出现的人物或关系的变化
2. glossary: 新的术语、缩写、专业词汇
3. projects: 项目或话题的更新
4. core_experiences: 足以改变人生的重要经历（高风险：仅提出建议）

输出为如下 JSON 格式:
{
  "relationships": [{"name": "人名", "context": "关系的说明"}],
  "glossary": [{"term": "术语", "definition": "定义"}],
  "projects": [{"name": "项目名", "description": "说明", "status": "active"}],
  "core_experiences": [{"date": "YYYY-MM-DD", "event": "事件", "impact": "影响"}]
}

没有对应信息的类别请返回空数组。
只输出 JSON。所有文本字段请使用简体中文。
"""

_BOOTSTRAP_SUMMARY_PROMPT = """\
你是一个阅读并总结月度日记的助手。
请阅读以下日记群，总结该月的活动。

输出为如下 JSON 格式:
{
  "summary": "YYYY年M月: 摘要内容"
}

只输出 JSON。摘要内容请使用简体中文。
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiaryProcessResult:
    """Result of daily diary processing."""

    summaries: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    glossary: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    core_experience_proposals: list[dict] = field(default_factory=list)
    processed_dates: list[str] = field(default_factory=list)


@dataclass
class BootstrapResult:
    """Result of bootstrap processing."""

    diary_count: int = 0
    month_count: int = 0
    summaries_generated: int = 0
    core_experiences_found: int = 0
    projects_found: int = 0
    relationships_found: int = 0
    glossary_found: int = 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_json_response(content: str) -> dict:
    """Parse JSON response, handling markdown code block wrapping."""
    text = content.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_date_from_filename(filename: str) -> date | None:
    """Parse a date from a YYYY-MM-DD.md filename."""
    m = _DATE_RE.match(filename)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return None


def _list_diary_files(diary_dir: Path) -> list[tuple[date, Path]]:
    """List all valid diary files with their dates, sorted chronologically."""
    if not diary_dir.exists() or not diary_dir.is_dir():
        return []

    entries: list[tuple[date, Path]] = []
    for f in diary_dir.iterdir():
        if not f.is_file():
            continue
        d = _parse_date_from_filename(f.name)
        if d is not None:
            entries.append((d, f))

    entries.sort(key=lambda e: e[0])
    return entries


# ---------------------------------------------------------------------------
# DiaryProcessor (daily incremental processing)
# ---------------------------------------------------------------------------

class DiaryProcessor:
    """Processes new diary entries incrementally.

    Pipeline:
        1. Detect new diary files since last processed date
        2. Read and summarize each entry
        3. Update diary_summary.md
        4. Extract relationships, glossary, projects, core_experiences
        5. Record last processed date
    """

    def __init__(
        self,
        llm: LLMAdapter,
        diary_dir: Path,
        context_dir: Path,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.diary_dir = diary_dir
        self.context_dir = context_dir
        self._model = model
        self._state_path = context_dir / _STATE_FILE

    def detect_new_entries(self, last_processed_date: date | None = None) -> list[Path]:
        """Detect diary files newer than last_processed_date.

        If last_processed_date is None, reads from persisted state.
        If no state exists either, returns all diary files.
        """
        if last_processed_date is None:
            last_processed_date = self.get_last_processed_date()

        all_entries = _list_diary_files(self.diary_dir)

        if last_processed_date is None:
            return [path for _, path in all_entries]

        return [path for d, path in all_entries if d > last_processed_date]

    async def process_entries(self, entry_paths: list[Path]) -> DiaryProcessResult:
        """Process a list of diary entries.

        1. Summarize each entry and update diary_summary.md
        2. Extract information for context layers
        3. Record processed dates
        """
        result = DiaryProcessResult()

        if not entry_paths:
            return result

        # Sort by date
        dated_entries: list[tuple[date, Path]] = []
        for path in entry_paths:
            d = _parse_date_from_filename(path.name)
            if d is not None:
                dated_entries.append((d, path))
        dated_entries.sort(key=lambda e: e[0])

        # Process each entry
        all_diary_text = []
        for entry_date, path in dated_entries:
            content = path.read_text(encoding="utf-8")
            date_str = entry_date.isoformat()

            # 1. Summarize
            summary = await self._summarize_entry(date_str, content)
            if summary:
                result.summaries.append(summary)

            all_diary_text.append(f"## {date_str}\n{content}")
            result.processed_dates.append(date_str)

        # 2. Update diary_summary.md
        if result.summaries:
            self._update_summary_file(result.summaries)

        # 3. Extract context information (if entries exist)
        if all_diary_text:
            extraction = await self._extract_context("\n\n".join(all_diary_text))
            result.relationships = extraction.get("relationships", [])
            result.glossary = extraction.get("glossary", [])
            result.projects = extraction.get("projects", [])
            result.core_experience_proposals = extraction.get("core_experiences", [])

        # 4. Record last processed date
        if dated_entries:
            last_date = dated_entries[-1][0]
            self._save_last_processed_date(last_date)

        return result

    def get_last_processed_date(self) -> date | None:
        """Get the last processed date from persistent state."""
        if not self._state_path.exists():
            return None
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            date_str = state.get("last_processed_date")
            if date_str:
                return date.fromisoformat(date_str)
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _save_last_processed_date(self, d: date) -> None:
        """Persist the last processed date."""
        state: dict = {}
        if self._state_path.exists():
            try:
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                pass
        state["last_processed_date"] = d.isoformat()
        self._state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def _summarize_entry(self, date_str: str, content: str) -> str | None:
        """Use LLM to summarize a single diary entry."""
        request = LLMRequest(
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"日付: {date_str}\n\n{content}"}],
            model=self._model,
            temperature=0.3,
            max_tokens=512,
        )
        try:
            response = await self.llm.generate(request)
        except Exception:
            return None

        parsed = _parse_json_response(response.content)
        return parsed.get("summary")

    async def _extract_context(self, diary_text: str) -> dict:
        """Use LLM to extract context information from diary text."""
        request = LLMRequest(
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": diary_text}],
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            response = await self.llm.generate(request)
        except Exception:
            return {}

        return _parse_json_response(response.content)

    def _update_summary_file(self, summaries: list[str]) -> None:
        """Update diary_summary.md with new summaries."""
        summary_path = self.context_dir / "diary_summary.md"

        new_lines = "\n".join(f"- {s}" for s in summaries)

        if summary_path.exists():
            existing = summary_path.read_text(encoding="utf-8")
            updated = existing.rstrip() + "\n\n" + new_lines + "\n"
        else:
            updated = "# 日记摘要\n\n" + new_lines + "\n"

        summary_path.write_text(updated, encoding="utf-8")


# ---------------------------------------------------------------------------
# BootstrapProcessor (bulk initial processing)
# ---------------------------------------------------------------------------

class BootstrapProcessor:
    """Bulk-processes all past diaries to generate initial context.

    Pipeline (B案: per-month extraction from raw text):
        1. Read all diary files in chronological order
        2. Chunk by month
        3. For each month:
           a. Generate monthly summary -> diary_summary.md
           b. Extract info from raw text (relationships, glossary, projects, core_experiences)
        4. Merge/dedup all monthly extractions
        5. Write output files (diary_summary.md, core_experiences.md, projects/, relations_draft.yaml, glossary.md)
    """

    def __init__(
        self,
        llm: LLMAdapter,
        diary_dir: Path,
        output_dir: Path,
        model: str | None = None,
        extraction_model: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.diary_dir = diary_dir
        self.output_dir = output_dir
        self._model = model
        self._extraction_model = extraction_model if extraction_model is not None else model
        self._progress_callback = progress_callback
        self._monthly_summaries: dict[str, str] = {}

    def read_all_entries(self) -> list[tuple[str, str]]:
        """Read all diary files in chronological order.

        Returns list of (date_string, content) tuples sorted ascending.
        """
        all_entries = _list_diary_files(self.diary_dir)
        return [
            (d.isoformat(), path.read_text(encoding="utf-8"))
            for d, path in all_entries
        ]

    def chunk_by_month(
        self, entries: list[tuple[str, str]]
    ) -> dict[str, list[tuple[str, str]]]:
        """Group diary entries by month (YYYY-MM key).

        Args:
            entries: List of (date_string, content) tuples.

        Returns:
            Dict mapping "YYYY-MM" to list of entries for that month.
        """
        chunks: dict[str, list[tuple[str, str]]] = {}
        for date_str, content in entries:
            month_key = date_str[:7]  # "YYYY-MM"
            if month_key not in chunks:
                chunks[month_key] = []
            chunks[month_key].append((date_str, content))
        return chunks

    async def generate_summaries(self) -> dict[str, str]:
        """Generate monthly summaries and write diary_summary.md.

        Returns dict mapping month key to summary text.
        """
        entries = self.read_all_entries()
        chunks = self.chunk_by_month(entries)

        total = len(chunks)
        current = 0

        summaries: dict[str, str] = {}
        for month_key in sorted(chunks.keys()):
            month_entries = chunks[month_key]

            # Combine all entries for this month
            combined = "\n\n".join(
                f"## {d}\n{c}" for d, c in month_entries
            )

            # LLM summarize
            request = LLMRequest(
                system_prompt=_BOOTSTRAP_SUMMARY_PROMPT,
                messages=[{"role": "user", "content": combined}],
                model=self._model,
                temperature=0.3,
                max_tokens=1024,
            )
            try:
                response = await self.llm.generate(request)
                parsed = _parse_json_response(response.content)
                summary = parsed.get("summary", f"{month_key}: (要約生成失敗)")
            except Exception:
                summary = f"{month_key}: (要約生成失敗)"

            summaries[month_key] = summary
            current += 1

            if self._progress_callback:
                self._progress_callback(current, total, f"{month_key} 摘要完成")

        self._monthly_summaries = summaries

        # Write diary_summary.md
        self._write_summary_file(summaries)

        return summaries

    async def extract_monthly_info(
        self, month_key: str, raw_entries: list[tuple[str, str]]
    ) -> dict:
        """Extract information from raw diary entries for a single month.

        Args:
            month_key: The month key (e.g. "2026-01").
            raw_entries: List of (date_string, content) tuples for this month.

        Returns:
            Dict with keys: relationships, glossary, projects, core_experiences.
        """
        empty_result = {
            "relationships": [],
            "glossary": [],
            "projects": [],
            "core_experiences": [],
        }

        # Combine all raw diary entries for this month
        combined = "\n\n".join(
            f"## {d}\n{c}" for d, c in raw_entries
        )

        request = LLMRequest(
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": combined}],
            model=self._extraction_model,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            response = await self.llm.generate(request)
            parsed = _parse_json_response(response.content)
            return {
                "relationships": parsed.get("relationships", []),
                "glossary": parsed.get("glossary", []),
                "projects": parsed.get("projects", []),
                "core_experiences": parsed.get("core_experiences", []),
            }
        except Exception:
            return empty_result

    def _merge_extractions(self, all_extractions: list[dict]) -> dict:
        """Merge and deduplicate all monthly extraction results.

        - Relationships: dedup by name (keep latest context)
        - Glossary: dedup by term (keep latest definition)
        - Projects: dedup by name (merge descriptions)
        - Core experiences: keep all (no dedup, user will curate)
        """
        merged: dict = {
            "relationships": [],
            "glossary": [],
            "projects": [],
            "core_experiences": [],
        }

        if not all_extractions:
            return merged

        # Relationships: dedup by name, keep latest context
        rel_map: dict[str, dict] = {}
        for extraction in all_extractions:
            for rel in extraction.get("relationships", []):
                name = rel.get("name", "")
                if name:
                    rel_map[name] = rel
        merged["relationships"] = list(rel_map.values())

        # Glossary: dedup by term, keep latest definition
        glo_map: dict[str, dict] = {}
        for extraction in all_extractions:
            for item in extraction.get("glossary", []):
                term = item.get("term", "")
                if term:
                    glo_map[term] = item
        merged["glossary"] = list(glo_map.values())

        # Projects: dedup by name, merge descriptions
        proj_map: dict[str, dict] = {}
        for extraction in all_extractions:
            for proj in extraction.get("projects", []):
                name = proj.get("name", "")
                if not name:
                    continue
                if name in proj_map:
                    # Merge descriptions
                    existing_desc = proj_map[name].get("description", "")
                    new_desc = proj.get("description", "")
                    if new_desc and new_desc != existing_desc:
                        proj_map[name]["description"] = f"{existing_desc} / {new_desc}"
                    # Keep latest status
                    if proj.get("status"):
                        proj_map[name]["status"] = proj["status"]
                else:
                    proj_map[name] = dict(proj)
        merged["projects"] = list(proj_map.values())

        # Core experiences: keep all (no dedup)
        for extraction in all_extractions:
            merged["core_experiences"].extend(
                extraction.get("core_experiences", [])
            )

        return merged

    async def run(self) -> BootstrapResult:
        """Execute the full bootstrap pipeline (B案).

        For each month:
          1. Generate summary from raw text -> diary_summary.md
          2. Extract info from raw text -> per-month extraction
        After all months:
          3. Merge/dedup all extractions
          4. Write all output files
        """
        result = BootstrapResult()

        entries = self.read_all_entries()
        result.diary_count = len(entries)
        chunks = self.chunk_by_month(entries)
        result.month_count = len(chunks)

        if result.diary_count == 0:
            return result

        # Total progress steps = months * 2 (summary + extraction)
        total_steps = len(chunks) * 2
        current_step = 0

        summaries: dict[str, str] = {}
        all_extractions: list[dict] = []

        for month_key in sorted(chunks.keys()):
            month_entries = chunks[month_key]

            # Step a: Generate monthly summary
            combined = "\n\n".join(
                f"## {d}\n{c}" for d, c in month_entries
            )

            request = LLMRequest(
                system_prompt=_BOOTSTRAP_SUMMARY_PROMPT,
                messages=[{"role": "user", "content": combined}],
                model=self._model,
                temperature=0.3,
                max_tokens=1024,
            )
            try:
                response = await self.llm.generate(request)
                parsed = _parse_json_response(response.content)
                summary = parsed.get("summary", f"{month_key}: (要約生成失敗)")
            except Exception:
                summary = f"{month_key}: (要約生成失敗)"

            summaries[month_key] = summary
            current_step += 1

            if self._progress_callback:
                self._progress_callback(
                    current_step, total_steps, f"{month_key} 要約完了"
                )

            # Step b: Extract info from raw text
            extraction = await self.extract_monthly_info(month_key, month_entries)
            all_extractions.append(extraction)
            current_step += 1

            if self._progress_callback:
                self._progress_callback(
                    current_step, total_steps, f"{month_key} 情報抽出完了"
                )

        self._monthly_summaries = summaries
        result.summaries_generated = len(summaries)

        # Merge/dedup all monthly extractions
        merged = self._merge_extractions(all_extractions)

        # Write all output files
        self._write_summary_file(summaries)
        self._write_core_experiences_draft(merged["core_experiences"])
        self._write_project_drafts(merged["projects"])
        self._write_relationships_draft(merged["relationships"])
        self._write_glossary_draft(merged["glossary"])

        result.core_experiences_found = len(merged["core_experiences"])
        result.projects_found = len(merged["projects"])
        result.relationships_found = len(merged["relationships"])
        result.glossary_found = len(merged["glossary"])

        return result

    # --- File writing helpers ---

    def _write_summary_file(self, summaries: dict[str, str]) -> None:
        """写入 diary_summary.md。"""
        lines = ["# 日记摘要\n"]
        for month_key in sorted(summaries.keys()):
            lines.append(f"## {month_key}\n")
            lines.append(f"- {summaries[month_key]}\n")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "diary_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_core_experiences_draft(self, experiences: list[dict]) -> None:
        """把 core_experiences.md 写为草稿。"""
        lines = [
            "# 核心经历（候选）\n",
            "> 本文件是自动生成的候选清单。请确认内容并编辑。\n",
        ]
        for exp in experiences:
            event = exp.get("event", "未知")
            impact = exp.get("impact", "")
            exp_date = exp.get("date", "未知")
            lines.append(f"## {event}\n")
            lines.append(f"- 日期: {exp_date}")
            lines.append(f"- 影响: {impact}\n")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "core_experiences.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_project_drafts(self, projects: list[dict]) -> None:
        """把 projects/*.md 写为草稿。"""
        projects_dir = self.output_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)

        for proj in projects:
            name = proj.get("name", "未命名")
            description = proj.get("description", "")
            status = proj.get("status", "active")

            # 清理文件名
            safe_name = re.sub(r"[^\w\-]", "-", name.lower())
            filename = f"{safe_name}.md"

            content = (
                f"# {name}\n\n"
                f"状态: {status}\n\n"
                f"## 概要\n\n{description}\n"
            )
            (projects_dir / filename).write_text(content, encoding="utf-8")

    def _write_relationships_draft(self, relationships: list[dict]) -> None:
        """写入 relations_draft.yaml（YAML 格式，便于并入 relations.yaml）。"""
        import yaml

        draft = {
            "_note": "这是自动生成的候选清单。请确认内容后并入 relations.yaml。",
            "relations": [
                {"name": rel.get("name", "未知"), "context": rel.get("context", "")}
                for rel in relationships
            ],
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "relations_draft.yaml").write_text(
            yaml.dump(draft, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    def _write_glossary_draft(self, glossary: list[dict]) -> None:
        """把 glossary.md 写为草稿。"""
        lines = [
            "# 术语表\n",
            "> 本文件是自动生成的候选清单。请确认内容并编辑。\n",
        ]
        for item in glossary:
            term = item.get("term", "未知")
            definition = item.get("definition", "")
            lines.append(f"## {term}\n")
            lines.append(f"- {definition}\n")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "glossary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
