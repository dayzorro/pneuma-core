"""UserContextLoader: loads user context from external directory.

Layer structure:
  Layer 1: identity.md - Basic profile
  Layer 2: values.md - Values, career vision
  Layer 3a: glossary.md - Glossary
  Layer 3b: core_experiences.md - Core experiences (no summarization)
  Layer 4: projects/*.md - Active projects
  Layer 5: diary/YYYY-MM-DD.md - Diary entries (sorted by date descending)
  Layer 6: diary_summary.md - Past diary summary

Note: relationships.md 已并入 relations.yaml (#123)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UserContext:
    """Holds user context file contents."""

    identity: str | None = None
    values: str | None = None
    glossary: str | None = None
    core_experiences: str | None = None
    projects: dict[str, str] = field(default_factory=dict)
    diary_entries: dict[str, str] = field(default_factory=dict)
    diary_summary: str | None = None


@dataclass
class UserContextChunk:
    """A chunk of user context for RAG retrieval."""

    content: str
    layer: int
    source_file: str
    metadata: dict[str, str] = field(default_factory=dict)


# Regex to split markdown by ## headings
_SECTION_RE = re.compile(r"(?=^## )", re.MULTILINE)


class UserContextLoader:
    """Loads user context from an external directory (e.g. Obsidian Vault).

    The directory structure follows the 6-layer convention defined in Issue #68.
    Missing files or directories are gracefully handled (no errors).
    """

    def __init__(
        self,
        base_path: str | Path,
        max_chunk_chars: int = 1000,
    ) -> None:
        self.base_path = Path(base_path)
        self.max_chunk_chars = max_chunk_chars

    async def load(self) -> UserContext:
        """Load all user context files from the base directory.

        Returns an empty UserContext if the directory does not exist.
        """
        if not self.base_path.exists() or not self.base_path.is_dir():
            return UserContext()

        return UserContext(
            identity=self._read_file("identity.md"),
            values=self._read_file("values.md"),
            glossary=self._read_file("glossary.md"),
            core_experiences=self._read_file("core_experiences.md"),
            projects=self._read_directory("projects"),
            diary_entries=self._read_diary(),
            diary_summary=self._read_file("diary_summary.md"),
        )

    def _read_file(self, filename: str) -> str | None:
        """Read a single file, returning None if it doesn't exist."""
        filepath = self.base_path / filename
        if not filepath.exists() or not filepath.is_file():
            return None
        return filepath.read_text(encoding="utf-8")

    def _read_directory(self, dirname: str) -> dict[str, str]:
        """Read all .md files from a subdirectory."""
        dirpath = self.base_path / dirname
        if not dirpath.exists() or not dirpath.is_dir():
            return {}

        result: dict[str, str] = {}
        for filepath in sorted(dirpath.iterdir()):
            if filepath.is_file() and filepath.suffix == ".md":
                result[filepath.name] = filepath.read_text(encoding="utf-8")
        return result

    def _read_diary(self) -> dict[str, str]:
        """Read diary entries sorted by date descending."""
        dirpath = self.base_path / "diary"
        if not dirpath.exists() or not dirpath.is_dir():
            return {}

        entries: list[tuple[str, str]] = []
        for filepath in dirpath.iterdir():
            if filepath.is_file() and filepath.suffix == ".md":
                entries.append(
                    (filepath.name, filepath.read_text(encoding="utf-8"))
                )

        # Sort by filename (YYYY-MM-DD.md) descending
        entries.sort(key=lambda e: e[0], reverse=True)

        return dict(entries)

    def to_chunks(self, ctx: UserContext) -> list[UserContextChunk]:
        """Convert a UserContext into RAG-ready chunks.

        Splitting strategy:
        - Layer 1 (identity): split by ## sections
        - Layer 2 (values): split by ## sections
        - Layer 3a (glossary): single chunk
        - Layer 3b (core_experiences): single chunk (no summarization)
        - Layer 4 (projects): one chunk per file
        - Layer 5 (diary): one chunk per entry
        - Layer 6 (diary_summary): split by ## sections
        """
        chunks: list[UserContextChunk] = []

        # Layer 1: identity
        if ctx.identity is not None:
            chunks.extend(
                self._split_by_sections(ctx.identity, layer=1, source_file="identity.md")
            )

        # Layer 2: values
        if ctx.values is not None:
            chunks.extend(
                self._split_by_sections(ctx.values, layer=2, source_file="values.md")
            )

        # Layer 3: glossary (single chunk)
        if ctx.glossary is not None:
            chunks.extend(
                self._make_single_chunk(ctx.glossary, layer=3, source_file="glossary.md")
            )

        # Layer 3: core_experiences (single chunk, no summarization)
        if ctx.core_experiences is not None:
            chunks.extend(
                self._make_single_chunk(
                    ctx.core_experiences, layer=3, source_file="core_experiences.md"
                )
            )

        # Layer 4: projects (one chunk per file)
        for filename, content in ctx.projects.items():
            chunks.extend(
                self._make_single_chunk(
                    content, layer=4, source_file=f"projects/{filename}"
                )
            )

        # Layer 5: diary (one chunk per entry)
        for filename, content in ctx.diary_entries.items():
            date_str = filename.replace(".md", "")
            chunks.append(
                UserContextChunk(
                    content=content,
                    layer=5,
                    source_file=f"diary/{filename}",
                    metadata={"date": date_str},
                )
            )

        # Layer 6: diary_summary
        if ctx.diary_summary is not None:
            chunks.extend(
                self._split_by_sections(
                    ctx.diary_summary, layer=6, source_file="diary_summary.md"
                )
            )

        return chunks

    def _split_by_sections(
        self,
        text: str,
        layer: int,
        source_file: str,
    ) -> list[UserContextChunk]:
        """Split markdown text by ## headings into chunks.

        If a section exceeds max_chunk_chars, it is further split.
        The top-level heading (# Title) before any ## section is discarded
        if empty of meaningful content, or kept as part of the first section.
        """
        sections = _SECTION_RE.split(text)

        # Filter out empty or whitespace-only sections and the top-level heading
        meaningful_sections: list[str] = []
        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue
            # If this section starts with a single # (top-level heading) and has
            # no ## heading, it's the document title - skip it if it's just a title
            if stripped.startswith("# ") and "\n## " not in stripped:
                # Check if there's content beyond the heading line
                lines = stripped.split("\n", 1)
                if len(lines) == 1 or not lines[1].strip():
                    continue
                # There's content after the heading, keep it
            meaningful_sections.append(stripped)

        if not meaningful_sections:
            # Fallback: treat the entire text as one chunk
            stripped_text = text.strip()
            if stripped_text:
                return self._split_long_text(stripped_text, layer, source_file)
            return []

        chunks: list[UserContextChunk] = []
        for section in meaningful_sections:
            chunks.extend(self._split_long_text(section, layer, source_file))

        return chunks

    def _make_single_chunk(
        self,
        text: str,
        layer: int,
        source_file: str,
    ) -> list[UserContextChunk]:
        """Create a single chunk, splitting if too long."""
        stripped = text.strip()
        if not stripped:
            return []
        return self._split_long_text(stripped, layer, source_file)

    def _split_long_text(
        self,
        text: str,
        layer: int,
        source_file: str,
    ) -> list[UserContextChunk]:
        """Split text into chunks of at most max_chunk_chars."""
        if len(text) <= self.max_chunk_chars:
            return [
                UserContextChunk(content=text, layer=layer, source_file=source_file)
            ]

        chunks: list[UserContextChunk] = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_chars
            chunk_text = text[start:end]
            chunks.append(
                UserContextChunk(
                    content=chunk_text, layer=layer, source_file=source_file
                )
            )
            start = end

        return chunks


def load_user_goals(goals_path: Path) -> GoalTree | None:
    """从 vault/user/goals.yaml 读取用户的 GoalTree。

    Returns:
        GoalTree if file exists, None otherwise.
    """
    if not goals_path.exists():
        return None

    import yaml

    from pneuma_core.models.goals import GoalTree, Objective, Task, Vision

    data = yaml.safe_load(goals_path.read_text(encoding="utf-8"))
    if not data:
        return None

    visions = [
        Vision(
            id=v["id"],
            character_id="user",
            content=v["content"],
        )
        for v in data.get("visions", [])
    ]

    objectives = [
        Objective(
            id=o["id"],
            character_id="user",
            vision_id=o["vision_id"],
            content=o["content"],
            status=o.get("status", "active"),
            progress=o.get("progress", 0.0),
        )
        for o in data.get("objectives", [])
    ]

    tasks = [
        Task(
            id=t["id"],
            character_id="user",
            objective_id=t["objective_id"],
            content=t["content"],
            status=t.get("status", "pending"),
        )
        for t in data.get("tasks", [])
    ]

    return GoalTree(visions=visions, objectives=objectives, tasks=tasks)


def load_user_relations(relations_path: Path) -> list[Relation] | None:
    """从 vault/user/relationships.yaml 读取用户的 Relations。

    Returns:
        list[Relation] if file exists, None otherwise.
    """
    if not relations_path.exists():
        return None

    import yaml

    from pneuma_core.models.relation import Relation as RelationModel

    data = yaml.safe_load(relations_path.read_text(encoding="utf-8"))
    if not data:
        return None

    from datetime import datetime as dt
    from datetime import timezone as tz

    now = dt.now(tz.utc)

    return [
        RelationModel(
            id=r["id"],
            owner_id="user",
            target_id=r["target_id"],
            target_name=r["target_name"],
            relationship_type=r["relationship_type"],
            description=r["description"],
            closeness=r.get("closeness", 0.5),
            trust=r.get("trust", 0.5),
            updated_at=now,
            notes=r.get("notes"),
        )
        for r in data.get("relations", [])
    ]
