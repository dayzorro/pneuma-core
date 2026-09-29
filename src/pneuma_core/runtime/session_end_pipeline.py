"""SessionEndPipeline：会话结束时的整合分析 (#130)。

一次 LLM 调用即可整合更新 episodic/semantic/relationship。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pneuma_core.llm.adapter import LLMAdapter
    from pneuma_core.llm.embedding import EmbeddingService
    from pneuma_core.models.memory import SemanticMemory
    from pneuma_core.models.relation import Relation
    from pneuma_core.runtime.session import ConversationSession
    from pneuma_core.storage.backend import StorageBackend

from pathlib import Path

from pneuma_core.llm.adapter import LLMRequest
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory as SemanticMemoryModel

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

_MD_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)

_SYSTEM_PROMPT = """\
你是一个 AI 角色的记忆管理助手。
请分析以下会话，并以 JSON 形式输出应当保存的信息。

## 输出格式
```json
{
  "episodic_memories": [
    {
      "content": "对具体事件的描述",
      "emotional_valence": -1.0〜1.0,
      "importance": 0.0〜1.0
    }
  ],
  "semantic_updates": [
    {
      "action": "add" | "modify" | "delete",
      "content": "泛化后的知识（add/modify 时）",
      "confidence": 0.0〜1.0（add/modify 时）,
      "memory_id": "既有 ID（modify/delete 时）",
      "reason": "变更理由（delete 时）"
    }
  ],
  "user_context_updates": [
    {
      "file": "identity.md | values.md | core_experiences.md | glossary.md | relationships.md | projects/*.md",
      "section": "## 章节名（仅 rewrite 时必填）",
      "action": "rewrite | append",
      "new_content": "新的章节内容，或要追加到文件末尾的内容",
      "reason": "更新理由的简要说明"
    }
  ],
  "relationship_changes": [
    {
      "relation_id": "关系 ID",
      "closeness_delta": -0.1〜0.1,
      "trust_delta": -0.1〜0.1
    }
  ]
}
```

## 规则
- episodic_memories: 记录本次会话中发生的重要事件。过于琐碎的内容请省略。
- semantic_updates: 从对话中获得的泛化知识。也包括对既有知识的修正，以及删除已过时的知识。
- user_context_updates: 仅当用户的生活、工作、兴趣、价值观等发生重要变化时才更新。细微变化请忽略。rewrite 表示整段重写某个 markdown 章节。append 表示在文件末尾追加。
- relationship_changes: 根据对话内容判断关系的变化。若没有变化则为空数组。
- 所有 content / new_content / reason 等文本字段必须使用简体中文。
- 若没有对应内容，请返回空数组。
"""


@dataclass
class SessionEndResult:
    """会话结束流水线的执行结果。"""

    success: bool
    episodic_memories_saved: int = 0
    semantic_updates_applied: int = 0
    user_context_updates: int = 0
    relationship_changes: int = 0


class SessionEndPipeline:
    """会话结束时的整合分析流水线。

    一次 LLM 调用生成 4 类输出，并分别应用到对应存储。
    """

    def __init__(
        self,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        memory_store: StorageBackend,
        storage: StorageBackend,
        user_context_dir: str | None = None,
        model: str = "claude-opus-4-6",
    ) -> None:
        self._llm = llm
        self._embedding = embedding_service
        self._memory_store = memory_store
        self._storage = storage
        self._user_context_dir = user_context_dir
        self._model = model

    async def run(
        self,
        session: ConversationSession,
        character_id: str,
        existing_semantics: list[SemanticMemory] | None = None,
        existing_relations: list[Relation] | None = None,
    ) -> SessionEndResult:
        """执行会话结束分析。"""
        # Empty session → skip LLM
        if not session.messages:
            return SessionEndResult(success=True)

        try:
            analysis = await self._call_llm(
                session, character_id, existing_semantics, existing_relations,
            )
        except Exception:
            logger.warning("Session-end pipeline LLM call failed", exc_info=True)
            return SessionEndResult(success=False)

        # Apply results
        ep_count = await self._apply_episodic(
            analysis.get("episodic_memories", []),
            character_id,
            session.session_id,
        )
        sem_count = await self._apply_semantic(
            analysis.get("semantic_updates", []),
            character_id,
        )
        uc_count = await self._apply_user_context(
            analysis.get("user_context_updates", []),
        )
        rel_count = await self._apply_relationships(
            analysis.get("relationship_changes", []),
            existing_relations or [],
        )

        return SessionEndResult(
            success=True,
            episodic_memories_saved=ep_count,
            semantic_updates_applied=sem_count,
            user_context_updates=uc_count,
            relationship_changes=rel_count,
        )

    async def _call_llm(
        self,
        session: ConversationSession,
        character_id: str,
        existing_semantics: list[SemanticMemory] | None,
        existing_relations: list[Relation] | None,
    ) -> dict:
        """调用 LLM 获取分析结果。"""
        system_prompt = _SYSTEM_PROMPT

        # Inject UserContext files
        if self._user_context_dir:
            uc_text = self._load_user_context_text()
            if uc_text:
                system_prompt += "\n\n## 用户上下文（当前内容）\n" + uc_text

        # Inject existing semantics
        if existing_semantics:
            sem_lines = [
                f"- [{m.id}] {m.content} (confidence={m.confidence})"
                for m in existing_semantics
            ]
            system_prompt += "\n\n## 既有的语义记忆\n" + "\n".join(sem_lines)

        # Inject existing relations
        if existing_relations:
            rel_lines = [
                f"- [{r.id}] {r.target_name} ({r.relationship_type}): "
                f"closeness={r.closeness}, trust={r.trust}"
                for r in existing_relations
            ]
            system_prompt += "\n\n## 既有关系\n" + "\n".join(rel_lines)

        request = LLMRequest(
            system_prompt=system_prompt,
            messages=session.messages,
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )

        response = await self._llm.generate(request)
        return self._parse_json(response.content)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """从 LLM 响应中解析 JSON。"""
        # Strip markdown code blocks
        match = _MD_CODE_BLOCK_RE.match(content.strip())
        if match:
            content = match.group(1)

        return json.loads(content)

    async def _apply_episodic(
        self,
        memories: list[dict],
        character_id: str,
        session_id: str,
    ) -> int:
        """保存情节记忆。"""
        if not memories:
            return 0

        contents = [m["content"] for m in memories]
        embeddings = await self._embedding.embed_batch(contents)

        now = datetime.now(JST)
        count = 0
        for mem_data, embedding in zip(memories, embeddings):
            memory = EpisodicMemory(
                id=f"ep-{uuid.uuid4().hex[:12]}",
                character_id=character_id,
                content=mem_data["content"],
                timestamp=now,
                emotional_valence=float(mem_data.get("emotional_valence", 0.0)),
                importance=float(mem_data.get("importance", 0.5)),
                conversation_id=session_id,
                embedding=embedding,
            )
            await self._memory_store.save_episodic_memory(memory)
            count += 1

        return count

    async def _apply_semantic(
        self,
        updates: list[dict],
        character_id: str,
    ) -> int:
        """新增/更新/删除语义记忆。"""
        count = 0
        for update in updates:
            action = update.get("action", "add")

            if action == "add":
                content = update["content"]
                embeddings = await self._embedding.embed_batch([content])
                memory = SemanticMemoryModel(
                    id=f"sem-{uuid.uuid4().hex[:12]}",
                    character_id=character_id,
                    content=content,
                    confidence=float(update.get("confidence", 0.5)),
                    embedding=embeddings[0],
                )
                await self._memory_store.save_semantic_memory(memory)
                count += 1

            elif action == "modify":
                memory_id = update["memory_id"]
                content = update["content"]
                confidence = float(update.get("confidence", 0.5))
                updated = SemanticMemoryModel(
                    id=memory_id,
                    character_id=character_id,
                    content=content,
                    confidence=confidence,
                )
                await self._memory_store.update_semantic_memory(updated)
                count += 1

            elif action == "delete":
                memory_id = update["memory_id"]
                await self._memory_store.delete_semantic_memory(memory_id)
                count += 1

        return count

    def _load_user_context_text(self) -> str:
        """从 UserContext 目录读取文件并转为文本。"""
        if not self._user_context_dir:
            return ""

        uc_dir = Path(self._user_context_dir)
        if not uc_dir.exists() or not uc_dir.is_dir():
            return ""

        parts: list[str] = []

        # Read top-level .md files
        for md_file in sorted(uc_dir.iterdir()):
            if md_file.is_file() and md_file.suffix == ".md":
                try:
                    content = md_file.read_text(encoding="utf-8")
                    parts.append(f"### {md_file.name}\n{content}")
                except OSError:
                    continue

        # Read projects/ subdirectory
        projects_dir = uc_dir / "projects"
        if projects_dir.exists() and projects_dir.is_dir():
            for md_file in sorted(projects_dir.iterdir()):
                if md_file.is_file() and md_file.suffix == ".md":
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        parts.append(f"### projects/{md_file.name}\n{content}")
                    except OSError:
                        continue

        return "\n\n".join(parts)

    async def _apply_user_context(self, updates: list[dict]) -> int:
        """应用 UserContext 的更新。"""
        if not updates or not self._user_context_dir:
            return 0

        from pneuma_core.runtime.user_context_writer import UserContextWriter

        writer = UserContextWriter(self._user_context_dir)
        try:
            return await writer.apply(updates)
        except Exception:
            logger.warning("UserContext update failed", exc_info=True)
            return 0

    async def _apply_relationships(
        self,
        changes: list[dict],
        existing_relations: list[Relation],
    ) -> int:
        """应用关系的增量变化。"""
        if not changes:
            return 0

        rel_map = {r.id: r for r in existing_relations}
        count = 0

        for change in changes:
            rel_id = change["relation_id"]
            rel = rel_map.get(rel_id)
            if rel is None:
                continue

            closeness_delta = float(change.get("closeness_delta", 0.0))
            trust_delta = float(change.get("trust_delta", 0.0))

            new_closeness = max(0.0, min(1.0, rel.closeness + closeness_delta))
            new_trust = max(0.0, min(1.0, rel.trust + trust_delta))

            rel.closeness = new_closeness
            rel.trust = new_trust
            rel.updated_at = datetime.now(JST)

            await self._storage.save_relation(rel)
            count += 1

        return count
