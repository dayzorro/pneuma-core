"""SQLiteStorageBackend: persistent storage using aiosqlite."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.models.change_record import ChangeRecord
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.relation import Relation
from pneuma_core.models.todo import TodoItem
from pneuma_core.models.values import Values

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    profile TEXT,
    appearance TEXT,
    speaking_style TEXT,
    background TEXT,
    personality_description TEXT,
    values_description TEXT,
    openness REAL, conscientiousness REAL, extraversion REAL,
    agreeableness REAL, neuroticism REAL,
    self_transcendence REAL, self_enhancement REAL,
    openness_to_change REAL, conservation REAL,
    created_at TIMESTAMP, updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS emotional_states (
    character_id TEXT PRIMARY KEY,
    pleasure REAL, arousal REAL, dominance REAL,
    emotion_label TEXT, situation TEXT
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    conversation_id TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    emotional_valence REAL,
    importance REAL,
    embedding TEXT
);

CREATE TABLE IF NOT EXISTS semantic_memories (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    content TEXT,
    confidence REAL,
    embedding TEXT
);

CREATE TABLE IF NOT EXISTS semantic_memory_sources (
    semantic_memory_id TEXT,
    episodic_memory_id TEXT,
    PRIMARY KEY (semantic_memory_id, episodic_memory_id)
);

CREATE TABLE IF NOT EXISTS visions (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    content TEXT
);

CREATE TABLE IF NOT EXISTS objectives (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    vision_id TEXT,
    content TEXT,
    status TEXT,
    progress REAL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    objective_id TEXT,
    content TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS change_log (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    type TEXT,
    before_value TEXT,
    after_value TEXT,
    reason TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS todos (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT 'someday',
    kind TEXT NOT NULL DEFAULT 'must',
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 2,
    due_date TEXT,
    recurrence TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    owner_id TEXT NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    description TEXT NOT NULL,
    closeness REAL NOT NULL DEFAULT 0.5,
    trust REAL NOT NULL DEFAULT 0.5,
    updated_at TEXT NOT NULL,
    notes TEXT
);
"""


class SQLiteStorageBackend:
    """SQLite 存储后端。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """打开 DB 连接并创建表结构。"""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._migrate()

    async def _migrate(self) -> None:
        """为已有表补充缺失字段的迁移处理。"""
        migrations = [
            ("todos", "owner_id", "ALTER TABLE todos ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'user'"),
        ]
        for table, column, sql in migrations:
            try:
                cursor = await self._conn.execute(f"PRAGMA table_info({table})")
                columns = {row[1] for row in await cursor.fetchall()}
                if column not in columns:
                    await self._conn.execute(sql)
                    await self._conn.commit()
            except Exception:
                pass

    async def close(self) -> None:
        """关闭 DB 连接。"""
        if self._db:
            await self._db.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    async def list_tables(self) -> set[str]:
        """返回表名列表（供测试用）。"""
        cursor = await self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        return {row[0] for row in rows}

    # --- Character ---

    async def save_character(self, character: Character) -> None:
        c = character
        p = c.personality
        v = c.values
        now = datetime.now().isoformat()
        await self._conn.execute(
            """INSERT OR REPLACE INTO characters
            (id, name, profile, appearance, speaking_style, background,
             personality_description, values_description,
             openness, conscientiousness, extraversion, agreeableness, neuroticism,
             self_transcendence, self_enhancement, openness_to_change, conservation,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (c.id, c.name, c.profile, c.appearance, c.speaking_style, c.background,
             c.personality_description, c.values_description,
             p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism,
             v.self_transcendence, v.self_enhancement, v.openness_to_change, v.conservation,
             now, now),
        )
        await self._conn.commit()

    async def get_character(self, character_id: str) -> Character | None:
        cursor = await self._conn.execute(
            "SELECT * FROM characters WHERE id = ?", (character_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Character(
            id=row["id"], name=row["name"],
            personality=Personality(
                openness=row["openness"], conscientiousness=row["conscientiousness"],
                extraversion=row["extraversion"], agreeableness=row["agreeableness"],
                neuroticism=row["neuroticism"],
            ),
            values=Values(
                self_transcendence=row["self_transcendence"],
                self_enhancement=row["self_enhancement"],
                openness_to_change=row["openness_to_change"],
                conservation=row["conservation"],
            ),
            profile=row["profile"], appearance=row["appearance"],
            speaking_style=row["speaking_style"], background=row["background"],
            personality_description=row["personality_description"],
            values_description=row["values_description"],
        )

    async def list_characters(self) -> list[Character]:
        cursor = await self._conn.execute("SELECT id FROM characters")
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            char = await self.get_character(row["id"])
            if char:
                result.append(char)
        return result

    # --- Memory ---

    async def save_episodic_memory(self, memory: EpisodicMemory) -> None:
        embedding_json = json.dumps(memory.embedding) if memory.embedding else None
        await self._conn.execute(
            """INSERT INTO episodic_memories
            (id, character_id, conversation_id, content, timestamp,
             emotional_valence, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory.id, memory.character_id, memory.conversation_id,
             memory.content, memory.timestamp.isoformat(),
             memory.emotional_valence, memory.importance, embedding_json),
        )
        await self._conn.commit()

    async def get_episodic_memories(self, character_id: str) -> list[EpisodicMemory]:
        cursor = await self._conn.execute(
            "SELECT * FROM episodic_memories WHERE character_id = ?", (character_id,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_episodic(row) for row in rows]

    async def find_similar_memories(
        self, character_id: str, embedding: list[float], threshold: float
    ) -> list[EpisodicMemory]:
        memories = await self.get_episodic_memories(character_id)
        results = []
        for m in memories:
            if m.embedding is None:
                continue
            sim = cosine_similarity(m.embedding, embedding)
            if sim >= threshold:
                results.append(m)
        return results

    async def save_semantic_memory(self, memory: SemanticMemory) -> None:
        embedding_json = json.dumps(memory.embedding) if memory.embedding else None
        await self._conn.execute(
            """INSERT INTO semantic_memories
            (id, character_id, content, confidence, embedding)
            VALUES (?, ?, ?, ?, ?)""",
            (memory.id, memory.character_id, memory.content,
             memory.confidence, embedding_json),
        )
        # Save source episode links
        for ep_id in memory.source_episode_ids:
            await self._conn.execute(
                """INSERT OR IGNORE INTO semantic_memory_sources
                (semantic_memory_id, episodic_memory_id) VALUES (?, ?)""",
                (memory.id, ep_id),
            )
        await self._conn.commit()

    async def get_semantic_memories(self, character_id: str) -> list[SemanticMemory]:
        cursor = await self._conn.execute(
            "SELECT * FROM semantic_memories WHERE character_id = ?", (character_id,)
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            # Get source episode IDs
            src_cursor = await self._conn.execute(
                "SELECT episodic_memory_id FROM semantic_memory_sources WHERE semantic_memory_id = ?",
                (row["id"],),
            )
            src_rows = await src_cursor.fetchall()
            source_ids = [r["episodic_memory_id"] for r in src_rows]

            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            result.append(SemanticMemory(
                id=row["id"], character_id=row["character_id"],
                content=row["content"], confidence=row["confidence"],
                source_episode_ids=source_ids, embedding=embedding,
            ))
        return result

    async def update_semantic_memory(self, memory: SemanticMemory) -> None:
        """按 ID 覆盖更新语义记忆。"""
        embedding_json = json.dumps(memory.embedding) if memory.embedding else None
        await self._conn.execute(
            """UPDATE semantic_memories
            SET content = ?, confidence = ?, character_id = ?, embedding = ?
            WHERE id = ?""",
            (memory.content, memory.confidence, memory.character_id,
             embedding_json, memory.id),
        )
        # Replace source episode links
        await self._conn.execute(
            "DELETE FROM semantic_memory_sources WHERE semantic_memory_id = ?",
            (memory.id,),
        )
        for ep_id in memory.source_episode_ids:
            await self._conn.execute(
                """INSERT OR IGNORE INTO semantic_memory_sources
                (semantic_memory_id, episodic_memory_id) VALUES (?, ?)""",
                (memory.id, ep_id),
            )
        await self._conn.commit()

    async def delete_semantic_memory(self, memory_id: str) -> None:
        """按 ID 删除语义记忆。"""
        await self._conn.execute(
            "DELETE FROM semantic_memory_sources WHERE semantic_memory_id = ?",
            (memory_id,),
        )
        await self._conn.execute(
            "DELETE FROM semantic_memories WHERE id = ?",
            (memory_id,),
        )
        await self._conn.commit()

    # --- MemoryStore protocol aliases ---

    async def add_episodic(self, memory: EpisodicMemory) -> None:
        """MemoryStore protocol: add_episodic -> save_episodic_memory."""
        await self.save_episodic_memory(memory)

    async def add_semantic(self, memory: SemanticMemory) -> None:
        """MemoryStore protocol: add_semantic -> save_semantic_memory."""
        await self.save_semantic_memory(memory)

    async def get_episodic_by_character(self, character_id: str) -> list[EpisodicMemory]:
        """MemoryStore protocol: get_episodic_by_character -> get_episodic_memories."""
        return await self.get_episodic_memories(character_id)

    async def get_semantic_by_character(self, character_id: str) -> list[SemanticMemory]:
        """MemoryStore protocol: get_semantic_by_character -> get_semantic_memories."""
        return await self.get_semantic_memories(character_id)

    async def update_semantic(self, memory: SemanticMemory) -> None:
        """MemoryStore protocol: update_semantic -> update_semantic_memory."""
        await self.update_semantic_memory(memory)

    async def delete_semantic(self, memory_id: str) -> None:
        """MemoryStore protocol: delete_semantic -> delete_semantic_memory."""
        await self.delete_semantic_memory(memory_id)

    async def find_similar_episodic(
        self, character_id: str, embedding: list[float], threshold: float
    ) -> list[EpisodicMemory]:
        """MemoryStore protocol: find_similar_episodic -> find_similar_memories."""
        return await self.find_similar_memories(character_id, embedding, threshold)

    # --- Goals ---

    async def save_goals(self, character_id: str, goals: GoalTree) -> None:
        # Delete existing goals for this character
        await self._conn.execute("DELETE FROM tasks WHERE character_id = ?", (character_id,))
        await self._conn.execute("DELETE FROM objectives WHERE character_id = ?", (character_id,))
        await self._conn.execute("DELETE FROM visions WHERE character_id = ?", (character_id,))

        for v in goals.visions:
            await self._conn.execute(
                "INSERT INTO visions (id, character_id, content) VALUES (?, ?, ?)",
                (v.id, v.character_id, v.content),
            )
        for o in goals.objectives:
            await self._conn.execute(
                """INSERT INTO objectives (id, character_id, vision_id, content, status, progress)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (o.id, o.character_id, o.vision_id, o.content, o.status, o.progress),
            )
        for t in goals.tasks:
            await self._conn.execute(
                """INSERT INTO tasks (id, character_id, objective_id, content, status)
                VALUES (?, ?, ?, ?, ?)""",
                (t.id, t.character_id, t.objective_id, t.content, t.status),
            )
        await self._conn.commit()

    async def get_goals(self, character_id: str) -> GoalTree | None:
        # Visions
        cursor = await self._conn.execute(
            "SELECT * FROM visions WHERE character_id = ?", (character_id,)
        )
        v_rows = await cursor.fetchall()
        if not v_rows:
            # Check if there are any objectives or tasks
            cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM objectives WHERE character_id = ?", (character_id,)
            )
            count = (await cursor.fetchone())[0]
            if count == 0:
                cursor = await self._conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE character_id = ?", (character_id,)
                )
                count = (await cursor.fetchone())[0]
                if count == 0:
                    return None

        visions = [
            Vision(id=r["id"], character_id=r["character_id"], content=r["content"])
            for r in v_rows
        ]

        # Objectives
        cursor = await self._conn.execute(
            "SELECT * FROM objectives WHERE character_id = ?", (character_id,)
        )
        o_rows = await cursor.fetchall()
        objectives = [
            Objective(
                id=r["id"], character_id=r["character_id"],
                vision_id=r["vision_id"], content=r["content"],
                status=r["status"], progress=r["progress"],
            )
            for r in o_rows
        ]

        # Tasks
        cursor = await self._conn.execute(
            "SELECT * FROM tasks WHERE character_id = ?", (character_id,)
        )
        t_rows = await cursor.fetchall()
        tasks = [
            Task(
                id=r["id"], character_id=r["character_id"],
                objective_id=r["objective_id"], content=r["content"],
                status=r["status"],
            )
            for r in t_rows
        ]

        return GoalTree(visions=visions, objectives=objectives, tasks=tasks)

    # --- State ---

    async def save_emotional_state(
        self, character_id: str, state: EmotionalState
    ) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO emotional_states
            (character_id, pleasure, arousal, dominance, emotion_label, situation)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (character_id, state.pleasure, state.arousal, state.dominance,
             state.emotion_label, state.situation),
        )
        await self._conn.commit()

    async def get_emotional_state(
        self, character_id: str
    ) -> EmotionalState | None:
        cursor = await self._conn.execute(
            "SELECT * FROM emotional_states WHERE character_id = ?", (character_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return EmotionalState(
            pleasure=row["pleasure"], arousal=row["arousal"],
            dominance=row["dominance"], emotion_label=row["emotion_label"],
            situation=row["situation"],
        )

    # --- ChangeLog ---

    async def save_change(self, change: ChangeRecord) -> None:
        before_json = json.dumps(change.before) if change.before is not None else None
        after_json = json.dumps(change.after)
        await self._conn.execute(
            """INSERT INTO change_log
            (id, character_id, type, before_value, after_value, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (change.id, change.character_id, change.type,
             before_json, after_json, change.reason,
             change.timestamp.isoformat()),
        )
        await self._conn.commit()

    async def get_changes(
        self, character_id: str, limit: int = 100
    ) -> list[ChangeRecord]:
        cursor = await self._conn.execute(
            """SELECT * FROM change_log WHERE character_id = ?
            ORDER BY created_at DESC LIMIT ?""",
            (character_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            ChangeRecord(
                id=r["id"], character_id=r["character_id"],
                type=r["type"],
                before=json.loads(r["before_value"]) if r["before_value"] else None,
                after=json.loads(r["after_value"]),
                reason=r["reason"],
                timestamp=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # --- Todo ---

    async def save_todo(self, todo: TodoItem) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO todos
            (id, content, label, kind, status, priority,
             due_date, recurrence, created_at, completed_at, owner_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                todo.id, todo.content, todo.label, todo.kind,
                todo.status, todo.priority,
                todo.due_date.isoformat() if todo.due_date else None,
                todo.recurrence,
                todo.created_at.isoformat(),
                todo.completed_at.isoformat() if todo.completed_at else None,
                todo.owner_id,
            ),
        )
        await self._conn.commit()

    async def get_todo(self, todo_id: str) -> TodoItem | None:
        cursor = await self._conn.execute(
            "SELECT * FROM todos WHERE id = ?", (todo_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_todo(row)

    async def list_todos(
        self, status: str | None = None, label: str | None = None,
        owner_id: str | None = None,
    ) -> list[TodoItem]:
        query = "SELECT * FROM todos"
        params: list[str] = []
        conditions: list[str] = []
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if label is not None:
            conditions.append("label = ?")
            params.append(label)
        if owner_id is not None:
            conditions.append("owner_id = ?")
            params.append(owner_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_todo(row) for row in rows]

    async def update_todo(self, todo: TodoItem) -> None:
        await self._conn.execute(
            """UPDATE todos SET
            content = ?, label = ?, kind = ?, status = ?,
            priority = ?, due_date = ?, recurrence = ?,
            created_at = ?, completed_at = ?, owner_id = ?
            WHERE id = ?""",
            (
                todo.content, todo.label, todo.kind, todo.status,
                todo.priority,
                todo.due_date.isoformat() if todo.due_date else None,
                todo.recurrence,
                todo.created_at.isoformat(),
                todo.completed_at.isoformat() if todo.completed_at else None,
                todo.owner_id,
                todo.id,
            ),
        )
        await self._conn.commit()

    async def delete_todo(self, todo_id: str) -> None:
        await self._conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        await self._conn.commit()

    @staticmethod
    def _row_to_todo(row: aiosqlite.Row) -> TodoItem:
        from datetime import date as date_type

        due_date = None
        if row["due_date"]:
            due_date = date_type.fromisoformat(row["due_date"])

        completed_at = None
        if row["completed_at"]:
            completed_at = datetime.fromisoformat(row["completed_at"])

        # 对不存在 owner_id 字段的旧 DB 做向后兼容
        try:
            owner_id = row["owner_id"]
        except (IndexError, KeyError):
            owner_id = "user"

        return TodoItem(
            id=row["id"],
            content=row["content"],
            label=row["label"],
            kind=row["kind"],
            status=row["status"],
            priority=row["priority"],
            due_date=due_date,
            recurrence=row["recurrence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=completed_at,
            owner_id=owner_id or "user",
        )

    # --- Relation ---

    async def save_relation(self, relation: Relation) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO relations
            (id, owner_id, target_id, target_name, relationship_type,
             description, closeness, trust, updated_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                relation.id, relation.owner_id, relation.target_id,
                relation.target_name, relation.relationship_type,
                relation.description, relation.closeness, relation.trust,
                relation.updated_at.isoformat(), relation.notes,
            ),
        )
        await self._conn.commit()

    async def get_relation(self, relation_id: str) -> Relation | None:
        cursor = await self._conn.execute(
            "SELECT * FROM relations WHERE id = ?", (relation_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)

    async def list_relations(
        self, owner_id: str | None = None,
    ) -> list[Relation]:
        if owner_id is not None:
            cursor = await self._conn.execute(
                "SELECT * FROM relations WHERE owner_id = ?", (owner_id,)
            )
        else:
            cursor = await self._conn.execute("SELECT * FROM relations")
        rows = await cursor.fetchall()
        return [self._row_to_relation(row) for row in rows]

    @staticmethod
    def _row_to_relation(row: aiosqlite.Row) -> Relation:
        return Relation(
            id=row["id"],
            owner_id=row["owner_id"],
            target_id=row["target_id"],
            target_name=row["target_name"],
            relationship_type=row["relationship_type"],
            description=row["description"],
            closeness=row["closeness"],
            trust=row["trust"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            notes=row["notes"],
        )

    # --- Helpers ---

    @staticmethod
    def _row_to_episodic(row: aiosqlite.Row) -> EpisodicMemory:
        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        return EpisodicMemory(
            id=row["id"], character_id=row["character_id"],
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            emotional_valence=row["emotional_valence"],
            importance=row["importance"],
            conversation_id=row["conversation_id"],
            embedding=embedding,
        )
