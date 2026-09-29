"""InMemoryStorageBackend：面向测试的内存存储实现。"""

from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.models.change_record import ChangeRecord
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.relation import Relation
from pneuma_core.models.todo import TodoItem


class InMemoryStorageBackend:
    """内存存储实现（用于测试与开发）。"""

    def __init__(self) -> None:
        self._characters: dict[str, Character] = {}
        self._episodic: list[EpisodicMemory] = []
        self._semantic: list[SemanticMemory] = []
        self._goals: dict[str, GoalTree] = {}
        self._states: dict[str, EmotionalState] = {}
        self._changes: list[ChangeRecord] = []
        self._todos: dict[str, TodoItem] = {}
        self._relations: dict[str, Relation] = {}

    # --- Character ---

    async def save_character(self, character: Character) -> None:
        self._characters[character.id] = character

    async def get_character(self, character_id: str) -> Character | None:
        return self._characters.get(character_id)

    async def list_characters(self) -> list[Character]:
        return list(self._characters.values())

    # --- Memory ---

    async def save_episodic_memory(self, memory: EpisodicMemory) -> None:
        """追加情节记忆（append-only：即使 ID 相同也不覆盖）。"""
        self._episodic.append(memory)

    async def get_episodic_memories(self, character_id: str) -> list[EpisodicMemory]:
        return [m for m in self._episodic if m.character_id == character_id]

    async def find_similar_memories(
        self, character_id: str, embedding: list[float], threshold: float
    ) -> list[EpisodicMemory]:
        results = []
        for m in self._episodic:
            if m.character_id != character_id or m.embedding is None:
                continue
            sim = cosine_similarity(m.embedding, embedding)
            if sim >= threshold:
                results.append(m)
        return results

    async def save_semantic_memory(self, memory: SemanticMemory) -> None:
        self._semantic.append(memory)

    async def get_semantic_memories(self, character_id: str) -> list[SemanticMemory]:
        return [m for m in self._semantic if m.character_id == character_id]

    async def update_semantic_memory(self, memory: SemanticMemory) -> None:
        """按 ID 覆盖更新语义记忆。"""
        self._semantic = [
            memory if m.id == memory.id else m for m in self._semantic
        ]

    async def delete_semantic_memory(self, memory_id: str) -> None:
        """按 ID 删除语义记忆。"""
        self._semantic = [m for m in self._semantic if m.id != memory_id]

    # --- Goals ---

    async def save_goals(self, character_id: str, goals: GoalTree) -> None:
        self._goals[character_id] = goals

    async def get_goals(self, character_id: str) -> GoalTree | None:
        return self._goals.get(character_id)

    # --- State ---

    async def save_emotional_state(
        self, character_id: str, state: EmotionalState
    ) -> None:
        self._states[character_id] = state

    async def get_emotional_state(
        self, character_id: str
    ) -> EmotionalState | None:
        return self._states.get(character_id)

    # --- ChangeLog ---

    async def save_change(self, change: ChangeRecord) -> None:
        self._changes.append(change)

    async def get_changes(
        self, character_id: str, limit: int = 100
    ) -> list[ChangeRecord]:
        filtered = [c for c in self._changes if c.character_id == character_id]
        filtered.sort(key=lambda c: c.timestamp, reverse=True)
        return filtered[:limit]

    # --- Todo ---

    async def save_todo(self, todo: TodoItem) -> None:
        self._todos[todo.id] = todo

    async def get_todo(self, todo_id: str) -> TodoItem | None:
        return self._todos.get(todo_id)

    async def list_todos(
        self, status: str | None = None, label: str | None = None,
        owner_id: str | None = None,
    ) -> list[TodoItem]:
        result = list(self._todos.values())
        if status is not None:
            result = [t for t in result if t.status == status]
        if label is not None:
            result = [t for t in result if t.label == label]
        if owner_id is not None:
            result = [t for t in result if t.owner_id == owner_id]
        return result

    async def update_todo(self, todo: TodoItem) -> None:
        self._todos[todo.id] = todo

    async def delete_todo(self, todo_id: str) -> None:
        self._todos.pop(todo_id, None)

    # --- Relation ---

    async def save_relation(self, relation: Relation) -> None:
        self._relations[relation.id] = relation

    async def get_relation(self, relation_id: str) -> Relation | None:
        return self._relations.get(relation_id)

    async def list_relations(
        self, owner_id: str | None = None,
    ) -> list[Relation]:
        result = list(self._relations.values())
        if owner_id is not None:
            result = [r for r in result if r.owner_id == owner_id]
        return result
