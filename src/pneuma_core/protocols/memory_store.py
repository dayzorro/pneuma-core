"""MemoryStore protocol: interface for memory persistence."""

from typing import Protocol

from pneuma_core.models.memory import EpisodicMemory, SemanticMemory


class MemoryStore(Protocol):
    """记忆存储协议。

    与 StorageBackend 分离的记忆专用接口。
    记忆需要向量相似检索，其访问模式与其他数据不同，因此独立出来。
    """

    async def add_episodic(self, memory: EpisodicMemory) -> None:
        """新增情节记忆。"""
        ...

    async def add_semantic(self, memory: SemanticMemory) -> None:
        """新增语义记忆。"""
        ...

    async def update_semantic(self, memory: SemanticMemory) -> None:
        """更新语义记忆。"""
        ...

    async def delete_semantic(self, memory_id: str) -> None:
        """删除语义记忆。"""
        ...

    async def get_episodic_by_character(self, character_id: str) -> list[EpisodicMemory]:
        """获取某个角色的全部情节记忆。"""
        ...

    async def get_semantic_by_character(self, character_id: str) -> list[SemanticMemory]:
        """获取某个角色的全部语义记忆。"""
        ...

    async def find_similar_episodic(
        self, character_id: str, embedding: list[float], threshold: float
    ) -> list[EpisodicMemory]:
        """检索相似度达到阈值以上的情节记忆。"""
        ...
