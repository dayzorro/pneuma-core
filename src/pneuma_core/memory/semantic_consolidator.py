"""语义记忆整合：由情节记忆归纳出语义记忆。

对相关的情节记忆进行聚类，并使用 LLM 从中抽取可泛化的知识（语义记忆）。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.llm.embedding import EmbeddingService
from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory

CONSOLIDATION_SYSTEM_PROMPT = """\
你是一个负责记忆整合的系统。
请从以下情节记忆群中抽取出共同的模式、理解与一般性知识。
所有文本字段必须使用简体中文。

情节记忆:
{episodes}

请严格按照以下 JSON 格式回答:
{{
  "content": "整合后的理解（1-2 句）",
  "confidence": 0.0-1.0（确信度，根据情节数量与一致性判断）
}}
"""


@dataclass(frozen=True)
class SemanticConsolidationConfig:
    """语义记忆整合的可调参数。"""

    min_episodes: int = 3
    similarity_threshold: float = 0.7
    duplicate_threshold: float = 0.9
    model: str = "claude-haiku-4-5-20251001"
    max_semantics_per_batch: int = 5


class SemanticConsolidator:
    """将情节记忆整合为语义记忆。

    流程:
        1. 过滤掉没有 embedding 的情节记忆
        2. 按 embedding 相似度对相关情节聚类
        3. 过滤掉成员数少于 min_episodes 的簇
        4. 对每个簇调用 LLM 生成语义记忆
        5. 与已有语义记忆做重复检查
        6. 为新语义记忆生成 embedding
        7. 返回新的 SemanticMemory 对象
    """

    def __init__(
        self,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        config: SemanticConsolidationConfig | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_service = embedding_service
        self.config = config or SemanticConsolidationConfig()

    async def consolidate(
        self,
        character_id: str,
        episodes: list[EpisodicMemory],
        existing_semantics: list[SemanticMemory],
    ) -> list[SemanticMemory]:
        """Consolidate episodic memories into semantic memories.

        1. Cluster related episodes by embedding similarity
        2. Filter clusters with < min_episodes
        3. For each cluster, call LLM to generate semantic memory
        4. Check for duplicates against existing semantics
        5. Embed the new semantic memories
        6. Return new SemanticMemory objects
        """
        # Filter episodes without embeddings
        episodes_with_emb = [ep for ep in episodes if ep.embedding is not None]
        if not episodes_with_emb:
            return []

        # 1. Cluster related episodes
        raw_clusters = self._cluster_episodes(episodes_with_emb)

        # 2. Filter clusters below min_episodes
        valid_clusters = [
            c for c in raw_clusters if len(c) >= self.config.min_episodes
        ]
        if not valid_clusters:
            return []

        # Limit to max_semantics_per_batch
        valid_clusters = valid_clusters[: self.config.max_semantics_per_batch]

        # 3-6. Process each cluster
        results: list[SemanticMemory] = []
        for cluster in valid_clusters:
            semantic = await self._process_cluster(
                character_id, cluster, existing_semantics
            )
            if semantic is not None:
                results.append(semantic)
                # Add to existing_semantics to prevent duplicates within batch
                existing_semantics = [*existing_semantics, semantic]

        return results

    def _cluster_episodes(
        self, episodes: list[EpisodicMemory]
    ) -> list[list[EpisodicMemory]]:
        """Cluster episodes by embedding similarity using greedy merging.

        Algorithm:
        1. Build adjacency: for each pair, check if similarity > threshold
        2. Use Union-Find to merge overlapping clusters
        3. Return list of clusters
        """
        n = len(episodes)
        if n == 0:
            return []

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build adjacency and merge
        threshold = self.config.similarity_threshold
        for i in range(n):
            for j in range(i + 1, n):
                assert episodes[i].embedding is not None
                assert episodes[j].embedding is not None
                sim = cosine_similarity(episodes[i].embedding, episodes[j].embedding)
                if sim >= threshold:
                    union(i, j)

        # Group by root
        clusters_map: dict[int, list[EpisodicMemory]] = {}
        for i in range(n):
            root = find(i)
            if root not in clusters_map:
                clusters_map[root] = []
            clusters_map[root].append(episodes[i])

        return list(clusters_map.values())

    async def _process_cluster(
        self,
        character_id: str,
        cluster: list[EpisodicMemory],
        existing_semantics: list[SemanticMemory],
    ) -> SemanticMemory | None:
        """Process a single cluster: LLM consolidation, dedup, embedding."""
        # Call LLM to generate semantic memory
        try:
            episodes_text = "\n".join(
                f"{i + 1}. {ep.content}" for i, ep in enumerate(cluster)
            )
            system_prompt = CONSOLIDATION_SYSTEM_PROMPT.format(episodes=episodes_text)

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": "请整合上述情节记忆。"}],
                model=self.config.model,
                temperature=0.3,
                max_tokens=512,
            )
            response = await self.llm.generate(request)
        except Exception:
            return None

        # Parse LLM response
        parsed = self._parse_llm_response(response.content)
        if parsed is None:
            return None

        content = parsed["content"]
        confidence = float(parsed["confidence"])

        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        # Generate embedding for the new semantic memory
        try:
            embedding = await self.embedding_service.embed(content)
        except Exception:
            return None

        # Check for duplicates against existing semantics
        for existing in existing_semantics:
            if existing.embedding is not None:
                sim = cosine_similarity(embedding, existing.embedding)
                if sim >= self.config.duplicate_threshold:
                    return None

        # Build and return SemanticMemory
        source_ids = [ep.id for ep in cluster]
        return SemanticMemory(
            id=f"sem-{uuid.uuid4().hex[:12]}",
            character_id=character_id,
            content=content,
            confidence=confidence,
            source_episode_ids=source_ids,
            embedding=embedding,
        )

    def _parse_llm_response(self, response: str) -> dict | None:
        """Parse LLM response JSON, handling markdown code blocks."""
        text = response.strip()

        # Strip markdown code blocks
        md_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
        if md_match:
            text = md_match.group(1).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(data, dict):
            return None
        if "content" not in data or "confidence" not in data:
            return None

        return data
