"""Tests for Relation model and storage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pneuma_core.storage.in_memory import InMemoryStorageBackend

JST = timezone(timedelta(hours=9))


# --- モデルテスト ---


class TestRelationModel:
    """Relation データモデルの検証."""

    def test_create_relation(self) -> None:
        """Relation を作成できる."""
        from pneuma_core.models.relation import Relation

        rel = Relation(
            id="rel-001",
            owner_id="user",
            target_id="mira",
            target_name="ミラ",
            relationship_type="partner",
            description="毎日会話するパートナー",
            closeness=0.8,
            trust=0.9,
            updated_at=datetime(2026, 3, 2, 10, 0, 0, tzinfo=JST),
        )
        assert rel.owner_id == "user"
        assert rel.target_name == "ミラ"
        assert rel.closeness == 0.8

    def test_relation_default_notes(self) -> None:
        """notes のデフォルトは None."""
        from pneuma_core.models.relation import Relation

        rel = Relation(
            id="rel-002",
            owner_id="mira",
            target_id="user",
            target_name="きょうへいさん",
            relationship_type="partner",
            description="大切な人",
            closeness=0.9,
            trust=0.95,
            updated_at=datetime(2026, 3, 2, 10, 0, 0, tzinfo=JST),
        )
        assert rel.notes is None


# --- ストレージテスト ---


class TestRelationStorage:
    """Relation のストレージ操作テスト."""

    @pytest.mark.asyncio
    async def test_save_and_get_relation(self) -> None:
        """Relation を保存して取得できる."""
        from pneuma_core.models.relation import Relation

        store = InMemoryStorageBackend()
        rel = Relation(
            id="rel-001",
            owner_id="user",
            target_id="mira",
            target_name="ミラ",
            relationship_type="partner",
            description="パートナー",
            closeness=0.8,
            trust=0.9,
            updated_at=datetime(2026, 3, 2, 10, 0, 0, tzinfo=JST),
        )
        await store.save_relation(rel)
        result = await store.get_relation("rel-001")
        assert result is not None
        assert result.target_name == "ミラ"

    @pytest.mark.asyncio
    async def test_list_relations_by_owner(self) -> None:
        """owner_id で Relation を絞り込める."""
        from pneuma_core.models.relation import Relation

        store = InMemoryStorageBackend()
        await store.save_relation(Relation(
            id="r-1", owner_id="user", target_id="mira",
            target_name="ミラ", relationship_type="partner",
            description="パートナー", closeness=0.8, trust=0.9,
            updated_at=datetime(2026, 3, 2, tzinfo=JST),
        ))
        await store.save_relation(Relation(
            id="r-2", owner_id="mira", target_id="user",
            target_name="きょうへいさん", relationship_type="partner",
            description="大切な人", closeness=0.9, trust=0.95,
            updated_at=datetime(2026, 3, 2, tzinfo=JST),
        ))

        user_rels = await store.list_relations(owner_id="user")
        mira_rels = await store.list_relations(owner_id="mira")

        assert len(user_rels) == 1
        assert user_rels[0].target_name == "ミラ"
        assert len(mira_rels) == 1
        assert mira_rels[0].target_name == "きょうへいさん"

    @pytest.mark.asyncio
    async def test_update_relation(self) -> None:
        """Relation を更新できる."""
        from dataclasses import replace

        from pneuma_core.models.relation import Relation

        store = InMemoryStorageBackend()
        rel = Relation(
            id="r-1", owner_id="user", target_id="mira",
            target_name="ミラ", relationship_type="partner",
            description="パートナー", closeness=0.5, trust=0.6,
            updated_at=datetime(2026, 3, 1, tzinfo=JST),
        )
        await store.save_relation(rel)

        updated = replace(rel, closeness=0.8, trust=0.9,
                          description="信頼できるパートナー")
        await store.save_relation(updated)

        result = await store.get_relation("r-1")
        assert result is not None
        assert result.closeness == 0.8
        assert result.description == "信頼できるパートナー"


# --- ローダーテスト ---


class TestLoadUserRelations:
    """vault/user/relationships.yaml から Relations を読み込む."""

    def test_load_relations_from_yaml(self, tmp_path: Path) -> None:
        """relationships.yaml が存在すれば Relations リストが返る."""
        from pneuma_core.runtime.user_context import load_user_relations

        yaml_file = tmp_path / "relationships.yaml"
        yaml_file.write_text(
            """\
relations:
  - id: r-1
    target_id: mira
    target_name: ミラ
    relationship_type: partner
    description: 毎日会話するパートナー
    closeness: 0.8
    trust: 0.9
  - id: r-2
    target_id: mother
    target_name: 母
    relationship_type: family
    description: 実家に住む母
    closeness: 0.7
    trust: 0.95
""",
            encoding="utf-8",
        )

        result = load_user_relations(yaml_file)
        assert result is not None
        assert len(result) == 2
        assert result[0].target_name == "ミラ"
        assert result[1].target_name == "母"
        assert all(r.owner_id == "user" for r in result)

    def test_load_relations_file_not_found(self, tmp_path: Path) -> None:
        """ファイルが存在しなければ None."""
        from pneuma_core.runtime.user_context import load_user_relations

        result = load_user_relations(tmp_path / "nonexistent.yaml")
        assert result is None


# --- PromptBuilder テスト ---


class TestRelationsInPrompt:
    """PromptBuilder で Relations がプロンプトに含まれる."""

    def test_character_relations_in_prompt(self) -> None:
        """キャラクターの Relations がプロンプトに含まれる."""
        from pneuma_core.models.character import Character
        from pneuma_core.models.emotion import EmotionalState
        from pneuma_core.models.goals import GoalTree
        from pneuma_core.models.personality import Personality
        from pneuma_core.models.relation import Relation
        from pneuma_core.models.values import Values
        from pneuma_core.runtime.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        char = Character(
            id="mira", name="ミラ",
            personality=Personality(0.7, 0.6, 0.8, 0.9, 0.3),
            values=Values(0.8, 0.3, 0.7, 0.4),
        )
        emotion = EmotionalState(0.3, 0.1, 0.0, "安らぎ", "穏やかな会話中")
        relations = [
            Relation(
                id="r-1", owner_id="mira", target_id="user",
                target_name="きょうへいさん", relationship_type="partner",
                description="大切な人。毎日会話している",
                closeness=0.9, trust=0.95,
                updated_at=datetime(2026, 3, 2, tzinfo=JST),
            ),
        ]

        prompt = builder.build(
            char, emotion, GoalTree(), [],
            character_relations=relations,
        )
        assert "关系" in prompt
        assert "きょうへいさん" in prompt
        assert "大切な人" in prompt

    def test_no_relations_no_section(self) -> None:
        """relations が空ならセクション無し."""
        from pneuma_core.models.character import Character
        from pneuma_core.models.emotion import EmotionalState
        from pneuma_core.models.goals import GoalTree
        from pneuma_core.models.personality import Personality
        from pneuma_core.models.values import Values
        from pneuma_core.runtime.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        char = Character(
            id="mira", name="ミラ",
            personality=Personality(0.7, 0.6, 0.8, 0.9, 0.3),
            values=Values(0.8, 0.3, 0.7, 0.4),
        )
        emotion = EmotionalState(0.3, 0.1, 0.0, "安らぎ", "穏やかな会話中")

        prompt = builder.build(char, emotion, GoalTree(), [])
        assert "关系" not in prompt
