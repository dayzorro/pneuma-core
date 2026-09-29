"""跨角色自主对话演示。

2 个角色（夏澜、知雨）交替向对方发送消息，
逐轮观察各角色的情感 (PAD) 与记忆/关系的变化。

会话中: emotion (PAD) 会逐轮变化
会话结束时: SessionEndPipeline 会更新 episodic_memory / semantic_memory /
                 relation

运行:
    export ANTHROPIC_API_KEY=...  # 用于 LLM (Claude)
    export OPENAI_API_KEY=...     # 用于 Embedding (text-embedding-3-small)
    .venv/bin/python examples/cross_chat.py [TURNS]   # 默认 6 轮

注意:
    - 最少 2 轮 × 2 个角色 = 4 次 LLM 调用 + session-end 再加 2 次调用
    - 6 轮总计约 14 次调用（注意成本，尽量简短）
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.llm.claude import ClaudeAdapter
from pneuma_core.llm.embedding import OpenAIEmbeddingService
from pneuma_core.models.message import MessageInput, MessageOutput
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.runtime.session import ConversationSession
from pneuma_core.runtime.session_end_pipeline import SessionEndPipeline
from pneuma_core.storage.sqlite import SQLiteStorageBackend


# ──────────────────────────────────────────────────────────────────────
# 快照辅助函数
# ──────────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path(__file__).parent


async def _snapshot(
    storage: SQLiteStorageBackend, character_id: str, name: str
) -> str:
    """压缩成一行展示的 PAD / memory / relation 快照。"""
    state = await storage.get_emotional_state(character_id)
    pad = (
        f"P={state.pleasure:+.2f} A={state.arousal:+.2f} D={state.dominance:+.2f} "
        f"({state.emotion_label})"
        if state
        else "(uninitialised)"
    )
    eps = await storage.get_episodic_memories(character_id)
    sems = await storage.get_semantic_memories(character_id)
    rels = await storage.list_relations(owner_id=character_id)
    rel_part = ""
    if rels:
        r = rels[0]
        rel_part = f" rel→{r.target_name} close={r.closeness:.2f} trust={r.trust:.2f}"
    return f"{name:>4}: {pad}  ep={len(eps)} sem={len(sems)}{rel_part}"


async def _setup_character(
    storage: SQLiteStorageBackend,
    sheet_path: Path,
) -> tuple[str, str]:
    """加载 YAML，并把角色、goals、初始情感保存到 storage。"""
    sheet = CharacterSheet.load(sheet_path)
    char = sheet.character
    await storage.save_character(char)
    if sheet.goal_tree is not None:
        await storage.save_goals(char.id, sheet.goal_tree)
    if sheet.initial_state is not None:
        await storage.save_emotional_state(char.id, sheet.initial_state)
    return char.id, char.name


# ──────────────────────────────────────────────────────────────────────
# 引擎工厂
# ──────────────────────────────────────────────────────────────────────


def _make_engine(
    *,
    character_id: str,
    storage: SQLiteStorageBackend,
    llm: ClaudeAdapter,
    embedding: OpenAIEmbeddingService,
) -> RuntimeEngine:
    """为 1 个角色生成 RuntimeEngine。storage 同时兼作 MemoryStore (SQLite)。"""
    return RuntimeEngine(
        character_id=character_id,
        storage=storage,
        llm=llm,
        embedding_service=embedding,
        memory_store=storage,  # SQLiteStorageBackend 也满足 MemoryStore Protocol
    )


# ──────────────────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────────────────


async def cross_chat(turns: int) -> None:
    # 可选的环境变量密钥 — 提前检查，避免用户花时间运行后才崩溃。
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: 未设置 ANTHROPIC_API_KEY。", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: 未设置 OPENAI_API_KEY。", file=sys.stderr)
        sys.exit(1)

    # 临时 sqlite 文件。结束时删除。
    tmpfile = tempfile.NamedTemporaryFile(
        prefix="pneuma_cross_", suffix=".db", delete=False
    )
    tmpfile.close()
    db_path = tmpfile.name
    print(f"# storage: {db_path}\n")

    storage = SQLiteStorageBackend(db_path)
    await storage.initialize()
    try:
        # 1. 加载 2 个角色
        aine_id, aine_name = await _setup_character(
            storage, EXAMPLES_DIR / "aine.character.yaml"
        )
        rin_id, rin_name = await _setup_character(
            storage, EXAMPLES_DIR / "rin.character.yaml"
        )

        # 2. 适配器 + Runtime
        llm = ClaudeAdapter.from_env()
        embedding = OpenAIEmbeddingService.from_env()
        aine_engine = _make_engine(
            character_id=aine_id, storage=storage, llm=llm, embedding=embedding
        )
        rin_engine = _make_engine(
            character_id=rin_id, storage=storage, llm=llm, embedding=embedding
        )

        # 3. 初始快照
        print("─" * 70)
        print(f"INITIAL  {await _snapshot(storage, aine_id, aine_name)}")
        print(f"         {await _snapshot(storage, rin_id, rin_name)}")
        print("─" * 70)

        # 4. 开场白（由人类的“系统”给出话题引子）
        opener = "初次见面。方便的话，说说你最近痴迷的事吧。"
        last_speaker_id, last_speaker_name = "system", "system"
        last_content = opener
        print(f"\n  [opener  → 全体] {opener}\n")

        # session 记账（供 session_end_pipeline 使用）
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        aine_session = ConversationSession(
            session_id=session_id,
            character_id=aine_id,
            user_id=rin_id,
            channel_id="cross-chat",
            started_at=now,
            last_active_at=now,
        )
        rin_session = ConversationSession(
            session_id=session_id,
            character_id=rin_id,
            user_id=aine_id,
            channel_id="cross-chat",
            started_at=now,
            last_active_at=now,
        )

        # 5. 逐轮交替推进
        speakers = [
            (aine_engine, aine_id, aine_name, aine_session, rin_session),
            (rin_engine, rin_id, rin_name, rin_session, aine_session),
        ]
        for turn in range(1, turns + 1):
            engine, char_id, char_name, own_session, peer_session = speakers[
                (turn - 1) % 2
            ]
            msg_in = MessageInput(
                content=last_content,
                sender_id=last_speaker_id,
                sender_name=last_speaker_name,
                sender_type="character" if last_speaker_id != "system" else "system",
            )
            output: MessageOutput = await engine.process_message(msg_in)
            # 写入 session（双方都记录）
            own_session.messages.append(
                {"role": "user", "content": last_content,
                 "sender_id": last_speaker_id, "sender_name": last_speaker_name}
            )
            own_session.messages.append(
                {"role": "assistant", "content": output.content,
                 "sender_id": char_id, "sender_name": char_name}
            )
            peer_session.messages.append(
                {"role": "assistant", "content": last_content,
                 "sender_id": last_speaker_id, "sender_name": last_speaker_name}
            )
            peer_session.messages.append(
                {"role": "user", "content": output.content,
                 "sender_id": char_id, "sender_name": char_name}
            )

            # 显示
            print(f"  [turn {turn} {char_name}] {output.content}")
            print(f"          {await _snapshot(storage, char_id, char_name)}\n")

            last_speaker_id = char_id
            last_speaker_name = char_name
            last_content = output.content

        # 6. 对两个角色分别执行会话结束处理（各调用一次 Opus）
        print("─" * 70)
        print("SESSION END pipeline (LLM analysis → episodic / semantic / relation)")
        print("─" * 70)
        pipeline = SessionEndPipeline(
            llm=llm,
            embedding_service=embedding,
            memory_store=storage,
            storage=storage,
        )
        for engine, char_id, char_name, own_session, _peer in speakers:
            existing_sem = await storage.get_semantic_memories(char_id)
            existing_rel = await storage.list_relations(owner_id=char_id)
            result = await pipeline.run(
                session=own_session,
                character_id=char_id,
                existing_semantics=existing_sem,
                existing_relations=existing_rel,
            )
            print(
                f"  {char_name}: success={result.success} "
                f"episodic={result.episodic_memories_saved} "
                f"semantic={result.semantic_updates_applied} "
                f"relations={result.relationship_changes}"
            )

        # 7. 最终快照
        print("─" * 70)
        print(f"FINAL    {await _snapshot(storage, aine_id, aine_name)}")
        print(f"         {await _snapshot(storage, rin_id, rin_name)}")
        print("─" * 70)

        # 8. 详细输出 memory / relation
        print("\n## 情节记忆")
        for char_id, char_name in [(aine_id, aine_name), (rin_id, rin_name)]:
            eps = await storage.get_episodic_memories(char_id)
            print(f"\n  {char_name}（{len(eps)} 条）:")
            for ep in eps:
                print(
                    f"    [{ep.importance:.2f}] valence={ep.emotional_valence:+.2f} "
                    f"{ep.content}"
                )

        print("\n## 语义记忆")
        for char_id, char_name in [(aine_id, aine_name), (rin_id, rin_name)]:
            sems = await storage.get_semantic_memories(char_id)
            print(f"\n  {char_name}（{len(sems)} 条）:")
            for sem in sems:
                print(f"    [conf={sem.confidence:.2f}] {sem.content}")

        print("\n## 关系")
        for char_id, char_name in [(aine_id, aine_name), (rin_id, rin_name)]:
            rels = await storage.list_relations(owner_id=char_id)
            for r in rels:
                print(
                    f"  {char_name} → {r.target_name}: closeness={r.closeness:.2f} "
                    f"trust={r.trust:.2f} type={r.relationship_type}"
                )
    finally:
        await storage.close()
        try:
            os.unlink(db_path)
        except OSError:
            pass


def main() -> None:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if turns < 1:
        print("turns 必须 >= 1", file=sys.stderr)
        sys.exit(1)
    asyncio.run(cross_chat(turns))


if __name__ == "__main__":
    main()
