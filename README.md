# Pneuma Core

为 AI 角色赋予「内心」的 Python 框架。

它提供这样一套机制：创建一个拥有性格、情感、记忆的角色，随着对话不断累积，角色与人之间的关系会逐渐成长。

## 特性

- **情感动态变化** —— 基于 PAD 三维模型 + Big Five 人格特质，情感会随对话内容实时变化。不同性格的角色反应各不相同，并会随时间自然衰减回基线。
- **按性格检索记忆** —— 通过带有 Big Five 偏置的 RAG 检索情节记忆与语义记忆。共情性高的角色会优先想起与情感相关的记忆，求知欲高的角色则优先想起事实性记忆。
- **每轮自动将角色内心注入提示词** —— 性格、当前情感、记忆、关系性都会被自动嵌入每一轮提示词。开发者无需再费心去做提示词工程。

## 安装

```bash
pip install pneuma-core
```

包含 LLM 适配器与后端服务：

```bash
pip install pneuma-core[all]
```

## 快速开始

### 1. 定义角色 (YAML)

```yaml
# aine.character.yaml
id: aine-001
name: アイネ
personality:
  openness: 0.9
  conscientiousness: 0.5
  extraversion: 0.3
  agreeableness: 0.8
  neuroticism: 0.6
values:
  self_transcendence: 0.3
  self_enhancement: 0.5
  openness_to_change: 0.8
  conservation: 0.2
profile: |
  内向的だけど好奇心が強い。
speaking_style: |
  丁寧だけど時々素が出る。
initial_state:
  pleasure: 0.0
  arousal: 0.0
  dominance: 0.0
  emotion_label: 中立
  situation: 初めての会話
```

### 2. 与角色对话

```python
import asyncio
from pathlib import Path

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.llm.claude import ClaudeAdapter
from pneuma_core.llm.embedding import OpenAIEmbeddingService
from pneuma_core.models.message import MessageInput
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.storage.sqlite import SQLiteStorageBackend

async def main() -> None:
    sheet = CharacterSheet.load(Path("aine.character.yaml"))
    character = sheet.character

    # SQLiteStorageBackend 同时实现了 StorageBackend 与 MemoryStore 两个协议
    storage = SQLiteStorageBackend("aine.db")
    await storage.initialize()
    await storage.save_character(character)
    if sheet.initial_state is not None:
        await storage.save_emotional_state(character.id, sheet.initial_state)

    engine = RuntimeEngine(
        character_id=character.id,
        storage=storage,
        llm=ClaudeAdapter(api_key="your-api-key"),
        embedding_service=OpenAIEmbeddingService(api_key="your-openai-api-key"),
        memory_store=storage,
    )

    output = await engine.process_message(
        MessageInput(
            content="最近読んだ本でおすすめある？",
            sender_id="user-1",
            sender_name="ユーザー",
            sender_type="human",
        )
    )
    print(output.content)
    print(output.emotion.emotion_label)

asyncio.run(main())
```

> 更完整的示例（两个角色自动对话，并观察情感 / 记忆 / 关系随轮次的变化）见 `examples/cross_chat.py`。

## 中间件

保持核心简单，通过中间件进行扩展。

```python
from pneuma_core.protocols.middleware import Middleware, PipelineContext
from pneuma_core.models.message import MessageInput, MessageOutput

class LoggingMiddleware:
    async def pre_process(self, message: MessageInput, context: PipelineContext) -> None:
        print(f"Received: {message.content}")

    async def post_process(
        self, message: MessageInput, output: MessageOutput, context: PipelineContext
    ) -> None:
        print(f"Response: {output.content}")

engine = RuntimeEngine(
    character_id=character.id,
    storage=storage,
    llm=llm,
    embedding_service=embedding_service,
    memory_store=storage,
    middlewares=[LoggingMiddleware()],
)
```

## 架构

Pneuma Core 以两个阶段运行：

**Per-turn（每轮）**：每当收到用户消息时，执行情感推断、记忆检索、上下文组装、LLM 调用、状态更新。借助中间件流水线，可以在各步骤前后插入自定义处理。

**Per-session（会话结束时）**：会话结束时，执行情节记忆的整合、语义记忆的抽取、关系性的更新、日记的生成。

```
Layer 0 (models)   : 数据模型 + 存储协议
Layer 1 (runtime)  : LLM 集成 + 情感引擎 + 记忆检索 + 中间件
```

更详细的架构与设计思路见 [`docs/`](docs/README.md)。

## 后端对话服务

仓库自带一个 FastAPI 的单用户 HTTP 服务：**设定「我是谁」之后就可以正常对话**。

```bash
cp .env.example .env     # 填入你的 API 配置
set -a && . ./.env && set +a
uv pip install -e ".[server]"
python -m pneuma_core.server
```

```bash
BASE=http://localhost:8001
# 1) 设定我是谁
curl -X POST $BASE/api/session/start -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","user_name":"太郎"}'
# 2) 正常对话
curl -X POST $BASE/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"最近読んだ本でおすすめある？"}'
# 3) 结束会话（触发记忆整合）
curl -X POST $BASE/api/session/end
```

接口：`/healthz`、`/api/character`、`/api/session/start`、`/api/chat`、`/api/session/end`、`/api/state`。

环境变量清单与完整说明见 [docs/service.md](docs/service.md)。

> 说明：LLM 与 Embedding 均走 **OpenAI 兼容端点**（如阿里百炼 / DeepSeek / OpenRouter）。若 `PNEUMA_EMBEDDING_*` 未单独设置，会复用 `PNEUMA_LLM_*` 的 base_url 与 key。

## 许可证

MIT
