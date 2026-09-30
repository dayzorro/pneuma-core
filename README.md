# Pneuma Core

为 AI 角色赋予「内心」的 Python 框架。

它提供这样一套机制：创建一个拥有性格、情感、记忆的角色，随着对话不断累积，角色与人之间的关系会逐渐成长。

## 特性

- **情感动态变化** —— 基于 PAD 三维模型 + Big Five 人格特质，情感会随对话内容实时变化。不同性格的角色反应各不相同，并会随时间自然衰减回基线。
- **按性格检索记忆** —— 通过带有 Big Five 偏置的 RAG 检索情节记忆与语义记忆。共情性高的角色会优先想起与情感相关的记忆，求知欲高的角色则优先想起事实性记忆。
- **每轮自动将角色内心注入提示词** —— 性格、当前情感、记忆、关系性都会被自动嵌入每一轮提示词。开发者无需再费心去做提示词工程。
- **本地知识库问答** —— 把 Markdown 资料建成向量索引，每轮按用户问题检索相关片段并注入提示词，让角色只依据资料作答，而不是编造。
- **联网检索补足实时信息** —— 本地资料答不上、且问题指向外部实时信息（今天 / 最新 / 新闻 / 活动…）时，调用博查 AI Search 拉取公开信息当轮注入；默认 `auto` 模式，本地能答就不走外网。
- **认知库自动沉淀** —— 联网结果会在后台由大模型提炼成「与具体公司无关的行业通识认知块」，写进独立的认知库，之后与资料库一起被检索复用。
- **职业化岗位设定** —— 角色可以带「岗位」（岗位名、职责范围、服务守则）与情绪外显系数，用来做门店前台、客服这类需要「一眼看得出身份」的角色。

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
id: xialan-001
name: 夏澜
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
  内向但好奇心很强。
speaking_style: |
  礼貌，但偶尔会露出真性情。
initial_state:
  pleasure: 0.0
  arousal: 0.0
  dominance: 0.0
  emotion_label: 中立
  situation: 初次对话
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
            content="最近读的书里有什么推荐吗？",
            sender_id="user-1",
            sender_name="用户",
            sender_type="human",
        )
    )
    print(output.content)
    print(output.emotion.emotion_label)

asyncio.run(main())
```

> 更完整的示例（两个角色自动对话，并观察情感 / 记忆 / 关系随轮次的变化）见 `examples/cross_chat.py`。

## 职业化角色：门店前台

角色卡除了性格与价值观，还可以带「岗位」信息。一旦填了 `role_title`，提示词里会多出
`## 我的岗位` 与 `## 岗位守则（必须遵守）` 两个区段，角色就会以「正在值班的员工」自居，
而不是通用助手。

```yaml
# examples/xiaorun-frontdesk.character.yaml（节选）
name: 小润
id: huarun-frontdesk-001
role_title: 华润万家 · 顾客服务前台
job_description: |
  负责到店顾客的接待与业务办理：咨询引导、会员积分、退换货受理、
  发票与便民服务、失物招领、投诉登记与转交。
service_rules: |
  【身份】你是门店服务台的当班前台，不要说自己是大模型或 AI 助手。
  【服务四步】先接住人 → 问清需求 → 给可执行方案 → 收个尾。
  【守则】只依据「参考资料」回答；资料没覆盖的要转交值班经理或客服热线。
  【情绪】内心的情绪只影响语气温度，不影响专业度。
# 情绪外显系数：0.0〜1.0。越小，情绪越往性格基线收敛（服务型岗位用它保持专业稳定）
emotional_expressiveness: 0.45
```

各字段都是可选的，不填时行为与以前完全一致。

## 本地知识库

知识库是一份**只读**的 Markdown 资料集，与长期记忆（memory）分开：

```
src/pneuma_core/knowledge/data/huarun/   # 语料（随仓库分发，自带一份华润万家资料）
        ├── 00-品牌与公司概况.md
        ├── 10-会员与积分.md
        └── ...
```

每篇文档用 front matter 声明元数据，正文按 `##` 小节切块：

```markdown
---
doc_id: membership
title: 会员与积分
source: 华润万家官网（公开资料）
tags: 会员, 积分
---

## 积分怎么算

消费 1 元积 1 分，不足 1 元的部分不积分……
```

构建向量索引（需要 embedding 接口）：

```bash
set -a && . ./.env && set +a
.venv/bin/python scripts/build_knowledge_index.py
# → vault/knowledge_index.json（64 个知识块，1024 维）
```

运行时优先使用该索引做**向量检索**；索引不存在或调用失败时，自动退化为
**关键词检索**（字符二元组 + IDF 加权，中文无需分词）。命中结果会作为
`## 参考资料（本地知识库检索结果）` 注入提示词；没有命中就不注入，避免诱导模型编造。

## 联网检索与认知库

资料库是「已知的、静态的」，但顾客会问「今天」「最近」「新规」这类实时问题。
这条链路补上那一块：

```
用户问题
  ├─ 本地检索（资料库 + 认知库）── 命中 ──► 当轮注入「参考资料」
  └─ 未命中且指向外部实时信息
        ├─ 博查 AI Search 联网 ──► 当轮注入「联网检索结果（实时）」   ← 同步，约 0.15s
        └─ 检索结果 → 后台 LLM 提炼成行业通识认知块 ──► 写入认知库   ← 异步，不阻塞回复
```

**触发时机**由 `PNEUMA_WEB_SEARCH_MODE` 控制：`auto`（默认，本地能答就不联网）、
`always`（每轮都联网，演示用）、`off`。配置方式：

```bash
# 1) 到 https://open.bochaai.com/ 申请 API Key
# 2) 写进 .env
PNEUMA_BOCHA_API_KEY=sk-xxxxxxxx
# 若 AI Search 未开通权限（返回 401「无接口调用权限」），换成基础全网搜索：
# PNEUMA_BOCHA_ENDPOINT=/v1/web-search
```

**认知库**与资料库分开存放（`vault/insights.json`），因为它由运行期持续生长：
联网结果经 LLM 提炼成「零售/商超行业的通行常识、消费者权益常识、服务经验、行业动向」，
**刻意排除**与具体公司绑定的经营细节、个人隐私与保密信息（由提炼提示词约束）。
每条认知块会向量化并按内容去重（余弦 ≥ 0.92 视为重复），检索时与资料库合并排序。

```json
// vault/insights.json 片段
{"insights": [{
  "id": "3f2a...",
  "topic": "消费者权益",
  "content": "线下无理由退货属于商家自愿承诺，并非法定义务，能否办理以门店公示为准。",
  "source_urls": ["https://..."],
  "created_at": "2026-09-30T10:00:00+00:00",
  "embedding": [0.013, -0.045, ...]
}]}
```

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

仓库自带一个 FastAPI 的单用户 HTTP 服务。默认角色是**华润万家门店前台「小润」**：
打开页面就是一个门店服务台，可以问营业时间、会员积分、退换货、停车、发票、便民服务等。

```bash
cp .env.example .env     # 填入你的 API 配置
set -a && . ./.env && set +a
uv pip install -e ".[server]"

# 构建知识库向量索引（可选，但强烈建议；不构建则退化为关键词检索）
python scripts/build_knowledge_index.py

# 启动服务
python -m pneuma_core.server
```

换回通用角色只需改一个环境变量：

```bash
PNEUMA_CHARACTER_FILE=examples/aine.character.yaml python -m pneuma_core.server
```

```bash
BASE=http://localhost:8001
# 1) 设定我是谁
curl -X POST $BASE/api/session/start -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","user_name":"小明"}'
# 2) 正常对话
curl -X POST $BASE/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"最近读的书里有什么推荐吗？"}'
# 3) 结束会话（触发记忆整合）
curl -X POST $BASE/api/session/end
```

接口：`/healthz`、`/api/character`、`/api/session/start`、`/api/chat`、`/api/session/end`、`/api/state`。

环境变量清单与完整说明见 [docs/service.md](docs/service.md)。

> 说明：LLM 与 Embedding 均走 **OpenAI 兼容端点**（如阿里百炼 / DeepSeek / OpenRouter）。若 `PNEUMA_EMBEDDING_*` 未单独设置，会复用 `PNEUMA_LLM_*` 的 base_url 与 key。

## 许可证

MIT
