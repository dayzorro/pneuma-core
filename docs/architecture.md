# 架构说明

本文档描述 Pneuma Core 的整体架构：分层、目录结构、核心数据模型、运行阶段以及各子系统职责。

## 1. 定位

Pneuma Core 是一个**库（library）**，不是一个应用或服务。它的目标是提供一套「角色引擎」的骨架：

- 输入：角色定义（YAML）+ 用户消息 + 外部依赖（LLM、Embedding、存储后端）
- 输出：带有情感、思考、动作、内部变更记录的回复

框架本身不绑定任何具体 LLM 厂商或部署形态，具体能力通过 Protocol（协议）由外部实现注入。

## 2. 分层设计

项目采用清晰的两层结构，依赖方向始终**从上到下**（上层依赖下层，下层不知道上层）。

```
Layer 1 (runtime)  : LLM 集成 + 情感引擎 + 记忆检索 + 提示词构建 + 中间件 + 会话管理
      │  依赖
      ▼
Layer 0 (models)   : 纯数据模型 + 存储/服务的抽象协议（Protocol）
```

| 层 | 职责 | 关键包 |
| --- | --- | --- |
| **Layer 0 — models / protocols** | 定义共享的数据结构与抽象接口，不含业务编排 | `models/`、`protocols/`、`emotion/`、`memory/` 中的纯计算 |
| **Layer 1 — runtime** | 编排一次对话的完整流水线，串联情感、记忆、提示词与 LLM | `runtime/`、`storage/`、`llm/` |

在这两层之下，还有一组**具体实现**（存储、LLM、Embedding），它们实现 Layer 0 定义的协议，可被自由替换。

## 3. 目录结构

```
src/pneuma_core/
├── __init__.py
├── character_sheet.py     # YAML ↔ 数据模型 的读写（角色卡）
├── vault.py               # vault 路径解析（PNEUMA_VAULT_PATH）
├── exceptions.py          # 领域异常（LLMTimeoutError 等）
├── models/                # 【Layer 0】纯数据模型（frozen dataclass）
│   ├── character.py       #   Character
│   ├── personality.py     #   Personality（Big Five，0.0〜1.0）
│   ├── values.py          #   Values（Schwartz 四维）
│   ├── emotion.py         #   EmotionalState / Mood
│   ├── memory.py          #   EpisodicMemory / SemanticMemory
│   ├── goals.py           #   GoalTree / Vision / Objective / Task
│   ├── message.py         #   MessageInput / MessageOutput / StructuredResponse
│   ├── relation.py        #   Relation（关系性）
│   ├── todo.py            #   TodoItem
│   ├── change_record.py   #   ChangeRecord（内部变更日志）
│   ├── entity.py          #   EntityContext（角色/用户）
│   └── diagnostic.py      #   DiagnosticInfo（诊断模式）
├── protocols/             # 【Layer 0】抽象协议，只定义契约
│   ├── llm.py             #   LLMAdapter / LLMRequest / LLMResponse
│   ├── embedding.py       #   EmbeddingService
│   ├── storage.py         #   StorageBackend（Character/Memory/Goals/… 的 CRUD）
│   ├── memory_store.py    #   MemoryStore（向量相似检索）
│   ├── middleware.py      #   Middleware / PipelineContext
│   ├── task.py            #   TaskBackend
│   └── voice.py           #   STTService / TTSAdapter
├── emotion/               # 【纯计算】情感模型
│   ├── baseline.py        #   Big Five → PAD 基线
│   ├── decay.py           #   指数衰减
│   └── pad_mapping.py     #   PAD → 离感情感标签
├── memory/                # 【纯计算】记忆检索与整合
│   ├── similarity.py      #   余弦相似度
│   ├── search.py          #   MemorySearchEngine（混合评分）
│   ├── store.py           #   记忆存储辅助
│   ├── consolidator.py    #   MemoryConsolidator（情节→长期）
│   └── semantic_consolidator.py
├── llm/                   # 【实现】外部 LLM / Embedding 适配器
│   ├── claude.py          #   ClaudeAdapter
│   ├── embedding.py       #   OpenAIEmbeddingService
│   └── adapter.py         #   向后兼容 re-export
├── storage/               # 【实现】存储后端
│   ├── backend.py         #   re-export
│   ├── in_memory.py       #   InMemoryStorageBackend（测试用）
│   └── sqlite.py          #   SQLiteStorageBackend（持久化）
├── runtime/               # 【Layer 1】运行时编排
│   ├── engine.py          #   RuntimeEngine —— 消息处理主流水线
│   ├── emotion_engine.py  #   EmotionEngine —— LLM 情感评估 + 衰减
│   ├── prompt_builder.py  #   PromptBuilder —— 组装 system prompt
│   ├── prompt_cache.py    #   PromptCache —— 静态/动态分段与复用
│   ├── context_assembler.py # ContextAssembler —— 三段式上下文注入
│   ├── response_parser.py #   结构化响应解析（speech/thought/action）
│   ├── session.py         #   SessionManager —— 会话与超时
│   ├── session_end_pipeline.py # 会话结束时的整合分析
│   ├── history_summarizer.py   # 历史溢出时的 LLM 摘要压缩
│   ├── middleware.py      #   中间件执行辅助
│   ├── proactive.py       #   ProactiveEngine —— 先回头发话
│   ├── diary_writer.py / diary_coaching.py / diary_processor.py
│   └── user_context*.py   #   用户上下文的三级加载 / 检索 / 更新
├── task/  voice/          # 领域包的向后兼容 re-export
```

## 4. 核心数据模型

所有模型都是 **frozen dataclass**，值域校验集中在 `__post_init__`，保证「非法状态不可构造」。

| 模型 | 说明 | 关键约束 |
| --- | --- | --- |
| `Character` | 角色身份 | 聚合 personality / values / profile / speaking_style |
| `Personality` | Big Five 人格 | 每维 0.0〜1.0；`is_high≥0.7 / is_low<0.3` |
| `Values` | Schwartz 价值观四维 | 每维 0.0〜1.0 |
| `EmotionalState` | 当前情感（PAD） | pleasure/arousal/dominance ∈ [-1.0, 1.0] |
| `Mood` | 长期心情（EMA） | `new = (1-α)·current + α·emotion` |
| `EpisodicMemory` | 情节记忆 | importance ∈ [0,1]，emotional_valence ∈ [-1,1]，带 timestamp |
| `SemanticMemory` | 语义记忆 | confidence ∈ [0,1]，无时间衰减 |
| `GoalTree` | 目标层级 | Vision → Objective → Task，按 id 解析层级 |
| `MessageInput / Output` | 消息输入/输出 | sender_type ∈ {human, character, system} |
| `Relation` | 关系性 | closeness / trust |
| `ChangeRecord` | 内部变更日志 | 记录情感等状态的前后值 |

> 设计要点：模型层**不依赖** runtime，也不含 IO；这样模型可以被任何上层复用，且易于测试。

## 5. 运行阶段

Pneuma Core 区分两个时间尺度的处理阶段。

### 5.1 Per-turn（每轮消息）

入口：`RuntimeEngine.process_message(msg)`，固定顺序如下（对应 `runtime.yaml` 中的 `pipeline-order` 不变式）：

```
1.   加载角色状态        character / emotional_state / goals
1.5  情感衰减            距上次更新超过阈值时，向性格基线指数衰减
2.   检索相关记忆        情节 + 语义，混合评分
2.5  检索用户上下文      RAG（Tier 3，可选）
3.   构建 system prompt  PromptCache（静态复用）/ PromptBuilder
3.5  中间件 pre_process  正序执行，可改写消息与上下文
4.   写入用户消息        加入历史并裁剪
5.   调用 LLM            生成回复 → 解析结构化响应
6.   记录情感变更        ChangeRecord
7.   保存变更
8.   情感评估            诊断模式：同步；普通模式：后台任务，次轮生效
      + 中间件 post_process（逆序执行）
输出 MessageOutput       content / emotion / thought / action / changes / diagnostic
```

关键特性：**降级容错（degraded mode）**。记忆检索、用户上下文检索、中间件、乃至 LLM 生成失败时，都会捕获异常、追加 `SystemMessage` 警告，并以兜底内容继续处理；只有 `LLMTimeoutError` 会被重新抛出。

### 5.2 Per-session（会话结束时）

入口：`SessionEndPipeline.run(...)`。**一次 LLM 调用**同时产出四类结果，分别应用到对应存储：

```
输入：整段会话
  │  单次 LLM 调用（结构化 JSON）
  ▼
├── episodic_memories      → 情节记忆
├── semantic_updates       → 语义记忆（add / modify / delete）
├── user_context_updates   → 用户上下文文件（rewrite / append）
└── relationship_changes   → 关系性（closeness / trust 增量）
```

> 与之配套的还有 `MemoryConsolidator` / `SemanticConsolidator` 等更细粒度的整合路径，用于按重要度阈值、去重、嵌入生成等步骤处理记忆。

## 6. 关键子系统

### 6.1 情感系统（`emotion/` + `runtime/emotion_engine.py`）

- **人格 → 情感基线**：Big Five 线性映射为 PAD 基线（见 [design.md](./design.md#1-big-five--pad-基线)）。
- **自然衰减**：`exponential_decay` 让情感值以半衰期向基线回归。
- **LLM 情感评估**：`EmotionEngine.estimate` 每轮用 LLM 直接输出 PAD 值与标签；解析失败或异常时返回中立态。
- **标签映射**：`pad_to_emotion_label` 用 8 个八分象限 + 中立（阈值 0.3）把连续 PAD 转成离散文案。

### 6.2 记忆系统（`memory/`）

- **混合评分检索** `MemorySearchEngine`：
  `score = α·similarity + β·importance + γ·recency + negativity_bias`
  - α=0.5；β=0.3 + 0.2·neuroticism；γ=0.2
  - recency：情节 `exp(-λ·天数)`，语义固定为 1.0
  - negativity_bias：仅对负向情节记忆生效，随 neuroticism 增加
  - **openness 影响检索多样性**（扩大有效 top_k），而非评分权重
- **余弦相似度**：维数不一致抛错，零向量返回 0.0；无 embedding 的记忆得分 0（实质被排除）。

### 6.3 提示词构建与缓存（`runtime/prompt_builder.py` / `prompt_cache.py`）

`PromptBuilder` 将**多路上下文**组装成 system prompt，分区包括：

```
Profile / Personality / Values / Relations /
UserContext(三级) / Memory / Goals / UserGoals / Tasks /
DateTime / State(PAD) / SpeakingStyle / ResponseFormat
```

`PromptCache` 将分区拆成：

- **静态区**（角色变化前不变，可被 API 层缓存）：Profile、Personality、Values、Relations、UserContext Tier1、UserGoals、SpeakingStyle、ResponseFormat
- **动态区**（每轮重建）：UserContext Tier2+3、Memory、Goals、Tasks、DateTime、EmotionalState

### 6.4 会话与用户上下文（`runtime/session.py`、`runtime/user_context*.py`）

- **SessionManager**：以 `channel_type:channel_id:user_id` 为键管理会话，超时自动结束并触发 `on_session_end` 回调。
- **三级用户上下文**：Tier1 始终注入（身份），Tier2 会话级，Tier3 按 RAG 检索（token 预算默认 500 / 1000 / 2000）。
- **用户上下文更新按风险分层**：LOW 自动应用 / MEDIUM 自动应用并上报 / HIGH 需审批；写入受文件名白名单限制。

### 6.5 存储与协议（`protocols/`、`storage/`）

- `StorageBackend`：统一的 CRUD 接口，覆盖 Character / Memory / Goals / State / ChangeLog / Todo / Relation。
- `MemoryStore`：**与存储分离**，专门定义向量相似检索（`find_similar_episodic`）。
- 提供两种实现：`InMemoryStorageBackend`（测试）与 `SQLiteStorageBackend`（持久化，初始化时执行 schema migration）。
- `SQLiteStorageBackend` 同时满足两个协议，可直接作为 `memory_store` 传入。

### 6.6 LLM 与 Embedding 适配器（`llm/`）

- `ClaudeAdapter`：仅对 429 / 5xx 做指数退避重试；超时转换为 `LLMTimeoutError`；支持 `cache_control` 的静态提示词缓存块。
- `OpenAIEmbeddingService`：`embed_batch` 会过滤空字符串并以零向量占位，保持索引对齐。

## 7. 扩展点

框架刻意保持核心简单，扩展主要通过以下方式：

1. **Protocol 实现**：自定义 `LLMAdapter` / `EmbeddingService` / `StorageBackend` / `MemoryStore` / `TaskBackend` / `TTS·STT`。
2. **中间件**：实现 `pre_process`（正序）与 `post_process`（逆序），通过 `PipelineContext` 共享/改写上下文。
3. **角色卡（YAML）**：`CharacterSheet` 支持 personality / values / initial_state / goals / voice_id 的声明式定义。

## 8. 相关文件

- 每轮流水线：[`src/pneuma_core/runtime/engine.py`](../src/pneuma_core/runtime/engine.py)
- 会话结束整合：[`src/pneuma_core/runtime/session_end_pipeline.py`](../src/pneuma_core/runtime/session_end_pipeline.py)
- 情感计算：[`src/pneuma_core/emotion/`](../src/pneuma_core/emotion)
- 记忆检索：[`src/pneuma_core/memory/search.py`](../src/pneuma_core/memory/search.py)
- 结构规格（Story / 不变式）：[`.vibe/spec/`](../.vibe/spec)
