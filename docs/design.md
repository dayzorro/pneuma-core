# 工程设计思路

本文档说明 Pneuma Core 「为什么这样设计」：核心设计原则、关键决策与权衡、扩展点，以及测试与质量策略。

## 目录

- [1. 设计原则](#1-设计原则)
- [2. 关键设计决策](#2-关键设计决策)
- [3. 一次对话的完整数据流](#3-一次对话的完整数据流)
- [4. 扩展思路](#4-扩展思路)
- [5. 测试与质量策略](#5-测试与质量策略)
- [6. 命名与约定](#6-命名与约定)

---

## 1. 设计原则

### 1.1 协议优先 / 依赖倒置

底层用 `Protocol` 定义契约（`LLMAdapter`、`EmbeddingService`、`StorageBackend`、`MemoryStore`…），上层只依赖抽象，不依赖具体实现。带来三个好处：

- **可替换**：换 LLM 厂商、换数据库都不需要改 runtime。
- **可测试**：测试里用 `AsyncMock` 或 `InMemoryStorageBackend` 即可，无需真实网络/磁盘。
- **可降级**：任何未实现的能力（如语音）都可留空而不阻塞主流程。

### 1.2 纯数据模型 + 边界校验

所有领域模型是 `frozen dataclass`，校验集中在 `__post_init__`：

```python
Personality(openness=1.5, ...)  # → ValueError，非法状态无法构造
```

「让非法状态无法被表示」使下游代码无需反复做防御性判断，也让测试可以聚焦在行为而非校验分支上。

### 1.3 降级容错（核心不因局部失败而中断）

对话是强交互场景，任何单点失败都不应让整段对话崩溃。因此 `process_message` 对**记忆检索、用户上下文检索、中间件、LLM 生成**的异常一律捕获为「警告 + 兜底」，仅对 `LLMTimeoutError` 重新抛出。

### 1.4 保持核心简单，用中间件扩展

核心 `RuntimeEngine` 只编排固定流水线；横切关注点（日志、观测、A/B、内容过滤）通过 `Middleware` 的 `pre_process` / `post_process` 注入，避免核心膨胀。

### 1.5 轻量每轮 + 重量每会话

高频路径（每轮）只做必要工作；昂贵的记忆整合与关系更新放在会话结束时一次性完成，兼顾**延迟/成本**与**内在深度**。

---

## 2. 关键设计决策

### 2.1 Big Five → PAD 基线

**问题**：如何让「性格」可计算地决定「情绪基调」，并让情绪在刺激后自然回归？

**决策**：用固定系数的线性映射把 Big Five 转成 PAD 基线，再用指数衰减把当前情感拉回基线。

```
P_baseline = 0.21·E + 0.59·A + 0.19·(1 - N)
A_baseline = 0.15·O + 0.30·(1 - A) + 0.57·N
D_baseline = 0.25·O + 0.17·C + 0.60·E - 0.32·A      （结果 clamp 到 [-1, 1]）
```

- **为什么线性**：系数可解释、可调参、无外部依赖、结果确定，便于测试与复现。
- **为什么衰减**：情感是短期状态，不应永久累积；半衰期参数（默认 3600s）给出可调的「情绪记忆时长」。
- **代价**：线性映射是简化模型，表达力有限。但作为默认基线，可解释性与稳定性优于「更花哨但不可控」的方案。

### 2.2 记忆检索的性格偏置

**问题**：如何让「性格」真实影响「想起什么」？

**决策**：混合评分 + 性格调制：

```
score = α·similarity + β·importance + γ·recency + negativity_bias
α = 0.5                       （相关度，常量）
β = 0.3 + 0.2·neuroticism     （神经质越高越看重重要性）
γ = 0.2                       （新近度，常量）
recency = exp(-0.01·天数)      （情节）；语义固定 1.0
negativity_bias = 0.15·neuroticism·|valence|  （仅负向情节记忆）
openness → effective_top_k = base + floor(0.5·openness·base)
```

- **设计理由**：
  - 高**神经质**：更易被重要事件、负向事件「抓住」——通过 β 与负向偏置体现。
  - 高**开放性**：检索面更广、更多样——通过扩大 top_k 而非改权重体现（多样性 ≠ 偏好）。
  - **共情性高**的角色偏向情感记忆，符合「共情 ↔ 情感显著」的直觉。
- **权衡**：参数是经验值，可在 `SearchConfig` 中调整；语义记忆不衰减，因为「事实」不像「经历」那样随时间褪色。

### 2.3 提示词静态 / 动态分离与缓存

**问题**：每轮都重建完整 system prompt，既费 token 又增延迟。

**决策**：`PromptCache` 把提示词切成静态区与动态区：

- **静态区**（角色未变时复用）：Profile、Personality、Values、Relations、UserContext Tier1、UserGoals、SpeakingStyle、ResponseFormat
- **动态区**（每轮重建）：UserContext Tier2+3、Memory、Goals、Tasks、DateTime、EmotionalState

配合 `ClaudeAdapter` 的 `cache_control` 块，静态区可在 API 层命中提示词缓存，仅对动态区付费。缓存以 `character.id` 为失效键。

**权衡**：多了一层静态/动态划分的复杂度，但换来显著的成本与延迟收益，且划分是有明确边界的。

### 2.4 情感评估异步化（次轮生效）

**决策**：普通模式下，情感评估作为**后台任务**发起，结果存入供**下一轮**使用；只有诊断模式才同步执行。

**权衡**：
- 优点：不阻塞本轮响应，降低用户可感知延迟。
- 代价：本轮返回的 `emotion` 是「进入本轮时的状态」，而非「本轮结束后的状态」——这是刻意的取舍（响应速度优先）。诊断模式提供同步、可观测的完整链路用于调试。

### 2.5 双阶段：Per-turn 轻、Per-session 重

**决策**：把「抽取情节/语义记忆、更新用户上下文、更新关系性」统一收敛到 `SessionEndPipeline`，用**一次 LLM 调用**产出四类结果。

**权衡**：
- 优点：高频路径更轻；整合分析能看到完整会话上下文，质量更高；调用次数少、成本低。
- 代价：记忆更新不是实时的，会话未结束前不落库。

### 2.6 三级用户上下文 + RAG

**决策**：用户信息按「总是/会话/按需」三级注入，并给出 token 预算（500 / 1000 / 2000）。更新按风险分层（LOW 自动 / MEDIUM 自动+上报 / HIGH 需批准），写入受白名单约束。

**理由**：用户画像信息量大但并非每轮都相关，分级注入能在「稳定人设」与「按需召回」之间取得平衡，同时用风险分级保护敏感写入。

### 2.7 结构化响应与兜底

**决策**：约定 LLM 输出 `speech / thought / action` 的结构化 JSON；解析失败时，把整段原始文本当作 `speech`。

**理由**：结构化字段让「内心（thought）」与「行为（action）」可与「台词（speech）」分离（例如历史只记录 speech）。宽容的兜底保证任意 LLM 都能工作。

### 2.8 存储抽象

**决策**：`StorageBackend`（领域 CRUD）与 `MemoryStore`（向量检索）分离；默认提供 `InMemory` 与 `SQLite` 两种实现。

**理由**：把「向量检索」这一可能昂贵/可替换的能力与普通 CRUD 解耦；SQLite 实现同时满足两个协议，兼顾开发便捷与生产可用。

---

## 3. 一次对话的完整数据流

```
用户消息
  │
  ▼
RuntimeEngine.process_message
  ├─ 加载 Character / EmotionalState / GoalTree（StorageBackend）
  ├─ 情感衰减：EmotionEngine.decay_towards_baseline（向 Big Five 基线）
  ├─ 记忆检索：embed(query) → MemorySearchEngine 混合评分 → top_k
  ├─ 用户上下文检索（Tier3 RAG，可选）
  ├─ 组装 system prompt：PromptCache（静态复用 + 动态重建）
  ├─ Middleware.pre_process（正序）
  ├─ LLMAdapter.generate（system + 历史）
  ├─ parse_structured_response → speech / thought / action
  ├─ 记录并保存 ChangeRecord
  ├─ 情感评估：EmotionEngine.estimate（后台任务，次轮生效）
  └─ Middleware.post_process（逆序）
  │
  ▼
MessageOutput(content, emotion, thought, action, internal_changes, diagnostic)

—— 会话结束 ——
SessionEndPipeline.run（1 次 LLM 调用）→ 情节记忆 / 语义记忆 / 用户上下文 / 关系性
```

---

## 4. 扩展思路

| 想做什么 | 扩展方式 |
| --- | --- |
| 换 LLM 厂商 | 实现 `LLMAdapter.generate`，无需改 runtime |
| 换 Embedding | 实现 `EmbeddingService.embed / embed_batch` |
| 换持久化 | 实现 `StorageBackend`（及可选 `MemoryStore`） |
| 加日志/观测/A-B | 写 `Middleware`，挂到 `middlewares=[...]` |
| 调情感/记忆行为 | 调整 `EmotionConfig` / `SearchConfig` 参数 |
| 接入新角色 | 写 `*.character.yaml`，用 `CharacterSheet.load_directory` 批量加载 |
| 接入语音 | 实现 `TTSAdapter` / `STTService` |

---

## 5. 测试与质量策略

- **测试驱动**：项目遵循 TDD（Red–Green–Refactor），`tests/` 覆盖各子系统，当前全量 **1044 项测试通过**。
- **不变式即规格**：`.vibe/spec/stories/*.yaml` 以 `invariants` 形式记录每条设计不变式，并关联到 `test` 与 `source_ref`。例如：
  - `pipeline-order`：每轮固定处理顺序
  - `personality-determines-pad-baseline`：PAD 基线由 Big Five 线性决定并 clamp
  - `personality-biased-recall`：神经质影响 importance 权重、开放性影响 top_k
  - `degraded-on-subsystem-failure`：局部失败降级继续
- **可替换依赖便于隔离测试**：用 `InMemoryStorageBackend` / `AsyncMock` 避免真实 IO。
- **静态检查**：`ruff`（配置见 `pyproject.toml`）。

运行方式：

```bash
# 安装（含开发依赖）
uv venv .venv && uv pip install --python .venv -e ".[dev]"
# 运行测试
.venv/bin/python -m pytest -q
# 运行示例（需要 API Key）
export ANTHROPIC_API_KEY=...   # 回复生成（Claude）
export OPENAI_API_KEY=...      # 嵌入（text-embedding-3-small）
.venv/bin/python examples/cross_chat.py 6
```

---

## 6. 命名与约定

- **目录 = 领域**：`emotion/`、`memory/`、`runtime/`、`storage/`… 每个目录聚合一个内聚领域。
- **models 层纯净**：只放数据结构，不做编排、不做 IO。
- **协议集中在 `protocols/`**：所有抽象接口与请求/响应 dataclass 汇总于此；`task/`、`voice/` 等为向后兼容的 re-export 层。
- **配置对象化**：可调参数用 `EmotionConfig` / `SearchConfig` / `SessionConfig` / `UserContextConfig` 等 frozen dataclass 承载，避免魔法数字散落。
- **vault 约定**：角色定义、用户上下文、运行时数据统一由 `vault.py` 按 `PNEUMA_VAULT_PATH`（默认 `./vault`）解析路径。
