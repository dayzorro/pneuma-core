# 后端对话服务

Pneuma Core 自带一个基于 FastAPI 的单用户 HTTP 对话服务：**先设定「我是谁」，之后就像正常聊天一样发消息即可**。

- 单用户：服务内部维护一个活动会话（`session`），无需每次传 user_id。
- 全 OpenAI 兼容：LLM 与 Embedding 都走任意 OpenAI 兼容端点（阿里百炼 / DeepSeek / OpenRouter / vLLM / LM Studio…）。
- 记忆在「会话结束」时由 LLM 统一整合（情节记忆 / 语义记忆 / 关系性）。

---

## 1. 环境变量清单

完整示例见仓库根目录的 [`.env.example`](../.env.example)。复制为 `.env` 后填入真实值：

```bash
cp .env.example .env
```

### 必填

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `PNEUMA_LLM_API_KEY` | LLM API Key | `sk-...` |
| `PNEUMA_LLM_BASE_URL` | LLM 的 OpenAI 兼容端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `PNEUMA_LLM_MODEL` | 对话模型名 | `qwen-plus` / `deepseek-chat` |

> 说明：`PNEUMA_LLM_BASE_URL` 若留空则使用 OpenAI SDK 默认的 `https://api.openai.com/v1`（国内通常不可达，建议显式配置）。

### Embedding（用于记忆检索）

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `PNEUMA_EMBEDDING_BASE_URL` | 向量端点，**留空则复用 `PNEUMA_LLM_BASE_URL`** | 同上 |
| `PNEUMA_EMBEDDING_API_KEY` | 向量 API Key，**留空则复用 `PNEUMA_LLM_API_KEY`** | 同上 |
| `PNEUMA_EMBEDDING_MODEL` | 向量模型名 | `text-embedding-v3` |

> 注意：**Embedding 是可选项但不是可省略项**。若端点不可用，记忆检索会失败——框架会降级继续对话（返回 `system_messages` 警告），但角色不会「想起」任何记忆。请务必配置一个可用的向量模型。

### 服务

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PNEUMA_HOST` | `0.0.0.0` | 监听地址 |
| `PNEUMA_PORT` | `8001` | 监听端口（避免与已占用的 8000 冲突） |

### 角色与存储

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PNEUMA_CHARACTER_FILE` | `examples/aine.character.yaml` | 角色卡 YAML |
| `PNEUMA_DB_PATH` | `vault/pneuma.db` | SQLite 路径（状态 / 记忆 / 关系持久化） |
| `PNEUMA_USER_CONTEXT_DIR` | 未设置 | 可选的用户上下文目录（三级用户上下文 / RAG） |

### 行为调优

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PNEUMA_HISTORY_LIMIT` | `30` | 历史消息上限，超出后由 LLM 摘要压缩 |
| `PNEUMA_DIAGNOSTIC` | `false` | `true`=每轮同步评估情感（更准但每轮多一次 LLM 调用）；`false`=后台异步评估，回复里带的是「本轮开始时」的情感 |
| `PNEUMA_LLM_TIMEOUT` | `60` | LLM 请求超时（秒） |

---

## 2. 启动

```bash
# 安装（含服务依赖）
uv pip install -e ".[server]"
# 或：pip install "pneuma-core[server]"

# 加载 .env 并启动
set -a && . ./.env && set +a
python -m pneuma_core.server
# 等价于安装了控制台脚本时： pneuma-server
```

启动后访问 API 文档：<http://localhost:8001/docs>

---

## 3. API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/healthz` | 健康检查 |
| `GET` | `/api/character` | 当前角色定义（名称 / 人设 / 性格） |
| `POST` | `/api/session/start` | **设定「我是谁」** 并开启会话 |
| `POST` | `/api/chat` | 发送一条消息，获取回复 |
| `POST` | `/api/session/end` | 结束会话并整合记忆 |
| `GET` | `/api/state` | 当前情感 / 记忆数量 / 关系性 |

### 完整交互示例

```bash
BASE=http://localhost:8001

# 1) 设定我是谁
curl -X POST $BASE/api/session/start \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","user_name":"小明"}'
# → {"session_id":"sess-...","user":{...},"character":{"name":"夏澜",...}}

# 2) 正常对话
curl -X POST $BASE/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"最近读的书里有什么推荐吗？"}'
# → {"reply":"...","thought":"...","action":null,
#    "emotion":{"pleasure":..,"arousal":..,"dominance":..,"label":".."},
#    "system_messages":[]}

# 3) 查看状态
curl $BASE/api/state

# 4) 结束会话（触发记忆整合）
curl -X POST $BASE/api/session/end
# → {"success":true,"episodic_memories_saved":..,"semantic_updates_applied":..,
#    "user_context_updates":..,"relationship_changes":..}
```

### 响应字段说明（`/api/chat`）

| 字段 | 含义 |
| --- | --- |
| `reply` | 角色的台词（若模型返回结构化 JSON，则取 `speech`） |
| `thought` | 角色的内心独白（模型的 `thought`） |
| `action` | 角色的动作（模型的 `action`） |
| `emotion` | 本轮的情感（PAD + 离散标签）；非诊断模式下为**本轮开始时**的状态 |
| `system_messages` | 降级提示，例如记忆检索失败、LLM 调用失败等 |

### 错误码

| 状态码 | 场景 |
| --- | --- |
| `409` | 未调用 `/api/session/start` 就 `/api/chat` 或 `/api/session/end` |
| `422` | 请求体校验失败（如 `message` 为空） |

---

## 4. 行为说明与注意事项

- **记忆何时写入**：情节/语义记忆与关系性在 `/api/session/end` 时由 LLM 一次性整合。会话未结束前不会落库，服务崩溃会丢失当前会话的未整合内容。
- **重复 start**：再次调用 `/api/session/start` 会先自动结束上一个会话（触发记忆整合），再开启新会话。
- **情感标签的滞后**：默认（`PNEUMA_DIAGNOSTIC=false`）为了降低延迟，情感评估在后台异步执行，因此 `/api/chat` 返回的 `emotion` 是「本轮开始时」的状态；想拿「本轮结束后」的实时情感，请查 `/api/state`，或开启 `PNEUMA_DIAGNOSTIC=true`。
- **模型名回退**：框架内部有少量硬编码的 Claude 模型名（如历史摘要、会话结束分析）。OpenAI 兼容适配器会自动把 `claude-*` 回退为你配置的 `PNEUMA_LLM_MODEL`，无需额外配置。
- **单用户**：服务只维护一个活动会话。若要多人隔离，需要把 `session` 与 `RuntimeEngine` 按用户分桶（当前版本未实现）。

---

## 5. 目录结构

```
src/pneuma_core/server/
├── config.py     # ServerConfig：从环境变量构建配置
├── service.py    # ChatService：存储/LLM/Embedding/Engine/会话生命周期
├── app.py        # FastAPI 路由与请求模型
├── __init__.py
└── __main__.py   # python -m pneuma_core.server
```

相关适配器：

- `src/pneuma_core/llm/openai_compat.py` —— `OpenAICompatAdapter`（OpenAI 兼容 LLM）
- `src/pneuma_core/llm/embedding.py` —— `OpenAIEmbeddingService`（支持 `base_url`）
