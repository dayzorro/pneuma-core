# Pneuma Core 文档

本目录存放 Pneuma Core 的架构说明与工程设计思路文档。

| 文档 | 内容 |
| --- | --- |
| [architecture.md](./architecture.md) | 系统架构：分层、目录结构、数据模型、Per-turn / Per-session 运行阶段、各子系统职责 |
| [design.md](./design.md) | 工程设计思路：设计原则、关键决策与权衡、扩展点、测试与质量策略 |

## 项目一句话简介

Pneuma Core 是一个为 AI 角色赋予「内心」的 Python 框架：角色拥有 Big Five 性格、PAD 情感状态、情节/语义记忆与关系性，框架在每一轮对话中自动将这些「内心」注入提示词，并在会话结束时做记忆与关系的整合。

## 阅读顺序建议

1. 先读 [architecture.md](./architecture.md) 建立整体地图；
2. 再读 [design.md](./design.md) 理解「为什么这样设计」；
3. 结合源码 `src/pneuma_core/` 与示例 `examples/` 加深理解。
