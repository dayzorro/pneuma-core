"""交互时延埋点：按阶段采集耗时并统计 P50 / P95。

只统计「一轮对话」关键路径上的耗时（远程调用 + 本地计算）。后台任务
（情绪评估、认知提炼、历史摘要）不计入，因为它们不占用用户等待时间。

用法::

    metrics = LatencyMetrics()
    metrics.observe({"embedding": 132.4, "llm_first_token": 486.1, ...})
    metrics.snapshot()   # -> {"turns": N, "stages": {...P50/P95...}}

阶段名（snake_case）与含义：

    ======================  ==========================================
    state                   角色 / 情绪 / 目标加载（本地存储）
    embedding               query 向量化（远程）
    retrieve_memory         记忆检索（并发分支之一）
    retrieve_user_context   用户上下文检索（并发分支之一）
    retrieve_knowledge      知识库检索（并发分支之一）
    retrieve_wait           三路并发检索的整体等待
    web_search              联网检索（关闭时不产生样本）
    prompt_build            提示词构建
    preprocess              中间件 pre_process
    history                 历史追加与裁剪
    llm_first_token         首字延迟（仅流式路径）
    llm_total               主 LLM 完整生成
    post_process            中间件 post_process
    total                   整轮（进入 process_message → 返回）
    ======================  ==========================================
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

__all__ = ["STAGE_LABELS", "LatencyMetrics"]

# 阶段名 → 中文标签（便于接口直接展示）
STAGE_LABELS: dict[str, str] = {
    "state": "状态加载",
    "embedding": "Query 向量化",
    "retrieve_memory": "记忆检索",
    "retrieve_user_context": "用户上下文检索",
    "retrieve_knowledge": "知识库检索",
    "retrieve_wait": "并发检索（整体等待）",
    "web_search": "联网检索",
    "prompt_build": "提示词构建",
    "preprocess": "中间件前置",
    "history": "历史追加/裁剪",
    "llm_first_token": "首字延迟",
    "llm_total": "主 LLM 生成",
    "post_process": "中间件后置",
    "total": "整轮",
}

def _percentile(sorted_values: list[float], percent: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认口径一致）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = percent / 100.0 * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    return sorted_values[low] * (high - rank) + sorted_values[high] * (rank - low)


class LatencyMetrics:
    """按阶段累积样本，输出 P50 / P95 / P99。

    样本按阶段各自维护一个定长滑动窗口（保留最近 ``max_turns`` 轮），
    因此统计反映的是「近期线上表现」，不会被子进程早期的脏数据长期污染。
    """

    def __init__(self, max_turns: int = 500) -> None:
        self._max_turns = max(1, max_turns)
        self._samples: dict[str, deque[float]] = {}
        self._turns = 0

    def observe(self, timings: Mapping[str, float | None]) -> None:
        """记录一轮各阶段耗时（毫秒）。值为 None 的阶段忽略。"""
        self._turns += 1
        for stage, value in timings.items():
            if value is None:
                continue
            buffer = self._samples.get(stage)
            if buffer is None:
                buffer = deque(maxlen=self._max_turns)
                self._samples[stage] = buffer
            buffer.append(float(value))

    def reset(self) -> None:
        """清空全部样本。"""
        self._samples.clear()
        self._turns = 0

    def snapshot(self) -> dict:
        """返回当前分位数快照。"""
        stages: dict[str, dict] = {}
        for stage, buffer in self._samples.items():
            values = sorted(buffer)
            if not values:
                continue
            entry: dict = {
                "label": STAGE_LABELS.get(stage, stage),
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 1),
                "p95_ms": round(_percentile(values, 95), 1),
                "p99_ms": round(_percentile(values, 99), 1),
                "avg_ms": round(sum(values) / len(values), 1),
                "max_ms": round(values[-1], 1),
            }
            stages[stage] = entry

        return {
            "turns": self._turns,
            "window": self._max_turns,
            "unit": "ms",
            "stages": stages,
        }
