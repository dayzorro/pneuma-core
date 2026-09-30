"""判断一轮对话是否需要联网检索。

前台能回答的问题优先用本地资料（人工整理的门店/品牌知识库 + 沉淀下来的
行业认知库）；只有本地答不上、且问题本身指向外部实时信息时，才走联网。

三种模式（``PNEUMA_WEB_SEARCH_MODE``）：
    - ``auto``（默认）：本地知识库没有命中，且问题带有时效/外部信息特征时联网
    - ``always``：每轮都联网（演示用）
    - ``off``：完全不联网
"""

from __future__ import annotations

import re

MODE_OFF = "off"
MODE_AUTO = "auto"
MODE_ALWAYS = "always"

VALID_MODES = frozenset({MODE_OFF, MODE_AUTO, MODE_ALWAYS})
DEFAULT_MODE = MODE_AUTO

# 指向「外部实时信息 / 行业动向」的线索词。
# 命中一词即可——这类问题的答案通常会随时间变化，本地资料一定会过时。
_TIME_SENSITIVE_RE = re.compile(
    "|".join(
        [
            r"今天|今日|现在|当前|此刻",
            r"最新|最近|近期|近来",
            r"今年|本月|本周|这周|上个?月|去年",
            r"新闻|热搜|头条|报道|消息",
            r"天气|气温|温度|降雨|台风|暴雨|下雪",
            r"股价|汇率|油价|金价|房价|基金|行情",
            r"涨价|降价|调价|价格调整",
            r"活动|促销|打折|优惠|大促|周年庆|开业|闭店|关店|停业|撤店",
            r"政策|新规|规定|法规|标准|条例|公告|通知",
            r"发布|上线|上市|推出|宣布|召回",
        ]
    )
)


def is_time_sensitive(query: str) -> bool:
    """问题是否指向会随时间变化的外部信息。"""
    return bool(_TIME_SENSITIVE_RE.search(query or ""))


def should_search_online(
    query: str,
    *,
    mode: str = DEFAULT_MODE,
    has_local_hits: bool = False,
) -> bool:
    """按配置模式判断本轮是否需要联网检索。"""
    if mode == MODE_OFF:
        return False
    if not (query or "").strip():
        return False
    if mode == MODE_ALWAYS:
        return True
    # auto：本地有资料就别联网，本地答不上且问题指向外部信息时才联网
    if has_local_hits:
        return False
    return is_time_sensitive(query)


def normalize_mode(mode: str | None) -> str:
    """把配置值收敛到合法取值（不认识的值按 auto 处理）。"""
    value = (mode or "").strip().lower()
    return value if value in VALID_MODES else DEFAULT_MODE


__all__ = [
    "DEFAULT_MODE",
    "MODE_ALWAYS",
    "MODE_AUTO",
    "MODE_OFF",
    "VALID_MODES",
    "is_time_sensitive",
    "normalize_mode",
    "should_search_online",
]
