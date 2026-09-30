"""把联网检索结果异步提炼成「行业通识认知块」。

这是认知库的写入端：一次联网检索之后，在后台用大模型把检索材料提炼成
与具体公司无关的行业常识，落进认知库（``InsightStore``）供后续检索复用。

放在后台执行，不占用对话的关键路径——用户不需要等提炼完成就能拿到回复。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone

from pneuma_core.knowledge.insights import Insight, InsightStore
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.websearch.models import WebSearchResponse

logger = logging.getLogger(__name__)

DEFAULT_MAX_INSIGHTS = 5
DEFAULT_MAX_MATERIAL_CHARS = 6000

_MD_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """\
你是零售门店行业的资深培训师，正在为「商超门店前台/顾客服务」岗位整理知识底座。

下面会给出一批从互联网检索到的公开信息。请从中提炼该岗位应当知道的
**行业通识认知**——与任何一家具体公司无关、可以独立成立的常识。

## 要提炼什么
- 行业惯例与常识：零售门店的服务流程、岗位分工、常见业务规则
- 消费者权益与法规常识：退换货、发票、会员、预付卡等领域的通行规则
- 服务经验与技巧：处理投诉、安抚情绪、引导顾客的通行做法
- 行业动向与趋势：零售业态变化、消费趋势、节假日消费特点

## 必须剔除
- 与某一家具体公司绑定的内容：公司经营细节、门店信息、企业专属政策条款、品牌宣传
- 个人隐私、客户数据、内部保密信息、未公开的经营数据
- 无法验证的传言、广告、营销话术、主观评价、情绪化表达
- 与门店前台岗位无关的内容

## 输出要求
- 每条独立成句，脱离原文也能看懂；长度 40〜80 字
- 简体中文，客观陈述，不用「据悉」「小编」这类新闻腔
- 最多 {max_insights} 条；材料里没有值得沉淀的内容时，返回空列表，不要凑数
- 严格按以下 JSON 输出，不要输出任何其它文本：

{{"insights": [{{"topic": "主题词，4〜8 字", "content": "认知内容"}}]}}"""


class InsightAcquirer:
    """联网结果 → 行业通识认知块 → 认知库。"""

    def __init__(
        self,
        llm: LLMAdapter,
        store: InsightStore,
        *,
        model: str | None = None,
        max_insights: int = DEFAULT_MAX_INSIGHTS,
        max_material_chars: int = DEFAULT_MAX_MATERIAL_CHARS,
    ) -> None:
        self._llm = llm
        self._store = store
        self._model = model
        self._max_insights = max(1, max_insights)
        self._max_material_chars = max_material_chars

    async def acquire(self, query: str, response: WebSearchResponse) -> int:
        """提炼并写入。返回新增的认知块条数（失败时返回 0，不抛出）。"""
        material = self._build_material(response)
        if not material:
            return 0

        try:
            insights = await self._extract(query, material, response)
        except Exception as e:  # noqa: BLE001 - 后台任务不应影响对话
            logger.warning(
                "Insight extraction failed (%s: %s)", type(e).__name__, e
            )
            return 0

        if not insights:
            return 0

        try:
            added = await self._store.add_many(insights)
        except Exception as e:  # noqa: BLE001 - 写入失败不影响对话
            logger.warning("Insight store write failed (%s: %s)", type(e).__name__, e)
            return 0

        if added:
            logger.info(
                "Insights acquired: +%d (total %d) from query %r",
                added,
                self._store.size,
                query,
            )
        return added

    @staticmethod
    def _build_material(response: WebSearchResponse) -> str:
        """把检索结果拼成给大模型的材料。"""
        blocks: list[str] = []
        for result in response.results:
            text = result.best_text()
            if not text:
                continue
            header = result.title or result.site_name or "未命名来源"
            if result.published_at:
                header = f"{header}（{result.published_at[:10]}）"
            blocks.append(f"【{header}】\n{text}")
        return "\n\n".join(blocks).strip()

    async def _extract(
        self, query: str, material: str, response: WebSearchResponse
    ) -> list[Insight]:
        if len(material) > self._max_material_chars:
            material = material[: self._max_material_chars]

        prompt = _SYSTEM_PROMPT.format(max_insights=self._max_insights)
        request = LLMRequest(
            system_prompt=prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"用户当时的问题是：{query}\n\n"
                        f"检索到的公开信息如下：\n\n{material}"
                    ),
                }
            ],
            model=self._model,
            temperature=0.2,
            max_tokens=1024,
            # 后台任务不影响交互延迟，保留思考以提升提炼质量
            enable_thinking=True,
        )
        llm_response = await self._llm.generate(request)
        parsed = _parse_insights(llm_response.content)

        urls = tuple(r.url for r in response.results if r.url)[: self._max_insights]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        return [
            Insight(
                id=uuid.uuid4().hex[:12],
                content=item["content"],
                topic=item.get("topic", ""),
                source_urls=urls,
                created_at=now,
            )
            for item in parsed[: self._max_insights]
        ]


def _parse_insights(raw: str) -> list[dict[str, str]]:
    """解析大模型返回的 JSON。任何异常都返回空列表。"""
    text = (raw or "").strip()
    if not text:
        return []

    match = _MD_CODE_BLOCK_RE.match(text)
    if match:
        text = match.group(1).strip()
    else:
        object_match = _JSON_OBJECT_RE.search(text)
        if object_match:
            text = object_match.group(0)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Insight extraction returned malformed JSON: %r", text[:200])
        return []

    if not isinstance(data, dict):
        return []

    items = data.get("insights")
    if not isinstance(items, list):
        return []

    results: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            content = item.strip()
            topic = ""
        elif isinstance(item, dict):
            content = str(item.get("content") or "").strip()
            topic = str(item.get("topic") or "").strip()
        else:
            continue
        if content:
            results.append({"content": content, "topic": topic})
    return results


__all__ = ["DEFAULT_MAX_INSIGHTS", "DEFAULT_MAX_MATERIAL_CHARS", "InsightAcquirer"]
