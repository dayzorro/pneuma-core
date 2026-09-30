"""Tests for 前台人设：岗位字段、岗位守则与知识库区段。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.knowledge.models import KnowledgeChunk, KnowledgeHit
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.prompt_builder import PromptBuilder
from pneuma_core.websearch.models import WebSearchResponse, WebSearchResult

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FRONTDESK_YAML = EXAMPLES / "xiaorun-frontdesk.character.yaml"

EMOTION = EmotionalState(
    pleasure=0.45, arousal=0.15, dominance=0.25,
    emotion_label="亲切", situation="在服务台值班",
)


def _make_personality() -> Personality:
    return Personality(
        openness=0.55,
        conscientiousness=0.85,
        extraversion=0.72,
        agreeableness=0.9,
        neuroticism=0.15,
    )


def _make_values() -> Values:
    return Values(
        self_transcendence=0.85,
        self_enhancement=0.35,
        openness_to_change=0.45,
        conservation=0.75,
    )


def _make_character(**kwargs) -> Character:
    defaults = dict(
        id="frontdesk-001",
        name="小润",
        personality=_make_personality(),
        values=_make_values(),
        profile="站在服务台后面的前台接待。",
        speaking_style="礼貌、热情、口语化。",
    )
    defaults.update(kwargs)
    return Character(**defaults)


def _hit(title: str, content: str) -> KnowledgeHit:
    return KnowledgeHit(
        chunk=KnowledgeChunk(
            id="x#0", doc_id="x", title=title, content=content
        ),
        score=0.8,
    )


class TestFrontdeskCharacterSheet:
    def test_loads_role_fields(self) -> None:
        sheet = CharacterSheet.load(FRONTDESK_YAML)
        character = sheet.character

        assert character.name == "小润"
        assert character.role_title == "华润万家 · 顾客服务前台"
        assert character.job_description
        assert "服务台" in character.job_description
        assert character.service_rules
        assert "岗位" in character.service_rules

    def test_loads_emotional_expressiveness(self) -> None:
        sheet = CharacterSheet.load(FRONTDESK_YAML)

        assert sheet.character.emotional_expressiveness == pytest.approx(0.45)

    def test_yaml_roundtrip_preserves_role_fields(self) -> None:
        sheet = CharacterSheet.load(FRONTDESK_YAML)

        restored = CharacterSheet.from_yaml(sheet.to_yaml()).character

        assert restored.role_title == sheet.character.role_title
        assert restored.job_description == sheet.character.job_description
        assert restored.service_rules == sheet.character.service_rules
        assert restored.emotional_expressiveness == pytest.approx(
            sheet.character.emotional_expressiveness
        )

    def test_defaults_to_full_expressiveness(self) -> None:
        yaml_text = (
            "name: 测试\nid: t\npersonality: {openness: 0.5, "
            "conscientiousness: 0.5, extraversion: 0.5, agreeableness: 0.5, "
            "neuroticism: 0.5}\nvalues: {self_transcendence: 0.5, "
            "self_enhancement: 0.5, openness_to_change: 0.5, conservation: 0.5}\n"
        )

        sheet = CharacterSheet.from_yaml(yaml_text)

        assert sheet.character.emotional_expressiveness == 1.0
        assert sheet.character.role_title is None
        assert "emotional_expressiveness" not in sheet.to_yaml()

    def test_rejects_out_of_range_expressiveness(self) -> None:
        with pytest.raises(ValueError, match="emotional_expressiveness"):
            _make_character(emotional_expressiveness=1.5)


class TestRoleSections:
    def test_prompt_contains_role_and_rules(self) -> None:
        character = _make_character(
            role_title="华润万家 · 顾客服务前台",
            job_description="负责到店顾客的接待与业务办理。",
            service_rules="先问候，再解决问题；不确定就转交值班经理。",
        )

        prompt = PromptBuilder().build(character, EMOTION, GoalTree(), [])

        assert "## 我的岗位" in prompt
        assert "岗位: 华润万家 · 顾客服务前台" in prompt
        assert "负责到店顾客的接待与业务办理。" in prompt
        assert "## 岗位守则（必须遵守）" in prompt
        assert "先问候，再解决问题；不确定就转交值班经理。" in prompt
        assert "不要以通用助手的身份作答" in prompt

    def test_prompt_omits_role_sections_when_unset(self) -> None:
        prompt = PromptBuilder().build(_make_character(), EMOTION, GoalTree(), [])

        assert "## 我的岗位" not in prompt
        assert "## 岗位守则（必须遵守）" not in prompt

    def test_static_sections_include_role_and_rules(self) -> None:
        character = _make_character(
            role_title="顾客服务前台", service_rules="保持专业。"
        )

        static = PromptBuilder().build_static_sections(character)

        assert "## 我的岗位" in static
        assert "## 岗位守则（必须遵守）" in static

    def test_rules_come_before_response_format(self) -> None:
        character = _make_character(
            role_title="顾客服务前台", service_rules="保持专业。"
        )

        prompt = PromptBuilder().build(character, EMOTION, GoalTree(), [])

        assert prompt.index("## 岗位守则（必须遵守）") < prompt.index(
            "## 回复格式（必须遵守）"
        )

    def test_role_section_precedes_personality(self) -> None:
        character = _make_character(role_title="顾客服务前台")

        prompt = PromptBuilder().build(character, EMOTION, GoalTree(), [])

        assert prompt.index("## 我的岗位") < prompt.index("## 性格")


class TestKnowledgeSection:
    def test_renders_hits(self) -> None:
        hits = [
            _hit("常见问题 · 你们几点开门？", "本店营业时间为 08:00 到 22:30。"),
            _hit("会员积分 · 积分怎么算", "消费一元积一分。"),
        ]

        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [], knowledge_hits=hits
        )

        assert "## 参考资料（本地知识库检索结果）" in prompt
        assert "【常见问题 · 你们几点开门？】" in prompt
        assert "本店营业时间为 08:00 到 22:30。" in prompt
        assert "【会员积分 · 积分怎么算】" in prompt
        assert "不要编造" in prompt

    def test_omitted_when_no_hits(self) -> None:
        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [], knowledge_hits=[]
        )

        assert "## 参考资料" not in prompt

    def test_multiline_content_is_flattened(self) -> None:
        hits = [_hit("标题", "第一行\n第二行")]

        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [], knowledge_hits=hits
        )

        assert "【标题】第一行 第二行" in prompt

    def test_dynamic_sections_include_knowledge(self) -> None:
        dynamic = PromptBuilder().build_dynamic_sections(
            EMOTION, GoalTree(), [], knowledge_hits=[_hit("标题", "内容")]
        )

        assert "## 参考资料（本地知识库检索结果）" in dynamic
        assert "## 当前时间" in dynamic


class TestDateTime:
    def test_uses_china_timezone(self) -> None:
        import re
        from datetime import datetime, timedelta, timezone

        prompt = PromptBuilder().build(_make_character(), EMOTION, GoalTree(), [])
        match = re.search(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日（周[一二三四五六日]）\s*(\d{1,2}):(\d{2})",
            prompt,
        )
        assert match, "未找到当前时间区段"

        cst = timezone(timedelta(hours=8))
        parsed = datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            tzinfo=cst,
        )

        assert abs((datetime.now(cst) - parsed).total_seconds()) < 120


def _web_response(**kwargs) -> WebSearchResponse:
    defaults = dict(
        query="最近零售行业有什么新动向",
        results=[
            WebSearchResult(
                title="零售周报",
                url="https://example.com/a",
                site_name="示例网",
                summary="多家商超调整门店业态，社区店占比提升。",
                published_at="2026-09-28T00:00:00+08:00",
            )
        ],
    )
    defaults.update(kwargs)
    return WebSearchResponse(**defaults)


class TestWebSearchSection:
    def test_renders_live_results(self) -> None:
        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [],
            web_search=_web_response(),
        )

        assert "## 联网检索结果（实时）" in prompt
        assert "【零售周报】" in prompt
        assert "示例网" in prompt
        assert "2026-09-28" in prompt
        assert "社区店占比提升" in prompt

    def test_omitted_without_response(self) -> None:
        prompt = PromptBuilder().build(_make_character(), EMOTION, GoalTree(), [])

        assert "## 联网检索结果" not in prompt

    def test_omitted_for_empty_response(self) -> None:
        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [],
            web_search=WebSearchResponse(query="q"),
        )

        assert "## 联网检索结果" not in prompt

    def test_warns_about_accuracy(self) -> None:
        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [],
            web_search=_web_response(),
        )

        assert "不要编造" in prompt

    def test_includes_web_answer_when_present(self) -> None:
        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [],
            web_search=_web_response(answer="联网总结的结论"),
        )

        assert "【联网总结】联网总结的结论" in prompt

    def test_limits_result_count(self) -> None:
        results = [
            WebSearchResult(title=f"标题{i}", url=f"https://e/{i}", summary="摘要")
            for i in range(12)
        ]

        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [],
            web_search=_web_response(results=results),
        )

        assert "标题4" in prompt
        assert "标题5" not in prompt

    def test_truncates_long_summary(self) -> None:
        prompt = PromptBuilder().build(
            _make_character(), EMOTION, GoalTree(), [],
            web_search=_web_response(
                results=[
                    WebSearchResult(title="长文", url="u", summary="占" * 900)
                ]
            ),
        )

        assert "占" * 301 not in prompt

    def test_dynamic_sections_include_web_results(self) -> None:
        dynamic = PromptBuilder().build_dynamic_sections(
            EMOTION, GoalTree(), [], web_search=_web_response()
        )

        assert "## 联网检索结果（实时）" in dynamic
