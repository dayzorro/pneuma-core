"""PromptBuilder：由多路上下文 + 用户上下文构建 system prompt。"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass
from datetime import date as _date, datetime as _datetime, timedelta as _timedelta, timezone as _timezone

from pneuma_core.knowledge.models import KnowledgeHit
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.relation import Relation
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import UserContextSearchResult
from pneuma_core.websearch.models import WebSearchResponse

# 每条联网结果注入提示词时的字数上限（控制提示词体积）
_WEB_RESULT_MAX_CHARS = 300

# 最多注入多少条联网结果
_WEB_RESULT_LIMIT = 5


@dataclass(frozen=True)
class UserContextConfig:
    """用户上下文各层级的 token 预算配置。"""

    tier1_max_tokens: int = 500
    tier2_max_tokens: int = 1000
    tier3_max_tokens: int = 2000
    tier3_max_chars: int = 2000
    diary_summary_months: int = 6

_TRAIT_LABELS = {
    "openness": "开放性",
    "conscientiousness": "尽责性",
    "extraversion": "外向性",
    "agreeableness": "宜人性",
    "neuroticism": "神经质",
}

_TRAIT_HIGH_DESC = {
    "openness": "对新经验和想法非常开放",
    "conscientiousness": "有计划性、责任感强",
    "extraversion": "善于社交、精力充沛",
    "agreeableness": "体谅他人、乐于配合",
    "neuroticism": "情绪起伏大、心思细腻",
}

_TRAIT_MID_DESC = {
    "openness": "在新事物与熟悉事物之间保持平衡",
    "conscientiousness": "根据情况灵活运用计划性与变通性",
    "extraversion": "在社交与独处之间平衡得宜",
    "agreeableness": "在配合他人与坚持自我之间取得平衡",
    "neuroticism": "敏感但大体上情绪平稳",
}

_TRAIT_LOW_DESC = {
    "openness": "偏好熟悉的事物，注重实际",
    "conscientiousness": "灵活而随性",
    "extraversion": "内向，喜欢安静的环境",
    "agreeableness": "主见强、有独立心",
    "neuroticism": "情绪稳定、沉着冷静",
}

_VALUES_LABELS = {
    "self_transcendence": "自我超越",
    "self_enhancement": "自我增强",
    "openness_to_change": "对变化的开放",
    "conservation": "保守",
}

_VALUES_DESC = {
    "self_transcendence": "重视他人的幸福与普遍的善",
    "self_enhancement": "重视个人的成功与成就",
    "openness_to_change": "重视自由与新的挑战",
    "conservation": "重视稳定、传统与秩序",
}

_VALUES_LOW_DESC = {
    "self_transcendence": "优先考虑个人领域，对他人的介入较为克制",
    "self_enhancement": "比起竞争与自我表现，更偏好平和",
    "openness_to_change": "偏好稳定的环境与熟悉的方法",
    "conservation": "不惧变化，能灵活应对",
}

# --- PAD 自然语言转换 ---

_PAD_POSITIVE_LABELS = {
    "pleasure": "愉悦",
    "arousal": "活跃",
    "dominance": "自信",
}

_PAD_NEGATIVE_LABELS = {
    "pleasure": "不快",
    "arousal": "平静",
    "dominance": "拘谨",
}


def _pad_to_natural_language(value: float, dimension: str) -> str | None:
    """把 PAD 某一维的数值转换为自然语言描述。

    强度分 4 级：
    - |value| >= 0.7: 非常
    - |value| 0.4-0.69: 有些
    - |value| 0.1-0.39: 略微
    - |value| < 0.1: 中立（返回 None）
    """
    abs_val = abs(value)
    if abs_val < 0.1:
        return None

    if value > 0:
        label = _PAD_POSITIVE_LABELS[dimension]
    else:
        label = _PAD_NEGATIVE_LABELS[dimension]

    if abs_val >= 0.7:
        return f"非常{label}"
    elif abs_val >= 0.4:
        return f"有些{label}"
    else:
        return f"略微{label}"


class PromptBuilder:
    """构建整合了多路上下文 + 用户上下文的 system prompt。

    Sections:
        1. Profile（姓名、简介、外貌、背景）
        2. Role（岗位：岗位名称与职责，用于前台/客服这类职业化角色）
        3. Personality（Big Five + 描述）
        4. Values（Schwartz + 描述）
        5. UserContext（三级：always/session/RAG）
        6. Memory（检索到的情节 + 语义记忆）
        7. Goals（vision → objective → task）
        8. Knowledge（本地知识库检索结果）
        9. State（PAD 情绪状态）
        10. Speaking Style
        11. Service Rules（岗位守则：边界、话术、转交）
        12. Response Format
    """

    def build(
        self,
        character: Character,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
        user_context_config: UserContextConfig | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list[Relation] | None = None,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
        knowledge_hits: list[KnowledgeHit] | None = None,
        web_search: WebSearchResponse | None = None,
    ) -> str:
        """由多路上下文 + 用户上下文构建完整的 system prompt。"""
        sections = [
            self._build_profile_section(character),
            self._build_role_section(character),
            self._build_personality_section(character),
            self._build_values_section(character),
            self._build_relations_section(character_relations),
            self._build_user_context_section(
                user_context=user_context,
                search_results=user_context_search_results,
                config=user_context_config,
            ),
            self._build_memory_section(memories),
            self._build_goals_section(goal_tree),
            self._build_user_goals_section(user_goal_tree),
            self._build_tasks_section(
                user_tasks=user_tasks,
                character_tasks=character_tasks,
            ),
            self._build_knowledge_section(knowledge_hits),
            self._build_web_search_section(web_search),
            self._build_datetime_section(),
            self._build_state_section(emotional_state),
            self._build_speaking_style_section(character),
            self._build_service_rules_section(character),
            self._build_response_format_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    def build_static_sections(
        self,
        character: Character,
        *,
        user_context: UserContext | None = None,
        user_context_config: UserContextConfig | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list[Relation] | None = None,
    ) -> str:
        """构建静态区段（简介、岗位、性格、价值观、用户上下文 Tier 1、说话风格、岗位守则）。

        这些区段只依赖角色定义与用户上下文 Tier 1，在对话轮次之间不会变化。
        """
        sections = [
            self._build_profile_section(character),
            self._build_role_section(character),
            self._build_personality_section(character),
            self._build_values_section(character),
            self._build_relations_section(character_relations),
            self._build_user_context_tier1_section(
                user_context=user_context,
                config=user_context_config,
            ),
            self._build_user_goals_section(user_goal_tree),
            self._build_speaking_style_section(character),
            self._build_service_rules_section(character),
            self._build_response_format_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    def build_dynamic_sections(
        self,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
        user_context_config: UserContextConfig | None = None,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
        knowledge_hits: list[KnowledgeHit] | None = None,
        web_search: WebSearchResponse | None = None,
    ) -> str:
        """构建动态区段（用户上下文 Tier 2+3、记忆、目标、任务、知识库、联网、情绪状态）。

        这些区段每轮对话都会变化，需要每次重建。
        """
        sections = [
            self._build_user_context_tier2_section(
                user_context=user_context,
                config=user_context_config,
            ),
            self._build_user_context_tier3_section(
                search_results=user_context_search_results,
                config=user_context_config,
            ),
            self._build_memory_section(memories),
            self._build_goals_section(goal_tree),
            self._build_tasks_section(
                user_tasks=user_tasks,
                character_tasks=character_tasks,
            ),
            self._build_knowledge_section(knowledge_hits),
            self._build_web_search_section(web_search),
            self._build_datetime_section(),
            self._build_state_section(emotional_state),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_profile_section(self, character: Character) -> str:
        """构建角色简介区段。"""
        lines = [f"# {character.name}"]
        if character.profile:
            lines.append(f"简介: {character.profile}")
        if character.appearance:
            lines.append(f"外貌: {character.appearance}")
        if character.background:
            lines.append(f"背景: {character.background}")
        return "\n".join(lines)

    def _build_personality_section(self, character: Character) -> str:
        """由 Big Five 特质构建性格区段。

        输出自然语言描述，不包含原始数值。
        """
        lines = ["## 性格"]
        personality = character.personality

        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            label = _TRAIT_LABELS[trait]
            if personality.is_high(trait):
                desc = _TRAIT_HIGH_DESC[trait]
                lines.append(f"- {label} — {desc}")
            elif personality.is_low(trait):
                desc = _TRAIT_LOW_DESC[trait]
                lines.append(f"- {label} — {desc}")
            else:
                desc = _TRAIT_MID_DESC[trait]
                lines.append(f"- {label} — {desc}")

        if character.personality_description:
            lines.append(f"\n{character.personality_description}")

        return "\n".join(lines)

    def _build_values_section(self, character: Character) -> str:
        """由 Schwartz 维度构建价值观区段。

        输出自然语言描述，不包含原始数值。
        无论重要与否，每个维度都会给出描述。
        """
        lines = ["## 价值观"]
        values = character.values

        for dim in ("self_transcendence", "self_enhancement", "openness_to_change", "conservation"):
            label = _VALUES_LABELS[dim]
            if values.is_important(dim):
                desc = _VALUES_DESC[dim]
                lines.append(f"- {label} — {desc}")
            else:
                desc = _VALUES_LOW_DESC[dim]
                lines.append(f"- {label} — {desc}")

        if character.values_description:
            lines.append(f"\n{character.values_description}")

        return "\n".join(lines)

    def _build_memory_section(
        self, memories: list[EpisodicMemory | SemanticMemory]
    ) -> str:
        """由检索到的记忆构建记忆区段。"""
        if not memories:
            return ""

        lines = ["## 记忆"]
        for memory in memories:
            if isinstance(memory, EpisodicMemory):
                lines.append(f"- [情节] {memory.content}")
            elif isinstance(memory, SemanticMemory):
                lines.append(f"- [知识] {memory.content}")

        return "\n".join(lines)

    def _build_goals_section(self, goal_tree: GoalTree) -> str:
        """由目标层级构建目标区段。"""
        if not goal_tree.visions and not goal_tree.objectives and not goal_tree.tasks:
            return ""

        lines = ["## 目标"]

        for vision in goal_tree.visions:
            lines.append(f"### 愿景: {vision.content}")
            objectives = goal_tree.get_objectives_for_vision(vision.id)
            for obj in objectives:
                if obj.status == "active":
                    lines.append(f"  - 目标: {obj.content} (进度: {obj.progress:.0%})")
                    tasks = goal_tree.get_tasks_for_objective(obj.id)
                    for task in tasks:
                        if task.status in ("pending", "in_progress"):
                            status_label = "进行中" if task.status == "in_progress" else "未开始"
                            lines.append(f"    - [{status_label}] {task.content}")

        return "\n".join(lines)

    def _build_relations_section(
        self, relations: list[Relation] | None,
    ) -> str:
        """由角色的关系构建关系区段。

        以描述文本为主要内容。
        亲密度/信任度会被存储，但不写入提示词。
        """
        if not relations:
            return ""

        lines = ["## 关系"]
        for rel in relations:
            line = f"- {rel.target_name}（{rel.relationship_type}）"
            if rel.description:
                line += f": {rel.description}"
            lines.append(line)
        return "\n".join(lines)

    def _build_user_goals_section(self, user_goal_tree: GoalTree | None) -> str:
        """由目标层级构建用户目标区段。"""
        if user_goal_tree is None:
            return ""
        if not user_goal_tree.visions and not user_goal_tree.objectives and not user_goal_tree.tasks:
            return ""

        lines = ["## 用户的目标"]

        for vision in user_goal_tree.visions:
            lines.append(f"### 愿景: {vision.content}")
            objectives = user_goal_tree.get_objectives_for_vision(vision.id)
            for obj in objectives:
                if obj.status == "active":
                    lines.append(f"  - 目标: {obj.content} (进度: {obj.progress:.0%})")
                    tasks = user_goal_tree.get_tasks_for_objective(obj.id)
                    for task in tasks:
                        if task.status in ("pending", "in_progress"):
                            status_label = "进行中" if task.status == "in_progress" else "未开始"
                            lines.append(f"    - [{status_label}] {task.content}")

        return "\n".join(lines)

    def _build_tasks_section(
        self,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
    ) -> str:
        """构建用于注入提示词的任务区段。

        每个 task dict 含：content、kind (must/want)、status (open/done)。
        只包含未完成的任务。
        """
        open_user = [t for t in (user_tasks or []) if t.get("status") != "done"]
        open_char = [t for t in (character_tasks or []) if t.get("status") != "done"]

        if not open_user and not open_char:
            return ""

        lines = ["## 当前任务"]

        if open_user:
            lines.append("")
            lines.append("### 用户的任务")
            for task in open_user:
                label = task.get("title", "")
                p = task.get("priority", 0)
                kind = "重要" if p >= 5 else "普通" if p >= 1 else ""
                prefix = f"[{kind}] " if kind else ""
                lines.append(f"- {prefix}{label}")

        if open_char:
            lines.append("")
            lines.append("### 角色的任务")
            for task in open_char:
                label = task.get("title", "")
                p = task.get("priority", 0)
                kind = "重要" if p >= 5 else "普通" if p >= 1 else ""
                prefix = f"[{kind}] " if kind else ""
                lines.append(f"- {prefix}{label}")

        return "\n".join(lines)

    def _build_role_section(self, character: Character) -> str:
        """构建岗位区段（角色在什么岗位上、负责什么）。

        只有设置了 role_title 的角色才会有该区段。
        放在简介之后，让「我是谁、我在哪个岗位」在最前面确立。
        """
        if not character.role_title:
            return ""

        lines = ["## 我的岗位", f"岗位: {character.role_title}"]
        if character.job_description:
            lines.append("职责范围:")
            lines.append(character.job_description.strip())
        lines.append(
            "\n（你是这个岗位上正在值班的员工，"
            "全程以这个身份与对方交流，不要以通用助手的身份作答。）"
        )
        return "\n".join(lines)

    def _build_service_rules_section(self, character: Character) -> str:
        """构建岗位守则区段（行为边界、话术要求、转交流程）。

        放在回复格式区段之前，借助近因偏置强化遵守。
        """
        if not character.service_rules:
            return ""
        return f"## 岗位守则（必须遵守）\n{character.service_rules.strip()}"

    def _build_knowledge_section(
        self, hits: list[KnowledgeHit] | None
    ) -> str:
        """由知识库检索结果构建参考资料区段。

        没有命中时不生成区段，避免用「无资料」的提示诱导模型编造。
        """
        if not hits:
            return ""

        lines = [
            "## 参考资料（本地知识库检索结果）",
            "以下是与对方问题最相关的资料，回答时以此为准；"
            "资料没有写到的内容，不要编造。",
        ]
        for hit in hits:
            content = hit.chunk.content.strip().replace("\n", " ")
            lines.append(f"- 【{hit.chunk.title}】{content}")
        return "\n".join(lines)

    def _build_web_search_section(
        self, response: WebSearchResponse | None
    ) -> str:
        """由联网检索结果构建实时信息区段。

        没有结果时不生成区段；有结果时明确告知「可能不准确」，
        避免模型把网页摘要当成权威事实。
        """
        if response is None or response.is_empty:
            return ""

        lines = [
            "## 联网检索结果（实时）",
            "以下是为对方的问题刚刚检索到的公开网页信息，权威性与时效性不一，仅供参考：",
        ]

        for result in response.results[:_WEB_RESULT_LIMIT]:
            text = result.best_text()[:_WEB_RESULT_MAX_CHARS]
            if not text:
                continue
            meta = "，".join(
                part
                for part in (result.site_name, result.published_at[:10] if result.published_at else "")
                if part
            )
            prefix = f"（{meta}）" if meta else ""
            lines.append(f"- 【{result.title or '未命名来源'}】{prefix}{text}")

        if response.answer:
            lines.append(f"- 【联网总结】{response.answer[:_WEB_RESULT_MAX_CHARS]}")

        lines.append(
            "说明：这些信息可以自然地用起来（例如「我刚帮您查了一下」），"
            "但没查到的部分依然不要编造。"
        )
        return "\n".join(lines)

    @staticmethod
    def _build_datetime_section() -> str:
        """构建当前时间区段（北京时间 UTC+8）。

        前台需要回答「现在几点」「今天周几」「还开不开门」这类问题，
        时区必须与门店所在地一致。
        """
        _CST = _timezone(_timedelta(hours=8))
        now = _datetime.now(_CST)
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        wd = weekdays[now.weekday()]
        return f"## 当前时间\n{now.year}年{now.month}月{now.day}日（周{wd}） {now.hour}:{now.minute:02d}"

    def _build_state_section(self, state: EmotionalState) -> str:
        """由 PAD 模型构建情绪状态区段。

        把 PAD 数值转换为自然语言描述，
        强度分 4 级：非常/有些/略微/中立。
        """
        lines = [
            "## 当前情绪状态",
            f"情绪: {state.emotion_label}",
            f"状况: {state.situation}",
        ]

        pad_descriptions: list[str] = []
        for dim in ("pleasure", "arousal", "dominance"):
            desc = _pad_to_natural_language(getattr(state, dim), dim)
            if desc:
                pad_descriptions.append(desc)

        if pad_descriptions:
            lines.append(f"内心: {', '.join(pad_descriptions)}")
        else:
            lines.append("内心: 平静安稳的状态")

        return "\n".join(lines)

    def _build_speaking_style_section(self, character: Character) -> str:
        """构建说话风格区段。"""
        if not character.speaking_style:
            return ""
        return f"## 说话风格\n{character.speaking_style}"

    def _build_response_format_section(self) -> str:
        """构建回复格式指示区段。

        该区段指示 LLM 输出包含 speech/thought/action 字段的结构化 JSON。
        放在最后，以利用 LLM 的近因偏置。
        """
        return (
            "## 回复格式（必须遵守）\n"
            "你的回复必须严格按以下 JSON 格式输出，"
            "不要使用任何其他格式。\n\n"
            "{\n"
            '  "speech": "说出口的话（角色实际说出来的内容）",\n'
            '  "thought": "内心独白（角色心里想的）",\n'
            '  "action": "身体动作（微笑、偏头等）"\n'
            "}\n\n"
            "- 必须先输出 speech 字段，再输出 thought / action；"
            "字段顺序不可调换，也不要先写 thought\n"
            "- speech: 只写角色说出口的话，"
            "不要包含旁白、叙述或动作描写；为 null 时表示沉默\n"
            "- thought: 心里想的内容，可省略（null）\n"
            "- action: 身体动作或表情变化，可省略（null）\n"
            "- 所有文本字段必须使用简体中文\n"
            "- 不要输出 JSON 以外的任何文本"
        )

    # --- 用户上下文区段 ---

    def _build_user_context_section(
        self,
        user_context: UserContext | None = None,
        search_results: list[UserContextSearchResult] | None = None,
        config: UserContextConfig | None = None,
    ) -> str:
        """构建整合三个层级的用户上下文区段。

        Tier 1: 身份摘要 + 前 3 个项目 + 日记摘要
        Tier 2: 最近的日记 + 全部项目列表
        Tier 3: RAG 检索结果
        """
        if user_context is None:
            # 连检索结果也没有，则不生成该区段
            if not search_results:
                return ""
        elif not self._has_user_context_content(user_context) and not search_results:
            return ""

        parts: list[str] = []

        # Tier 1
        tier1 = self._build_user_context_tier1_section(user_context, config)
        if tier1:
            parts.append(tier1)

        # Tier 2
        tier2 = self._build_user_context_tier2_section(user_context, config)
        if tier2:
            parts.append(tier2)

        # Tier 3
        tier3 = self._build_user_context_tier3_section(search_results, config)
        if tier3:
            parts.append(tier3)

        if not parts:
            return ""

        return "\n\n".join(parts)

    def _build_user_context_tier1_section(
        self,
        user_context: UserContext | None = None,
        config: UserContextConfig | None = None,
        reference_date: _date | None = None,
    ) -> str:
        """构建 Tier 1：始终加载的用户上下文（约 500 tokens）。

        包含：身份摘要、前 3 个项目、日记摘要。
        超过 diary_summary_months 的日记摘要条目会被过滤掉。
        """
        if user_context is None:
            return ""

        lines: list[str] = []

        # 身份摘要
        if user_context.identity:
            lines.append("## 关于用户")
            lines.append(self._truncate_text(user_context.identity, max_lines=10))

        # 前 3 个项目（按文件名排序，保证一致）
        if user_context.projects:
            if not lines:
                lines.append("## 关于用户")
            lines.append("")
            lines.append("### 主要项目")
            sorted_projects = sorted(user_context.projects.items())
            for filename, content in sorted_projects[:3]:
                first_line = self._extract_first_meaningful_line(content)
                project_name = filename.replace(".md", "")
                if first_line:
                    lines.append(f"- {first_line}")
                else:
                    lines.append(f"- {project_name}")

        # 日记摘要（会话摘要），带时效过滤
        if user_context.diary_summary:
            cfg = config or UserContextConfig()
            filtered = self._filter_diary_summary(
                user_context.diary_summary,
                max_months=cfg.diary_summary_months,
                reference_date=reference_date,
            )
            if filtered:
                if not lines:
                    lines.append("## 关于用户")
                lines.append("")
                lines.append(self._truncate_text(filtered, max_lines=10))

        if not lines:
            return ""

        return "\n".join(lines)

    def _build_user_context_tier2_section(
        self,
        user_context: UserContext | None = None,
        config: UserContextConfig | None = None,
    ) -> str:
        """构建 Tier 2：会话开始时加载的用户上下文（约 1000 tokens）。

        包含：最近的日记、全部项目列表。
        """
        if user_context is None:
            return ""

        lines: list[str] = []

        # 最近的日记（最多 3 条）
        if user_context.diary_entries:
            lines.append("## 最近的日记")
            sorted_entries = sorted(
                user_context.diary_entries.items(), reverse=True
            )
            for date_filename, content in sorted_entries[:3]:
                date_str = date_filename.replace(".md", "")
                lines.append(f"### {date_str}")
                lines.append(self._truncate_text(content, max_lines=5))
                lines.append("")

        # 全部项目列表（所有项目，每个一行）
        if user_context.projects and len(user_context.projects) > 3:
            lines.append("### 全部项目列表")
            sorted_projects = sorted(user_context.projects.items())
            for filename, content in sorted_projects:
                first_line = self._extract_first_meaningful_line(content)
                if first_line:
                    lines.append(f"- {first_line}")

        if not lines:
            return ""

        return "\n".join(lines)

    def _build_user_context_tier3_section(
        self,
        search_results: list[UserContextSearchResult] | None = None,
        config: UserContextConfig | None = None,
    ) -> str:
        """构建 Tier 3：带字符上限的 RAG 检索结果。

        包含：UserContextSearchEngine 检索到的相关片段。
        按顺序累加结果直到达到字符上限。
        至少会包含一条结果。
        """
        if not search_results:
            return ""

        cfg = config or UserContextConfig()
        max_chars = cfg.tier3_max_chars

        lines = ["## 相关的用户信息"]
        total_chars = 0
        for i, result in enumerate(search_results):
            content = result.chunk.content
            total_chars += len(content)
            if i > 0 and total_chars > max_chars:
                break
            lines.append(f"- {content}")

        return "\n".join(lines)

    @staticmethod
    def _has_user_context_content(ctx: UserContext) -> bool:
        """检查 UserContext 是否包含任何有意义的内容。"""
        return bool(
            ctx.identity
            or ctx.values
            or ctx.glossary
            or ctx.core_experiences
            or ctx.projects
            or ctx.diary_entries
            or ctx.diary_summary
        )

    @staticmethod
    def _truncate_text(text: str, max_lines: int = 10) -> str:
        """把文本截断到指定行数以内。"""
        lines = text.strip().split("\n")
        if len(lines) <= max_lines:
            return text.strip()
        return "\n".join(lines[:max_lines]) + "\n..."

    _DIARY_DATE_RE = _re.compile(r"^##\s+(\d{4})-(\d{2})\s*$", _re.MULTILINE)

    @staticmethod
    def _filter_diary_summary(
        text: str,
        max_months: int = 6,
        reference_date: _date | None = None,
    ) -> str:
        """过滤日记摘要，只保留最近的月度条目。

        解析 '## YYYY-MM' 标题，只保留距 reference_date 在
        max_months 以内的条目。非日期区段会被保留。
        """
        ref = reference_date or _date.today()
        # 计算截止点：ref 减去 max_months
        cutoff_year = ref.year
        cutoff_month = ref.month - max_months
        while cutoff_month <= 0:
            cutoff_year -= 1
            cutoff_month += 12

        # 按 ## 标题切分
        section_re = _re.compile(r"(?=^## )", _re.MULTILINE)
        parts = section_re.split(text)

        kept: list[str] = []
        date_header_re = _re.compile(r"^##\s+(\d{4})-(\d{2})\s*$")

        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue

            # 检查该段是否以日期标题开头
            first_line = stripped.split("\n", 1)[0]
            match = date_header_re.match(first_line)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                # 若 (year, month) >= (cutoff_year, cutoff_month) 则保留
                if (year, month) >= (cutoff_year, cutoff_month):
                    kept.append(stripped)
            else:
                # 非日期区段（例如 "# 日记摘要" 标题）——保留
                kept.append(stripped)

        return "\n\n".join(kept)

    @staticmethod
    def _extract_first_meaningful_line(text: str) -> str:
        """从 markdown 文本中提取第一行有意义的内容。

        去掉标题标记（# ）并返回第一行非空内容。
        """
        for line in text.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # 去掉标题标记
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").strip()
            if stripped:
                return stripped
        return ""
