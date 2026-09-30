"""角色设定卡（Character sheet）YAML 读写器。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values


def _require(data: dict, key: str) -> Any:
    """获取必需字段。缺失时抛出 ValueError。"""
    if key not in data or data[key] is None:
        raise ValueError(f"Required field '{key}' is missing")
    return data[key]


def _parse_personality(data: dict) -> Personality:
    return Personality(**data)


def _parse_values(data: dict) -> Values:
    return Values(**data)


def _parse_initial_state(data: dict) -> EmotionalState:
    return EmotionalState(
        pleasure=data["pleasure"],
        arousal=data["arousal"],
        dominance=data["dominance"],
        emotion_label=str(data.get("emotion_label", "中立")),
        situation=str(data.get("situation", "")),
    )


def _parse_goals(data: dict, character_id: str) -> GoalTree:
    visions_data = data.get("visions", [])
    objectives_data = data.get("objectives", [])
    tasks_data = data.get("tasks", [])

    visions = []
    for v in visions_data:
        visions.append(Vision(
            id=str(uuid.uuid4()),
            character_id=character_id,
            content=v["content"],
        ))

    objectives = []
    for o in objectives_data:
        vi = o.get("vision_index", 0)
        vision_id = visions[vi].id if vi < len(visions) else ""
        objectives.append(Objective(
            id=str(uuid.uuid4()),
            character_id=character_id,
            vision_id=vision_id,
            content=o["content"],
            status=o.get("status", "active"),
            progress=float(o.get("progress", 0.0)),
        ))

    tasks = []
    for t in tasks_data:
        oi = t.get("objective_index", 0)
        objective_id = objectives[oi].id if oi < len(objectives) else ""
        tasks.append(Task(
            id=str(uuid.uuid4()),
            character_id=character_id,
            objective_id=objective_id,
            content=t["content"],
            status=t.get("status", "pending"),
        ))

    return GoalTree(visions=visions, objectives=objectives, tasks=tasks)


def _goals_to_dict(goal_tree: GoalTree) -> dict:
    """GoalTree → 用于 YAML 的 dict。"""
    # vision_id → index 的映射
    vision_ids = [v.id for v in goal_tree.visions]
    objective_ids = [o.id for o in goal_tree.objectives]

    result: dict[str, list] = {}

    if goal_tree.visions:
        result["visions"] = [{"content": v.content} for v in goal_tree.visions]

    if goal_tree.objectives:
        objs = []
        for o in goal_tree.objectives:
            vi = vision_ids.index(o.vision_id) if o.vision_id in vision_ids else 0
            objs.append({
                "content": o.content,
                "status": o.status,
                "progress": o.progress,
                "vision_index": vi,
            })
        result["objectives"] = objs

    if goal_tree.tasks:
        tsks = []
        for t in goal_tree.tasks:
            oi = objective_ids.index(t.objective_id) if t.objective_id in objective_ids else 0
            tsks.append({
                "content": t.content,
                "status": t.status,
                "objective_index": oi,
            })
        result["tasks"] = tsks

    return result


@dataclass
class CharacterSheet:
    """YAML ↔ 数据模型转换。

    character: Character（必需）
    initial_state: EmotionalState（可选）
    goal_tree: GoalTree（可选）
    voice_id: str | None（可选）
    """

    character: Character
    initial_state: EmotionalState | None = None
    goal_tree: GoalTree | None = None
    voice_id: str | None = None

    @classmethod
    def from_yaml(cls, yaml_str: str) -> CharacterSheet:
        """从 YAML 字符串解析。"""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Invalid YAML: expected a mapping")
        return cls._from_dict(data)

    @classmethod
    def load(cls, path: Path) -> CharacterSheet:
        """从文件读取。"""
        text = path.read_text(encoding="utf-8")
        return cls.from_yaml(text)

    @classmethod
    def load_directory(cls, directory: Path) -> list[CharacterSheet]:
        """批量读取目录内的 *.character.yaml。"""
        sheets = []
        for f in sorted(directory.glob("*.character.yaml")):
            sheets.append(cls.load(f))
        return sheets

    @classmethod
    def _from_dict(cls, data: dict) -> CharacterSheet:
        name = _require(data, "name")
        char_id = _require(data, "id")
        personality_data = _require(data, "personality")
        values_data = _require(data, "values")

        personality = _parse_personality(personality_data)
        values = _parse_values(values_data)

        character = Character(
            id=str(char_id),
            name=str(name),
            personality=personality,
            values=values,
            profile=data.get("profile"),
            appearance=data.get("appearance"),
            speaking_style=data.get("speaking_style"),
            background=data.get("background"),
            personality_description=data.get("personality_description"),
            values_description=data.get("values_description"),
            role_title=data.get("role_title"),
            job_description=data.get("job_description"),
            service_rules=data.get("service_rules"),
            emotional_expressiveness=float(data.get("emotional_expressiveness", 1.0)),
        )

        initial_state = None
        if "initial_state" in data and data["initial_state"]:
            initial_state = _parse_initial_state(data["initial_state"])

        goal_tree = None
        if "goals" in data and data["goals"]:
            goal_tree = _parse_goals(data["goals"], str(char_id))

        voice_id = data.get("voice_id")

        return cls(
            character=character,
            initial_state=initial_state,
            goal_tree=goal_tree,
            voice_id=voice_id,
        )

    def to_yaml(self) -> str:
        """序列化为 YAML 字符串。"""
        data = self._to_dict()
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def save(self, path: Path) -> None:
        """写入文件。"""
        path.write_text(self.to_yaml(), encoding="utf-8")

    def _to_dict(self) -> dict:
        c = self.character
        data: dict[str, Any] = {
            "name": c.name,
            "id": c.id,
        }

        # 可选的字符串字段
        for field in ("profile", "appearance", "speaking_style",
                      "background", "personality_description", "values_description",
                      "role_title", "job_description", "service_rules"):
            val = getattr(c, field)
            if val is not None:
                data[field] = val

        # 情绪外显系数（1.0 为默认值，不写回 YAML）
        if c.emotional_expressiveness != 1.0:
            data["emotional_expressiveness"] = c.emotional_expressiveness

        # 性格
        p = c.personality
        data["personality"] = {
            "openness": p.openness,
            "conscientiousness": p.conscientiousness,
            "extraversion": p.extraversion,
            "agreeableness": p.agreeableness,
            "neuroticism": p.neuroticism,
        }

        # 价值观
        v = c.values
        data["values"] = {
            "self_transcendence": v.self_transcendence,
            "self_enhancement": v.self_enhancement,
            "openness_to_change": v.openness_to_change,
            "conservation": v.conservation,
        }

        # 初始状态
        if self.initial_state:
            s = self.initial_state
            data["initial_state"] = {
                "pleasure": s.pleasure,
                "arousal": s.arousal,
                "dominance": s.dominance,
                "emotion_label": s.emotion_label,
                "situation": s.situation,
            }

        # 目标
        if self.goal_tree:
            data["goals"] = _goals_to_dict(self.goal_tree)

        # 语音
        data["voice_id"] = self.voice_id

        return data
