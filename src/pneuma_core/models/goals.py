"""Goal hierarchy models: Vision, Objective, Task, GoalTree."""

from dataclasses import dataclass, field
from typing import Literal

ObjectiveStatus = Literal["active", "achieved", "abandoned"]
TaskStatus = Literal["pending", "in_progress", "completed", "abandoned"]

_OBJECTIVE_STATUSES = ("active", "achieved", "abandoned")
_TASK_STATUSES = ("pending", "in_progress", "completed", "abandoned")


@dataclass(frozen=True)
class Vision:
    """长期愿景（5-10 年）。"""

    id: str
    character_id: str
    content: str


@dataclass(frozen=True)
class Objective:
    """中期目標（四半期〜年単位）.

    status: active / achieved / abandoned
    progress: 0.0〜1.0
    """

    id: str
    character_id: str
    vision_id: str
    content: str
    status: ObjectiveStatus
    progress: float

    def __post_init__(self) -> None:
        if self.status not in _OBJECTIVE_STATUSES:
            raise ValueError(
                f"status must be one of {_OBJECTIVE_STATUSES}, got '{self.status}'"
            )
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"progress must be between 0.0 and 1.0, got {self.progress}")


@dataclass(frozen=True)
class Task:
    """具体任务（以日〜周为单位）。

    status: pending / in_progress / completed / abandoned
    """

    id: str
    character_id: str
    objective_id: str
    content: str
    status: TaskStatus

    def __post_init__(self) -> None:
        if self.status not in _TASK_STATUSES:
            raise ValueError(
                f"status must be one of {_TASK_STATUSES}, got '{self.status}'"
            )


@dataclass
class GoalTree:
    """三层目标结构。

    提供 Vision → Objective → Task 的层级引用。
    """

    visions: list[Vision] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def get_objectives_for_vision(self, vision_id: str) -> list[Objective]:
        """获取与 Vision 关联的 Objectives。"""
        return [o for o in self.objectives if o.vision_id == vision_id]

    def get_tasks_for_objective(self, objective_id: str) -> list[Task]:
        """获取与 Objective 关联的 Tasks。"""
        return [t for t in self.tasks if t.objective_id == objective_id]
