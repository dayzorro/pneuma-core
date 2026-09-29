"""Task 数据模型 (#132)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Task:
    """任务。"""

    id: str
    project_id: str
    title: str
    content: str = ""
    status: int = 0  # 0=Normal, 2=Completed（遵循 TickTick 规范）
    priority: int = 0  # 0=None, 1=Low, 3=Medium, 5=High
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)
    repeat: str | None = None
    created_date: datetime | None = None
    completed_date: datetime | None = None


@dataclass
class TaskCreate:
    """创建任务请求。"""

    title: str
    content: str = ""
    priority: int = 0
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskUpdate:
    """更新任务请求（为 None 的字段不更新）。"""

    title: str | None = None
    content: str | None = None
    priority: int | None = None
    due_date: date | None = None
    tags: list[str] | None = None
