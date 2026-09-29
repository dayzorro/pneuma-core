"""TaskBackend Protocol (#132)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pneuma_core.task.models import Task, TaskCreate, TaskUpdate


@runtime_checkable
class TaskBackend(Protocol):
    """任务管理后端的抽象接口。"""

    async def list_tasks(
        self,
        *,
        status: int | None = None,
        tags: list[str] | None = None,
    ) -> list[Task]: ...

    async def get_task(self, task_id: str) -> Task | None: ...

    async def create_task(self, data: TaskCreate) -> Task: ...

    async def update_task(self, task_id: str, data: TaskUpdate) -> Task: ...

    async def complete_task(self, task_id: str) -> None: ...

    async def delete_task(self, task_id: str) -> None: ...
