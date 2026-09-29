"""TodoItem：TODO + 习惯管理的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class TodoItem:
    """TODO 条目。"""

    id: str                          # UUID
    content: str                     # 任务内容
    label: str                       # habit | deadline | this_week | someday
    kind: str                        # must | want
    status: str                      # pending | done | skipped
    priority: int                    # 1(高) 〜 3(低)
    due_date: date | None            # 截止日期
    recurrence: str | None           # daily | weekly | weekdays | None
    created_at: datetime
    completed_at: datetime | None = None
    owner_id: str = "user"           # 用户或角色 ID
