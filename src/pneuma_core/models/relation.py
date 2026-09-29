"""Relation：实体间关系模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Relation:
    """实体间的关系。

    表示用户-角色、角色-角色之间的关系。
    owner_id 是关系中的「主体」，target_id 是「对方」。
    """

    id: str
    owner_id: str              # 关系的主体 (user, mira, etc.)
    target_id: str             # 关系的对方
    target_name: str           # 对方的显示名
    relationship_type: str     # partner, friend, family, mentor, etc.
    description: str           # 关系的说明
    closeness: float           # 亲密度 0.0-1.0
    trust: float               # 信任度 0.0-1.0
    updated_at: datetime
    notes: str | None = None   # 附加备注
