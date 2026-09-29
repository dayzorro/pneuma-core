"""Vault 路径解析工具。

从 PNEUMA_VAULT_PATH 环境变量解析子路径的工具。
在 vault/ 目录下统一管理角色定义、用户上下文与运行时数据。
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_VAULT_PATH = "./vault"
_ENV_VAR = "PNEUMA_VAULT_PATH"


def get_vault_path() -> Path:
    """返回 Vault 的根路径。

    若设置了 PNEUMA_VAULT_PATH 环境变量则使用其值，
    未设置时回退到默认值 ``./vault``。
    """
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_VAULT_PATH))


def get_characters_dir() -> Path:
    """返回角色定义目录 ``vault/characters/`` 的路径。"""
    return get_vault_path() / "characters"


def get_user_context_dir() -> Path:
    """返回用户上下文目录 ``vault/user/`` 的路径。"""
    return get_vault_path() / "user"


def get_logs_dir() -> Path:
    """返回日志目录 ``vault/logs/`` 的路径。"""
    return get_vault_path() / "logs"


def get_db_path() -> Path:
    """返回数据库文件 ``vault/pneuma.db`` 的路径。"""
    return get_vault_path() / "pneuma.db"


def get_entity_dir(entity_name: str) -> Path:
    """返回实体数据目录 ``vault/{entity_name}/`` 的路径。

    用于以 vault/{entity}/ 的模式管理实体专属数据。
    例如: vault/mira/、vault/user/、vault/aine/ 等。
    """
    return get_vault_path() / entity_name


def get_entity_diary_dir(entity_name: str) -> Path:
    """返回实体的日记目录 ``vault/{entity_name}/diary/`` 的路径。"""
    return get_entity_dir(entity_name) / "diary"


def get_entity_relations_path(entity_name: str) -> Path:
    """返回实体的关系文件 ``vault/{entity_name}/relations.yaml`` 的路径。"""
    return get_entity_dir(entity_name) / "relations.yaml"
