#!/usr/bin/env python
"""构建本地知识库的向量索引。

把 ``--docs`` 目录下的 Markdown 资料切块、调用 embedding 接口生成向量，
并落盘为一个自包含的 JSON 索引文件（默认 ``vault/knowledge_index.json``）。
服务启动时直接加载该索引，因此**不需要**在每次启动时调用 embedding 接口。

用法::

    # 先加载环境变量（embedding 端点与模型）
    set -a && . ./.env && set +a
    .venv/bin/python scripts/build_knowledge_index.py

    # 换一份语料 / 换一个输出位置
    .venv/bin/python scripts/build_knowledge_index.py \\
        --docs path/to/docs --out vault/my_index.json

不构建索引也能跑：服务会自动退化为「关键词检索」模式，只是召回质量差一些。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from pneuma_core.knowledge import DEFAULT_DATA_DIR
from pneuma_core.knowledge.index import DEFAULT_BATCH_SIZE, build_index
from pneuma_core.llm.embedding import OpenAIEmbeddingService

DEFAULT_OUTPUT = Path("vault/knowledge_index.json")


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("PNEUMA_EMBEDDING_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        print(
            "错误：未找到 embedding API key。请先加载 .env：\n"
            "    set -a && . ./.env && set +a",
            file=sys.stderr,
        )
        return 1

    model = args.model or os.environ.get("PNEUMA_EMBEDDING_MODEL", "")
    if not model:
        print(
            "错误：未指定 embedding 模型。请设置 PNEUMA_EMBEDDING_MODEL 或传 --model。",
            file=sys.stderr,
        )
        return 1

    base_url = os.environ.get("PNEUMA_EMBEDDING_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )

    service = OpenAIEmbeddingService(
        api_key=api_key, model=model, base_url=base_url
    )

    print(f"语料目录: {args.docs}")
    print(f"向量模型: {model}")
    print(f"输出索引: {args.out}")

    index = await build_index(
        args.docs,
        service,
        model=model,
        batch_size=args.batch_size,
    )
    index.save(args.out)

    print(
        f"完成：{len(index.chunks)} 个知识块，"
        f"{index.dimension} 维向量 → {args.out}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 Pneuma Core 本地知识库向量索引"
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Markdown 语料目录（默认 {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"索引输出路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument("--model", default=None, help="覆盖 PNEUMA_EMBEDDING_MODEL")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"单次请求的最大条数（默认 {DEFAULT_BATCH_SIZE}）",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
