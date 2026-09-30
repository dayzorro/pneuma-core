"""把 Markdown 知识文档切分为可检索的知识块。

切分策略：
    1. 解析可选的 front matter（``---`` 包裹的 YAML），取出 doc_id / title / source / tags
    2. 以二级及以下标题（``##``）为界切分小节；文档开头到第一个标题之间的内容
       作为「概述」小节，标题取文档标题
    3. 小节正文超过 ``max_chars`` 时，按空行分段后再按段落聚合成多个块

每个块的 ``title`` 形如「会员制度 · 积分规则」，便于检索时携带上下文。
"""

from __future__ import annotations

import re

import yaml

from pneuma_core.knowledge.models import DocumentMeta, KnowledgeChunk

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_H1_RE = re.compile(r"^#\s+.*$", re.MULTILINE)

DEFAULT_MAX_CHARS = 600

_TITLE_SEPARATOR = " · "


def parse_front_matter(text: str, *, fallback_doc_id: str) -> tuple[DocumentMeta, str]:
    """解析 front matter，返回（元数据, 去掉 front matter 后的正文）。"""
    meta = DocumentMeta(doc_id=fallback_doc_id, title=fallback_doc_id)
    body = text

    match = _FRONT_MATTER_RE.match(text)
    if match:
        try:
            raw = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            raw = {}
        if isinstance(raw, dict):
            raw_tags = raw.get("tags") or []
            # 支持 `tags: a, b` 与 `tags: [a, b]` 两种写法
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            meta = DocumentMeta(
                doc_id=str(raw.get("doc_id") or fallback_doc_id),
                title=str(raw.get("title") or fallback_doc_id),
                source=str(raw.get("source") or ""),
                tags=tuple(str(t) for t in raw_tags),
            )
        body = text[match.end() :]

    return meta, body


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    """按 ``##`` 及以下标题切分，返回 [(小节标题 | None, 正文)]。"""
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        # 一级标题属于文档标题，不作为小节
        if match and len(match.group(1)) >= 2:
            sections.append((current_heading, current_lines))
            current_heading = match.group(2)
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))

    result: list[tuple[str | None, str]] = []
    for heading, lines in sections:
        # 一级标题是文档标题，不是正文内容
        text = _H1_RE.sub("", "\n".join(lines)).strip()
        if text:
            result.append((heading, text))
    return result


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """把长正文按空行分段后聚合为不超过 max_chars 的若干段。"""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    pieces: list[str] = []
    buffer: list[str] = []
    length = 0

    for paragraph in paragraphs:
        # 单个段落本身就超长时，按行硬切
        if len(paragraph) > max_chars:
            if buffer:
                pieces.append("\n\n".join(buffer))
                buffer, length = [], 0
            lines = paragraph.splitlines()
            chunk_lines: list[str] = []
            chunk_len = 0
            for line in lines:
                if chunk_len + len(line) + 1 > max_chars and chunk_lines:
                    pieces.append("\n".join(chunk_lines))
                    chunk_lines, chunk_len = [], 0
                chunk_lines.append(line)
                chunk_len += len(line) + 1
            if chunk_lines:
                pieces.append("\n".join(chunk_lines))
            continue

        if length + len(paragraph) + 2 > max_chars and buffer:
            pieces.append("\n\n".join(buffer))
            buffer, length = [], 0
        buffer.append(paragraph)
        length += len(paragraph) + 2

    if buffer:
        pieces.append("\n\n".join(buffer))

    return pieces or [text]


def split_markdown(
    text: str,
    *,
    fallback_doc_id: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[KnowledgeChunk]:
    """把一篇 Markdown 文档切分为知识块列表。"""
    meta, body = parse_front_matter(text, fallback_doc_id=fallback_doc_id)

    chunks: list[KnowledgeChunk] = []
    index = 0

    for heading, section_text in _split_sections(body):
        title = (
            meta.title
            if heading is None
            else f"{meta.title}{_TITLE_SEPARATOR}{heading}"
        )
        for piece in _split_paragraphs(section_text, max_chars):
            chunks.append(
                KnowledgeChunk(
                    id=f"{meta.doc_id}#{index}",
                    doc_id=meta.doc_id,
                    title=title,
                    content=piece,
                    source=meta.source,
                    tags=meta.tags,
                )
            )
            index += 1

    return chunks
