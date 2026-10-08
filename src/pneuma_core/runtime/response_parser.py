"""Response parser: parse structured LLM output (speech/thought/action).

Follows the same JSON parsing pattern as diary_processor._parse_json_response.
"""

from __future__ import annotations

import json
import re

from pneuma_core.models.message import StructuredResponse

# Keys that identify a structured response
_STRUCTURED_KEYS = frozenset({"speech", "thought", "action"})

# 匹配 "speech" 键及其后的起始引号
_SPEECH_KEY_RE = re.compile(r'"speech"\s*:\s*"')

# JSON 字符串中的简单转义
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
}


def extract_partial_speech(buffer: str) -> str:
    """从（可能不完整的）流式 JSON 缓冲区中提取 speech 字段的当前值。

    用于流式输出：模型按 ``{"speech": "...", ...}`` 的 JSON 逐个 token 生成，
    本函数在任意时刻安全地取出已经生成的那部分 speech 文本（含转义与
    不完整的 ``\\uXXXX`` 处理）。

    当缓冲区明显不是 JSON（纯文本兜底回复）时，直接返回原文以支持流式。

    Args:
        buffer: 目前已累积的原始输出。

    Returns:
        当前可用的 speech 文本（可能为空字符串）。
    """
    text = buffer

    # 去掉可能的前导 markdown 代码围栏
    stripped = text.lstrip()
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl == -1:
            return ""
        offset = len(text) - len(stripped)
        text = text[offset + nl + 1:]

    # 纯文本兜底：首个非空白字符既不是 { 也不是 ` 时，按原文流式输出
    head = text.lstrip()[:1]
    if head and head not in "{`":
        return text

    if "{" not in text:
        # 还没开始 JSON，继续等待
        return ""

    match = _SPEECH_KEY_RE.search(text)
    if not match:
        return ""

    out: list[str] = []
    i = match.end()
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 >= n:
                break  # 转义序列尚未到达
            nxt = text[i + 1]
            if nxt == "u":
                if i + 6 > n:
                    break  # \uXXXX 不完整
                try:
                    out.append(chr(int(text[i + 2:i + 6], 16)))
                except ValueError:
                    break
                i += 6
                continue
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            break  # speech 字符串结束
        out.append(ch)
        i += 1

    return "".join(out)


def speech_field_complete(buffer: str) -> bool:
    """判断流式缓冲区里的 ``speech`` 字段是否已经闭合。

    用于「speech 一旦说完就先发完成事件」的流式优化：只要检测到 speech
    值的结束引号，就说明台词已经生成完，后续的 thought / action 还在这条
    流里继续生成，不必等它们。

    Args:
        buffer: 目前已累积的原始输出。

    Returns:
        speech 字段的值是否已经闭合（``"speech": null`` 这类非字符串值
        不会被判定为闭合，交由调用方在流结束时兜底）。
    """
    text = buffer

    stripped = text.lstrip()
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl == -1:
            return False
        offset = len(text) - len(stripped)
        text = text[offset + nl + 1:]

    if "{" not in text:
        return False

    match = _SPEECH_KEY_RE.search(text)
    if not match:
        return False

    i = match.end()
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 >= n:
                return False
            if text[i + 1] == "u":
                if i + 6 > n:
                    return False
                i += 6
                continue
            i += 2
            continue
        if ch == '"':
            return True
        i += 1

    return False


def parse_structured_response(raw: str) -> StructuredResponse:
    """Parse a raw LLM response into a StructuredResponse.

    Parsing strategy:
        1. Strip markdown code blocks (```json ... ```)
        2. Try json.loads()
        3. If valid dict with at least one structured key -> StructuredResponse
        4. Otherwise, fallback: entire raw text becomes speech

    Args:
        raw: Raw text from LLM response.

    Returns:
        StructuredResponse with speech/thought/action fields.
    """
    text = raw.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON -> fallback to plain text
        return StructuredResponse(speech=raw)

    if not isinstance(parsed, dict):
        # Non-dict JSON (e.g. array) -> fallback
        return StructuredResponse(speech=raw)

    # Check if it has at least one structured key
    if not _STRUCTURED_KEYS & set(parsed.keys()):
        # No structured keys -> fallback
        return StructuredResponse(speech=raw)

    return StructuredResponse(
        speech=parsed.get("speech"),
        thought=parsed.get("thought"),
        action=parsed.get("action"),
    )
