"""ReAct 输出解析器。"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.schema import ReactStep

ANSWER_PATTERN = re.compile(r"""["']?Answer["']?\s*:\s*["']?(.*)""", re.DOTALL)
THOUGHT_PATTERN = re.compile(r"""["']?Thought["']?\s*:\s*(.*)""", re.DOTALL)
ACTION_HEAD_PATTERN = re.compile(r"""["']?Action["']?\s*:\s*["']?([a-zA-Z0-9_]+)\(""")


def parse_react_output(raw_text: str) -> ReactStep:
    """解析模型输出为标准 ReAct 步骤。"""

    text = raw_text.strip()
    if not text:
        return ReactStep(type="empty", content="")

    action_call = _extract_first_action_call(text)
    if action_call is not None:
        tool_name, args_text, start_index, end_index = action_call
        thought_text = _extract_thought_prefix(text[:start_index])
        return ReactStep(
            type="action",
            content=thought_text,
            tool_name=tool_name,
            tool_args=_parse_tool_args(args_text),
            raw_action_text=text[start_index:end_index].strip(),
        )

    answer_match = ANSWER_PATTERN.search(text)
    if answer_match:
        answer_text = _strip_wrapping_quotes(answer_match.group(1).strip())
        if not answer_text:
            return ReactStep(type="empty", content="")
        return ReactStep(type="answer", content=answer_text)

    thought_match = THOUGHT_PATTERN.match(text)
    if thought_match:
        return ReactStep(type="thought", content=thought_match.group(1).strip())

    return ReactStep(type="answer", content=text)


def _extract_thought_prefix(text: str) -> str:
    """提取动作前的 Thought 内容。"""

    cleaned_text = text.strip()
    cleaned_text = re.sub(
        r"""^\{?\s*["']?Thought["']?\s*:\s*["']?""",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    return _strip_wrapping_quotes(cleaned_text.strip(" ,"))


def _normalize_args_text(args_text: str) -> str:
    """清理工具参数文本。"""

    return args_text.strip().replace('\\"', '"')


def _extract_first_action_call(text: str) -> tuple[str, str, int, int] | None:
    """提取第一个完整 Action 调用，忽略模型误输出的后续 Action。"""

    action_match = ACTION_HEAD_PATTERN.search(text)
    if action_match is None:
        return None

    args_start = action_match.end()
    args_end = _find_matching_paren(text, args_start)
    if args_end is None:
        return None

    tool_name = action_match.group(1).strip()
    args_text = _normalize_args_text(text[args_start:args_end])
    return tool_name, args_text, action_match.start(), args_end + 1


def _find_matching_paren(text: str, start_index: int) -> int | None:
    """从参数起点开始寻找第一个 Action 调用的闭合括号。"""

    quote: str | None = None
    escape_next = False
    depth = 1
    for index in range(start_index, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _parse_tool_args(args_text: str) -> dict[str, Any]:
    """解析工具参数。支持 JSON 和 `key=value` 格式。"""

    if not args_text:
        return {}

    stripped_args = args_text.strip()
    if stripped_args.startswith("{"):
        try:
            parsed_json = json.loads(stripped_args)
            if isinstance(parsed_json, dict):
                return parsed_json
        except json.JSONDecodeError:
            pass

    result: dict[str, Any] = {}
    for item in _split_top_level_args(stripped_args):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        result[normalized_key] = _parse_value(raw_value.strip())
    return result


def _split_top_level_args(args_text: str) -> list[str]:
    """按顶层逗号切分参数，避免拆坏嵌套结构。"""

    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    bracket_depth = 0
    brace_depth = 0
    paren_depth = 0
    escape_next = False

    for char in args_text:
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\":
            current.append(char)
            escape_next = True
            continue

        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue

        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)

        if char == "," and not any([brace_depth, bracket_depth, paren_depth]):
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_value(value_text: str) -> Any:
    """解析单个参数值。"""

    if not value_text:
        return ""

    stripped_text = _strip_wrapping_quotes(value_text)
    if stripped_text != value_text:
        return stripped_text

    try:
        return json.loads(value_text)
    except (json.JSONDecodeError, ValueError):
        return value_text


def _strip_wrapping_quotes(text: str) -> str:
    """去掉最外层对称引号。"""

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text
