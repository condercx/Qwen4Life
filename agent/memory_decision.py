"""长期记忆保存决策。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

VALID_MEMORY_TYPES = {"preference", "alias", "habit", "home_rule", "agreement"}


@dataclass(slots=True)
class MemoryDecision:
    """Agent 对本轮是否需要保存长期记忆的结构化判断。"""

    should_save: bool
    memory_type: str = ""
    memory_text: str = ""


def build_memory_decision_messages(user_input: str, assistant_reply: str) -> list[dict[str, str]]:
    """构造让 LLM 判断是否保存长期记忆的消息。"""

    system_prompt = """\
/no_think
你是智能家居 Agent 的长期记忆筛选器。
请判断本轮对话是否包含长期有价值的信息，直接输出 JSON，不要输出思考过程、解释或 Markdown。

只有以下信息值得保存：
- 用户偏好：preference，例如温度、亮度、模式偏好。
- 设备别名：alias，例如把某个设备叫做某个名字。
- 用户习惯：habit，例如每天、每晚、回家后通常做什么。
- 家庭规则：home_rule，例如某个时间段不要运行设备。
- 历史约定：agreement，例如以后默认怎么处理。

不要保存：
- 普通寒暄、一次性设备控制、实时设备状态、工具执行结果。
- fallback、异常、空回答。
- Thought、Action、Observation、raw messages 或内部检索细节。

输出 JSON 格式：
{
  "should_save": true 或 false,
  "memory_type": "preference|alias|habit|home_rule|agreement",
  "memory_text": "一句简洁、可独立理解的中文记忆"
}

如果不应保存，输出：
{"should_save": false, "memory_type": "", "memory_text": ""}
"""
    user_prompt = f"""\
用户输入：
{user_input}

助手最终回答：
{assistant_reply}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def request_memory_decision(client: Any, user_input: str, assistant_reply: str) -> MemoryDecision:
    """请求 agent 对本轮长期记忆保存做判断。"""

    messages = build_memory_decision_messages(user_input, assistant_reply)
    if hasattr(client, "chat_completion_with_overrides"):
        raw_text = client.chat_completion_with_overrides(
            messages,
            max_tokens=256,
            temperature=0.0,
            enable_thinking=False,
            thinking_budget=0,
            force_json_output=True,
        )
    else:
        raw_text = client.chat_completion(messages)
    return parse_memory_decision(raw_text)


def parse_memory_decision(raw_text: str) -> MemoryDecision:
    """解析 LLM 的长期记忆保存决策。"""

    payload = _safe_load_json_object(raw_text)
    if payload is None:
        return MemoryDecision(should_save=False)

    should_save = bool(payload.get("should_save"))
    memory_type = str(payload.get("memory_type") or "").strip()
    memory_text = _clean_memory_text(str(payload.get("memory_text") or ""))

    if not should_save:
        return MemoryDecision(should_save=False)
    if memory_type not in VALID_MEMORY_TYPES or not memory_text:
        return MemoryDecision(should_save=False)
    return MemoryDecision(
        should_save=True,
        memory_type=memory_type,
        memory_text=memory_text,
    )


def _safe_load_json_object(raw_text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON object。"""

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "should_save" in result:
            candidates.append(result)
    return candidates[-1] if candidates else None


def _clean_memory_text(text: str) -> str:
    """清理模型生成的记忆文本，避免保存内部过程。"""

    lines = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Thought:", "Action:", "Observation:")):
            continue
        lines.append(stripped)
    return "\n".join(lines)
