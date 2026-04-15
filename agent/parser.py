"""模型输出解析与校验。"""

from __future__ import annotations

import json
from typing import Any

from agent.schema import AgentPlan


def parse_agent_plan(raw_text: str) -> AgentPlan:
	"""清洗模型输出并转换为结构化计划。"""

	clean_text = _strip_code_fence(raw_text)
	try:
		payload = json.loads(clean_text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"模型输出不是合法 JSON: {raw_text}") from exc

	plan_type = _require_string(payload, "plan_type")
	intent = _require_string(payload, "intent")
	reply_hint = payload.get("reply_hint")
	if reply_hint is not None and not isinstance(reply_hint, str):
		raise ValueError("reply_hint 必须是字符串或 null")

	if plan_type == "state_query":
		if payload.get("action") is not None:
			raise ValueError("state_query 的 action 必须为 null")
		return AgentPlan(
			plan_type=plan_type,
			intent=intent,
			reply_hint=reply_hint,
			action=None,
		)

	if plan_type != "environment_action":
		raise ValueError(f"不支持的 plan_type: {plan_type}")

	action = payload.get("action")
	if not isinstance(action, dict):
		raise ValueError("environment_action 的 action 必须是字典")

	for key in ("device", "target", "command"):
		_require_string(action, key)
	params = action.get("params")
	if params is None:
		action["params"] = {}
	elif not isinstance(params, dict):
		raise ValueError("action.params 必须是字典")

	return AgentPlan(
		plan_type=plan_type,
		intent=intent,
		reply_hint=reply_hint,
		action=action,
	)


def _strip_code_fence(raw_text: str) -> str:
	"""移除模型可能附带的 Markdown 代码块。"""

	text = raw_text.strip()
	if text.startswith("```"):
		lines = text.splitlines()
		if lines and lines[0].startswith("```"):
			lines = lines[1:]
		if lines and lines[-1].strip() == "```":
			lines = lines[:-1]
		text = "\n".join(lines).strip()
	return text


def _require_string(payload: dict[str, Any], key: str) -> str:
	value = payload.get(key)
	if not isinstance(value, str) or not value.strip():
		raise ValueError(f"{key} 必须是非空字符串")
	return value
