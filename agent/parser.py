"""ReAct 格式文本解析。"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.schema import ReactStep


def parse_react_output(raw_text: str) -> ReactStep:
	"""解析模型输出，提取 Thought / Action / Answer。

	支持格式：
	  Thought: ...
	  Action: tool_name(arg1=value1, arg2=value2)
	或
	  Answer: ...
	或直接无标签的自然语言（视为 Answer）。
	"""

	text = raw_text.strip()

	# 1. 尝试提取 Action (允许前面和后面带双引号，适应有些模型擅自输出 JSON 格式)
	action_match = re.search(
		r"[\"']?Action[\"']?\s*:\s*[\"']?([a-zA-Z0-9_]+)\((.*?)\)[\"']?",
		text,
		re.MULTILINE | re.DOTALL,
	)
	if action_match:
		tool_name = action_match.group(1).strip()
		args_raw = (action_match.group(2) or "").strip()
		# 清理 JSON 转义带来的问题
		args_raw = args_raw.replace('\\"', '"')
		tool_args = _parse_tool_args(args_raw)
		# 提取 Thought 部分（Action 之前的内容）
		thought_text = text[:action_match.start()].strip()
		thought_text = re.sub(r"^{?\s*[\"']?Thought[\"']?\s*:\s*[\"']?", "", thought_text, flags=re.IGNORECASE).strip()
		thought_text = thought_text.strip('",')
		return ReactStep(
			type="action",
			content=thought_text,
			tool_name=tool_name,
			tool_args=tool_args,
		)

	# 2. 尝试提取 Answer
	answer_match = re.search(r"[\"']?Answer[\"']?\s*:\s*[\"']?(.*)", text, re.DOTALL)
	if answer_match:
		answer_text = answer_match.group(1).strip()
		answer_text = re.sub(r"[\"']?\s*}?\s*$", "", answer_text).strip() # 移除末尾潜在的结尾括号和引号
		return ReactStep(type="answer", content=answer_text)

	# 3. 如果只有 Thought 没有 Action/Answer
	thought_match = re.match(r"Thought:\s*(.*)", text, re.DOTALL)
	if thought_match:
		# 纯思考，没有给出 Action 或 Answer
		return ReactStep(type="thought", content=thought_match.group(1).strip())

	# 4. 不符合任何格式 → 视为直接回复（Answer）
	return ReactStep(type="answer", content=text)


def _parse_tool_args(args_raw: str) -> dict[str, Any]:
	"""解析工具参数。

	支持两种格式：
	1. JSON 风格：{"device_id": "xxx", "command": "yyy"}
	2. 关键字风格：device_id="xxx", command="yyy"
	"""

	if not args_raw:
		return {}

	# 尝试 JSON 格式
	args_raw_stripped = args_raw.strip()
	if args_raw_stripped.startswith("{"):
		try:
			return json.loads(args_raw_stripped)
		except json.JSONDecodeError:
			pass

	# 尝试关键字参数格式
	result: dict[str, Any] = {}
	# 匹配 key=value 或 key="value" 或 key={...}
	pattern = re.compile(r'(\w+)\s*=\s*({[^}]*}|"[^"]*"|\'[^\']*\'|[^,\s]+)')
	for match in pattern.finditer(args_raw):
		key = match.group(1)
		value_str = match.group(2).strip()
		result[key] = _parse_value(value_str)

	return result


def _parse_value(value_str: str) -> Any:
	"""将字符串值解析为 Python 对象。"""

	# 去引号
	if len(value_str) >= 2 and value_str[0] == value_str[-1] and value_str[0] in {'"', "'"}:
		return value_str[1:-1]

	# 尝试 JSON（处理 dict/list/number/bool/null）
	try:
		return json.loads(value_str)
	except (json.JSONDecodeError, ValueError):
		pass

	# 原样返回字符串
	return value_str
