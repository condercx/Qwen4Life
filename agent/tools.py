"""ReAct agent 工具注册与执行。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from environment.remote_adapter import RemoteEnvironmentAdapter


@dataclass(slots=True)
class ToolDefinition:
	"""单个工具的元信息。"""

	name: str
	description: str
	parameters: dict[str, str]


@dataclass(slots=True)
class ToolRegistry:
	"""注册并执行 agent 可调用的工具。

	所有工具的返回值统一为人类可读中文文本，
	作为 Observation 注入到 ReAct 对话循环中。
	"""

	adapter: RemoteEnvironmentAdapter = field(default_factory=RemoteEnvironmentAdapter)
	_tools: dict[str, Callable] = field(default_factory=dict, init=False)
	_definitions: list[ToolDefinition] = field(default_factory=list, init=False)

	def __post_init__(self) -> None:
		self._register_builtin_tools()

	# ── 公共接口 ──────────────────────────────────────────────

	def execute(self, session_id: str, tool_name: str, args: dict[str, Any]) -> str:
		"""执行指定工具并返回文本化结果。"""

		handler = self._tools.get(tool_name)
		if handler is None:
			return f"错误：不存在名为 '{tool_name}' 的工具。可用工具：{', '.join(self._tools.keys())}"
		try:
			return handler(session_id, **args)
		except Exception as exc:
			return f"工具 {tool_name} 执行失败：{exc}"

	def get_definitions(self) -> list[ToolDefinition]:
		"""获取所有工具定义，用于构建 system prompt。"""

		return list(self._definitions)

	def get_tools_prompt(self) -> str:
		"""生成工具描述文本，嵌入 system prompt。"""

		lines: list[str] = []
		for defn in self._definitions:
			params_desc = ", ".join(f"{k}: {v}" for k, v in defn.parameters.items())
			lines.append(f"- {defn.name}({params_desc})")
			lines.append(f"  说明：{defn.description}")
		return "\n".join(lines)

	# ── 内置工具注册 ──────────────────────────────────────────

	def _register_builtin_tools(self) -> None:
		"""注册所有内置工具。"""

		self._register(
			name="query_all_devices",
			description="查询所有设备的当前状态",
			parameters={},
			handler=self._tool_query_all_devices,
		)
		self._register(
			name="control_device",
			description="控制指定设备。执行设备命令并返回结果",
			parameters={
				"device_id": "设备实例 ID，如 living_room_light_1",
				"command": "命令名，如 turn_on, set_temperature 等",
				"params": "命令参数字典，如 {\"temperature\": 24}（可选，默认为空）",
			},
			handler=self._tool_control_device,
		)

	def _register(
		self,
		name: str,
		description: str,
		parameters: dict[str, str],
		handler: Callable,
	) -> None:
		self._tools[name] = handler
		self._definitions.append(ToolDefinition(name=name, description=description, parameters=parameters))

	# ── 工具实现 ──────────────────────────────────────────────

	def _tool_query_all_devices(self, session_id: str) -> str:
		"""查询所有设备状态。"""

		state = self.adapter.fetch_state(session_id)
		devices = state.get("devices", {})
		if not devices:
			return "当前没有任何设备。"
		parts: list[str] = []
		for device_id, device in devices.items():
			parts.append(self._describe_device(device_id, device))
		return "\n".join(parts)

	def _tool_control_device(
		self,
		session_id: str,
		device_id: str,
		command: str,
		params: dict[str, Any] | None = None,
	) -> str:
		"""控制设备。"""

		# 从设备 ID 推断设备类型
		device_type = self._infer_device_type(device_id)
		action = {
			"device": device_type,
			"target": device_id,
			"command": command,
			"params": params or {},
		}
		response = self.adapter.send_action(
			session_id=session_id,
			action=action,
			intent=f"控制 {device_id}: {command}",
		)

		if response.get("success"):
			events = response.get("events", [])
			event_types = [evt.get("type", "未知事件") for evt in events]
			result = f"操作成功。"
			if event_types:
				result += f" 产生事件：{'、'.join(event_types)}。"
			# 附带操作后的设备状态
			obs = response.get("observation", {})
			device_state = obs.get("devices", {}).get(device_id)
			if device_state:
				result += f" 设备当前状态：{self._describe_device(device_id, device_state)}"
			return result

		error = response.get("error", {})
		return f"操作失败。错误码 {error.get('code')}：{error.get('message', '未知错误')}"

	# ── 辅助方法 ──────────────────────────────────────────────

	def _infer_device_type(self, device_id: str) -> str:
		"""从设备 ID 推断设备类型。"""

		if "light" in device_id:
			return "light"
		if "ac" in device_id:
			return "ac"
		if "washing" in device_id:
			return "washing_machine"
		return "unknown"

	def _describe_device(self, device_id: str, device: dict[str, Any]) -> str:
		"""将设备状态转为人类可读描述。"""

		device_type = device.get("device_type", "unknown")
		name = device.get("name", device_id)

		if device_type == "light":
			on_off = "开启" if device.get("is_on") else "关闭"
			brightness = device.get("brightness", 0)
			return f"[{name}] 状态：{on_off}，亮度 {brightness}"

		if device_type == "ac":
			on_off = "开启" if device.get("is_on") else "关闭"
			mode = device.get("mode", "未知")
			mode_map = {"cool": "制冷", "heat": "制热", "fan": "送风", "dry": "除湿"}
			mode_cn = mode_map.get(mode, mode)
			temp = device.get("target_temperature", "未知")
			fan = device.get("fan_speed", "未知")
			return f"[{name}] 状态：{on_off}，模式 {mode_cn}，目标温度 {temp}°C，风速 {fan}"

		if device_type == "washing_machine":
			status = device.get("status", "unknown")
			status_map = {
				"idle": "空闲",
				"running": "运行中",
				"paused": "已暂停",
				"completed": "已完成",
				"cancelled": "已取消",
			}
			status_cn = status_map.get(status, status)
			program = device.get("program", "标准")
			remaining = device.get("remaining_seconds", 0)
			desc = f"[{name}] 状态：{status_cn}，程序 {program}"
			if status == "running" and remaining > 0:
				desc += f"，剩余 {self._format_duration(remaining)}"
			return desc

		return f"[{name}] 类型 {device_type}，状态未知"

	def _format_duration(self, seconds: int) -> str:
		"""格式化时长。"""

		if seconds < 60:
			return f"{seconds} 秒"
		minutes, remain = divmod(seconds, 60)
		if remain == 0:
			return f"{minutes} 分钟"
		return f"{minutes} 分 {remain} 秒"
