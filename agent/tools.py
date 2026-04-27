"""Agent 可调用工具的注册与执行层。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from environment.adapter import EnvironmentAdapter


def _create_default_adapter() -> EnvironmentAdapter:
    """仅在注册表需要默认实现时创建生产 HTTP 适配器。"""

    from environment.remote_adapter import RemoteEnvironmentAdapter

    return RemoteEnvironmentAdapter()


@dataclass(slots=True)
class ToolDefinition:
    """工具元数据定义。"""

    name: str
    description: str
    parameters: dict[str, str]


@dataclass(slots=True)
class ToolRegistry:
    """管理 Agent 工具注册、描述和执行。"""

    adapter: EnvironmentAdapter = field(default_factory=_create_default_adapter)
    memory: Any | None = None
    _tools: dict[str, Callable[..., str]] = field(default_factory=dict, init=False)
    _definitions: list[ToolDefinition] = field(default_factory=list, init=False)
    _memory_tools_registered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._register_builtin_tools()
        if self.memory is not None:
            self.set_memory(self.memory)

    def set_memory(self, memory: Any | None) -> None:
        """接入长期记忆，并按需注册记忆管理工具。"""

        self.memory = memory
        if self.memory is not None and not self._memory_tools_registered:
            self._register_memory_tools()
            self._memory_tools_registered = True

    def execute(self, session_id: str, tool_name: str, args: dict[str, Any]) -> str:
        """执行指定工具并返回可读文本。"""

        handler = self._tools.get(tool_name)
        if handler is None:
            available_tools = ", ".join(sorted(self._tools))
            return f"错误：不存在名为 `{tool_name}` 的工具。可用工具：{available_tools}"

        try:
            return handler(session_id, **args)
        except TypeError as exc:
            return f"工具 `{tool_name}` 参数不合法：{exc}"
        except Exception as exc:
            return f"工具 `{tool_name}` 执行失败：{exc}"

    def get_definitions(self) -> list[ToolDefinition]:
        """返回工具定义列表。"""

        return list(self._definitions)

    def get_tools_prompt(self) -> str:
        """生成供提示词使用的工具说明。"""

        lines: list[str] = []
        for definition in self._definitions:
            params_text = ", ".join(f"{name}: {desc}" for name, desc in definition.parameters.items())
            lines.append(f"- {definition.name}({params_text})")
            lines.append(f"  说明：{definition.description}")
        return "\n".join(lines)

    def _register_builtin_tools(self) -> None:
        """注册内置工具。"""

        self._register(
            name="query_all_devices",
            description="查询当前会话内所有设备的状态。",
            parameters={},
            handler=self._tool_query_all_devices,
        )
        self._register(
            name="control_device",
            description="控制指定设备并返回执行结果。",
            parameters={
                "device_id": "设备 ID，例如 living_room_light_1",
                "command": "命令名称，例如 turn_on 或 set_temperature",
                "params": "命令参数字典，例如 {\"temperature\": 24}",
            },
            handler=self._tool_control_device,
        )

    def _register_memory_tools(self) -> None:
        """注册长期记忆管理工具。"""

        self._register(
            name="list_memories",
            description="列出当前用户已保存的长期记忆。只有用户明确要求查看记忆时调用。",
            parameters={},
            handler=self._tool_list_memories,
        )
        self._register(
            name="delete_memory",
            description="按 memory_id 删除当前用户的一条长期记忆。只有用户明确要求忘掉某条记忆时调用。",
            parameters={
                "memory_id": "要删除的长期记忆 ID，可先通过 list_memories 查询",
            },
            handler=self._tool_delete_memory,
        )
        self._register(
            name="clear_user_memory",
            description="清空当前用户的全部长期记忆。只有用户明确要求清空或删除全部记忆时调用。",
            parameters={},
            handler=self._tool_clear_user_memory,
        )

    def _register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        handler: Callable[..., str],
    ) -> None:
        """注册单个工具。"""

        self._tools[name] = handler
        self._definitions.append(ToolDefinition(name=name, description=description, parameters=parameters))

    def _tool_query_all_devices(self, session_id: str) -> str:
        """查询所有设备状态。"""

        state = self.adapter.fetch_state(session_id)
        devices = state.get("devices", {})
        if not devices:
            return "当前没有任何设备。"
        return "\n".join(
            self._describe_device(device_id=device_id, device=device)
            for device_id, device in devices.items()
        )

    def _tool_control_device(
        self,
        session_id: str,
        device_id: str,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """控制指定设备。"""

        normalized_device_id = device_id.strip()
        normalized_command = command.strip()
        if not normalized_device_id or not normalized_command:
            return "错误：`device_id` 和 `command` 不能为空。"

        device_type = self._infer_device_type(normalized_device_id)
        response = self.adapter.send_action(
            session_id=session_id,
            action={
                "device": device_type,
                "target": normalized_device_id,
                "command": normalized_command,
                "params": params or {},
            },
            intent=f"控制设备 {normalized_device_id}: {normalized_command}",
        )

        if response.get("success"):
            events = response.get("events", [])
            event_types = [event.get("type", "未知事件") for event in events]
            result = "操作成功。"
            if event_types:
                result += " 产生事件：" + "、".join(event_types) + "。"
            device_state = response.get("observation", {}).get("devices", {}).get(normalized_device_id)
            if device_state:
                result += " 设备当前状态：" + self._describe_device(normalized_device_id, device_state)
            return result

        error = response.get("error", {})
        return (
            "操作失败。"
            f"错误码={error.get('code', 'unknown')}，"
            f"错误信息={error.get('message', '未知错误')}"
        )

    def _tool_list_memories(self, session_id: str) -> str:
        """列出当前会话用户的长期记忆。"""

        if self.memory is None:
            return "长期记忆未启用。"
        return str(self.memory.list_memories(user_id=session_id))

    def _tool_delete_memory(self, session_id: str, memory_id: str) -> str:
        """删除当前会话用户的一条长期记忆。"""

        if self.memory is None:
            return "长期记忆未启用。"
        return str(self.memory.delete_memory(user_id=session_id, memory_id=memory_id))

    def _tool_clear_user_memory(self, session_id: str) -> str:
        """清空当前会话用户的长期记忆。"""

        if self.memory is None:
            return "长期记忆未启用。"
        return str(self.memory.clear_user_memory(user_id=session_id))

    @staticmethod
    def _infer_device_type(device_id: str) -> str:
        """根据设备 ID 推断环境协议中的设备类型。"""

        if "light" in device_id:
            return "light"
        if "ac" in device_id:
            return "ac"
        if "washing" in device_id:
            return "washing_machine"
        if "curtain" in device_id:
            return "curtain"
        if "sensor" in device_id:
            return "temperature_humidity_sensor"
        if "plug" in device_id:
            return "smart_plug"
        return "unknown"

    def _describe_device(self, device_id: str, device: dict[str, Any]) -> str:
        """将设备状态转为中文摘要。"""

        device_type = str(device.get("device_type", "unknown"))
        name = str(device.get("name", device_id))

        if device_type == "light":
            state = "开启" if device.get("is_on") else "关闭"
            brightness = device.get("brightness", 0)
            return f"[{name}] 状态：{state}，亮度={brightness}"

        if device_type == "ac":
            state = "开启" if device.get("is_on") else "关闭"
            mode_map = {"cool": "制冷", "heat": "制热", "fan": "送风", "dry": "除湿"}
            mode = mode_map.get(str(device.get("mode", "unknown")), str(device.get("mode", "unknown")))
            temperature = device.get("target_temperature", "未知")
            fan_speed = device.get("fan_speed", "未知")
            return f"[{name}] 状态：{state}，模式={mode}，目标温度={temperature}°C，风速={fan_speed}"

        if device_type == "washing_machine":
            status_map = {
                "idle": "空闲",
                "running": "运行中",
                "paused": "已暂停",
                "completed": "已完成",
                "cancelled": "已取消",
            }
            status = status_map.get(str(device.get("status", "unknown")), str(device.get("status", "unknown")))
            program = device.get("program", "标准")
            remaining_seconds = int(device.get("remaining_seconds", 0) or 0)
            description = f"[{name}] 状态：{status}，程序={program}"
            if remaining_seconds > 0:
                description += f"，剩余时间={self._format_duration(remaining_seconds)}"
            return description

        if device_type == "curtain":
            position_percent = int(device.get("position_percent", 0) or 0)
            if position_percent == 0:
                state = "关闭"
            elif position_percent == 100:
                state = "完全打开"
            else:
                state = "部分打开"
            return f"[{name}] 状态：{state}，开合度={position_percent}%"

        if device_type == "temperature_humidity_sensor":
            temperature = device.get("temperature", "未知")
            humidity = device.get("humidity", "未知")
            return f"[{name}] 温度={temperature}°C，湿度={humidity}%"

        if device_type == "smart_plug":
            state = "开启" if device.get("is_on") else "关闭"
            power_watts = device.get("power_watts", 0.0)
            return f"[{name}] 状态：{state}，功率={power_watts}W"

        return f"[{name}] 类型={device_type}，状态未知"

    @staticmethod
    def _format_duration(seconds: int) -> str:
        """格式化秒数。"""

        if seconds < 60:
            return f"{seconds} 秒"
        minutes, remain = divmod(seconds, 60)
        if remain == 0:
            return f"{minutes} 分钟"
        return f"{minutes} 分 {remain} 秒"
