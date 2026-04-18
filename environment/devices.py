"""设备模型与设备级业务规则。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Any

from environment.actions import (
    ERROR_DEVICE_OFFLINE,
    ERROR_INVALID_PARAM,
    ERROR_UNSUPPORTED_COMMAND,
    ProtocolError,
)

SUPPORTED_AC_MODES = {"cool", "heat", "fan", "dry"}
SUPPORTED_WASH_PROGRAMS = {"standard", "quick"}


@dataclass(slots=True)
class Device:
    """所有设备的基础字段与默认行为。"""

    device_id: str
    device_type: str
    name: str
    online: bool = True

    def ensure_online(self) -> None:
        """确保设备处于在线状态。"""

        if not self.online:
            raise ProtocolError(ERROR_DEVICE_OFFLINE, f"设备 `{self.device_id}` 当前离线。")

    def handle_command(
        self,
        command: str,
        params: dict[str, Any],
        current_time: float,
    ) -> list[dict[str, Any]]:
        """处理设备命令，默认不支持任何动作。"""

        raise ProtocolError(
            ERROR_UNSUPPORTED_COMMAND,
            f"`{self.device_type}` 不支持命令 `{command}`。",
        )

    def sync_time(self, current_time: float) -> list[dict[str, Any]]:
        """执行与时间相关的状态推进。"""

        return []

    def snapshot(self) -> dict[str, Any]:
        """返回设备当前快照。"""

        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "name": self.name,
            "online": self.online,
        }


@dataclass(slots=True)
class Light(Device):
    """灯光设备，支持开关与亮度调节。"""

    is_on: bool = False
    brightness: int = 0

    def handle_command(
        self,
        command: str,
        params: dict[str, Any],
        current_time: float,
    ) -> list[dict[str, Any]]:
        """处理灯光命令。"""

        del current_time
        self.ensure_online()

        match command:
            case "turn_on":
                self.is_on = True
                self.brightness = max(self.brightness, 100)
            case "turn_off":
                self.is_on = False
                self.brightness = 0
            case "set_brightness":
                brightness = _require_int_in_range(params, "brightness", 0, 100)
                self.brightness = brightness
                self.is_on = brightness > 0
            case _:
                raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"灯光不支持命令 `{command}`。")

        event_type = "light_brightness_changed" if command == "set_brightness" else "light_state_changed"
        return [
            {
                "type": event_type,
                "source": self.device_id,
                "payload": {"is_on": self.is_on, "brightness": self.brightness},
            }
        ]

    def snapshot(self) -> dict[str, Any]:
        """返回灯光设备快照。"""

        return {
            **Device.snapshot(self),
            "is_on": self.is_on,
            "brightness": self.brightness,
        }


@dataclass(slots=True)
class AirConditioner(Device):
    """空调设备，支持模式、温度和风速设置。"""

    is_on: bool = False
    mode: str = "cool"
    target_temperature: float = 26.0
    fan_speed: float = 1.0

    def handle_command(
        self,
        command: str,
        params: dict[str, Any],
        current_time: float,
    ) -> list[dict[str, Any]]:
        """处理空调命令。"""

        del current_time
        self.ensure_online()

        match command:
            case "turn_on":
                self.is_on = True
            case "turn_off":
                self.is_on = False
            case "set_mode":
                mode = _require_choice(params, "mode", SUPPORTED_AC_MODES)
                self.mode = mode
                self.is_on = True
            case "set_temperature":
                temperature = round(_require_float_in_range(params, "temperature", 16.0, 30.0), 1)
                self.target_temperature = temperature
                self.is_on = True
            case "set_fan_speed":
                fan_speed = round(_require_float_in_range(params, "fan_speed", 0.1, 5.0), 2)
                self.fan_speed = fan_speed
                self.is_on = True
            case _:
                raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"空调不支持命令 `{command}`。")

        event_type = "ac_state_changed" if command in {"turn_on", "turn_off", "set_mode"} else "ac_setting_changed"
        return [
            {
                "type": event_type,
                "source": self.device_id,
                "payload": {
                    "is_on": self.is_on,
                    "mode": self.mode,
                    "target_temperature": self.target_temperature,
                    "fan_speed": self.fan_speed,
                },
            }
        ]

    def snapshot(self) -> dict[str, Any]:
        """返回空调设备快照。"""

        return {
            **Device.snapshot(self),
            "is_on": self.is_on,
            "mode": self.mode,
            "target_temperature": self.target_temperature,
            "fan_speed": self.fan_speed,
        }


@dataclass(slots=True)
class WashingMachine(Device):
    """洗衣机设备，支持定时推进与状态流转。"""

    status: str = "idle"
    program: str = "standard"
    duration_seconds: int = 1800
    remaining_seconds: int = 0
    started_at: float | None = None
    expected_finish_at: float | None = None
    paused_at: float | None = None
    completed_at: float | None = None

    def handle_command(
        self,
        command: str,
        params: dict[str, Any],
        current_time: float,
    ) -> list[dict[str, Any]]:
        """处理洗衣机命令。"""

        self.ensure_online()
        self._refresh_remaining(current_time)

        match command:
            case "start_wash":
                return self._start_wash(params, current_time)
            case "pause":
                return self._pause(current_time)
            case "resume":
                return self._resume(current_time)
            case "cancel":
                return self._cancel()
            case _:
                raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"洗衣机不支持命令 `{command}`。")

    def sync_time(self, current_time: float) -> list[dict[str, Any]]:
        """推进洗衣机的计时任务。"""

        if self.status != "running":
            return []

        self._refresh_remaining(current_time)
        if self.expected_finish_at is None or current_time < self.expected_finish_at:
            return []

        self.status = "completed"
        self.remaining_seconds = 0
        self.completed_at = self.expected_finish_at
        self.expected_finish_at = None
        self.paused_at = None
        return [
            {
                "type": "washing_completed",
                "source": self.device_id,
                "payload": {
                    "program": self.program,
                    "completed_at": _format_timestamp(self.completed_at),
                },
            }
        ]

    def snapshot(self) -> dict[str, Any]:
        """返回洗衣机设备快照。"""

        return {
            **Device.snapshot(self),
            "status": self.status,
            "program": self.program,
            "duration_seconds": self.duration_seconds,
            "remaining_seconds": self.remaining_seconds,
            "started_at": _format_timestamp(self.started_at),
            "expected_finish_at": _format_timestamp(self.expected_finish_at),
            "paused_at": _format_timestamp(self.paused_at),
            "completed_at": _format_timestamp(self.completed_at),
        }

    def _start_wash(self, params: dict[str, Any], current_time: float) -> list[dict[str, Any]]:
        """启动新的洗衣任务。"""

        if self.status == "running":
            raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机正在运行，不能重复启动。")

        program = _require_choice(params, "program", SUPPORTED_WASH_PROGRAMS, default="standard")
        duration_seconds = _require_positive_int(params, "duration_seconds", default=1800)
        self.status = "running"
        self.program = program
        self.duration_seconds = duration_seconds
        self.remaining_seconds = duration_seconds
        self.started_at = current_time
        self.expected_finish_at = current_time + duration_seconds
        self.paused_at = None
        self.completed_at = None
        return [
            {
                "type": "washing_started",
                "source": self.device_id,
                "payload": {
                    "program": self.program,
                    "duration_seconds": self.duration_seconds,
                    "expected_finish_at": _format_timestamp(self.expected_finish_at),
                },
            }
        ]

    def _pause(self, current_time: float) -> list[dict[str, Any]]:
        """暂停当前洗衣任务。"""

        if self.status != "running":
            raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机当前不在运行状态，无法暂停。")

        self._refresh_remaining(current_time)
        self.status = "paused"
        self.paused_at = current_time
        self.expected_finish_at = None
        return [
            {
                "type": "washing_paused",
                "source": self.device_id,
                "payload": {"remaining_seconds": self.remaining_seconds},
            }
        ]

    def _resume(self, current_time: float) -> list[dict[str, Any]]:
        """继续已暂停的洗衣任务。"""

        if self.status != "paused":
            raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机当前不在暂停状态，无法继续。")

        self.status = "running"
        self.paused_at = None
        self.expected_finish_at = current_time + self.remaining_seconds
        return [
            {
                "type": "washing_resumed",
                "source": self.device_id,
                "payload": {
                    "remaining_seconds": self.remaining_seconds,
                    "expected_finish_at": _format_timestamp(self.expected_finish_at),
                },
            }
        ]

    def _cancel(self) -> list[dict[str, Any]]:
        """取消当前洗衣任务。"""

        if self.status not in {"running", "paused", "completed"}:
            raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机当前没有可取消的任务。")

        self.status = "cancelled"
        self.remaining_seconds = 0
        self.expected_finish_at = None
        self.paused_at = None
        return [
            {
                "type": "washing_cancelled",
                "source": self.device_id,
                "payload": {"program": self.program},
            }
        ]

    def _refresh_remaining(self, current_time: float) -> None:
        """根据当前时间刷新剩余时长。"""

        if self.status != "running" or self.expected_finish_at is None:
            return
        self.remaining_seconds = max(0, ceil(self.expected_finish_at - current_time))


def _require_numeric_value(params: dict[str, Any], key: str) -> float:
    """读取并校验数值类型参数。"""

    value = params.get(key)
    if not isinstance(value, (int, float)):
        raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须是数值。")
    return float(value)


def _require_int_in_range(params: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    """读取并校验整数范围。"""

    value = _require_numeric_value(params, key)
    if not value.is_integer():
        raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须是整数。")
    integer = int(value)
    if not minimum <= integer <= maximum:
        raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须在 {minimum} 到 {maximum} 之间。")
    return integer


def _require_float_in_range(params: dict[str, Any], key: str, minimum: float, maximum: float) -> float:
    """读取并校验浮点范围。"""

    value = _require_numeric_value(params, key)
    if not minimum <= value <= maximum:
        raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须在 {minimum} 到 {maximum} 之间。")
    return value


def _require_positive_int(params: dict[str, Any], key: str, default: int | None = None) -> int:
    """读取并校验正整数参数。"""

    if key not in params:
        if default is None:
            raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须是正整数。")
        return default

    value = _require_numeric_value(params, key)
    if not value.is_integer() or value <= 0:
        raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须是正整数。")
    return int(value)


def _require_choice(
    params: dict[str, Any],
    key: str,
    allowed_values: set[str],
    default: str | None = None,
) -> str:
    """读取并校验枚举型字符串参数。"""

    raw_value = params.get(key, default)
    if not isinstance(raw_value, str) or raw_value not in allowed_values:
        allowed_text = "/".join(sorted(allowed_values))
        raise ProtocolError(ERROR_INVALID_PARAM, f"`{key}` 必须是以下值之一：{allowed_text}。")
    return raw_value


def _format_timestamp(timestamp: float | None) -> str | None:
    """将时间戳格式化为 ISO 字符串。"""

    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
