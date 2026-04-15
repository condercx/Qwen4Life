"""设备模型与计时任务规则。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Any

from environment.actions import ERROR_DEVICE_OFFLINE, ERROR_INVALID_PARAM, ERROR_UNSUPPORTED_COMMAND, ProtocolError


@dataclass(slots=True)
class Device:
	"""所有设备的基础字段。"""

	device_id: str
	device_type: str
	name: str
	online: bool = True

	def ensure_online(self) -> None:
		if not self.online:
			raise ProtocolError(ERROR_DEVICE_OFFLINE, f"设备 {self.device_id} 当前离线")

	def handle_command(self, command: str, params: dict[str, Any], current_time: float) -> list[dict[str, Any]]:
		raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"{self.device_type} 不支持命令 {command}")

	def sync_time(self, current_time: float) -> list[dict[str, Any]]:
		return []

	def snapshot(self) -> dict[str, Any]:
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

	def handle_command(self, command: str, params: dict[str, Any], current_time: float) -> list[dict[str, Any]]:
		self.ensure_online()
		if command == "turn_on":
			self.is_on = True
			self.brightness = max(self.brightness, 100)
		elif command == "turn_off":
			self.is_on = False
			self.brightness = 0
		elif command == "set_brightness":
			brightness = int(_require_number(params, "brightness"))
			if brightness < 0 or brightness > 100:
				raise ProtocolError(ERROR_INVALID_PARAM, "brightness 必须位于 0-100")
			self.brightness = brightness
			self.is_on = brightness > 0
		else:
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"灯光不支持命令 {command}")

		event_type = "light_brightness_changed" if command == "set_brightness" else "light_state_changed"
		return [{
			"type": event_type,
			"source": self.device_id,
			"payload": {"is_on": self.is_on, "brightness": self.brightness},
		}]

	def snapshot(self) -> dict[str, Any]:
		data = super().snapshot()
		data.update({"is_on": self.is_on, "brightness": self.brightness})
		return data


@dataclass(slots=True)
class AirConditioner(Device):
	"""空调设备，支持模式、目标温度和风速控制。"""

	is_on: bool = False
	mode: str = "cool"
	target_temperature: float = 26.0
	fan_speed: float = 1.0

	def handle_command(self, command: str, params: dict[str, Any], current_time: float) -> list[dict[str, Any]]:
		self.ensure_online()
		if command == "turn_on":
			self.is_on = True
		elif command == "turn_off":
			self.is_on = False
		elif command == "set_mode":
			mode = params.get("mode")
			if mode not in {"cool", "heat", "fan", "dry"}:
				raise ProtocolError(ERROR_INVALID_PARAM, "mode 必须是 cool/heat/fan/dry 之一")
			self.mode = mode
			self.is_on = True
		elif command == "set_temperature":
			temperature = round(_require_number(params, "temperature"), 1)
			if temperature < 16.0 or temperature > 30.0:
				raise ProtocolError(ERROR_INVALID_PARAM, "temperature 必须位于 16-30")
			self.target_temperature = temperature
			self.is_on = True
		elif command == "set_fan_speed":
			fan_speed = round(_require_number(params, "fan_speed"), 2)
			if fan_speed < 0.1 or fan_speed > 5.0:
				raise ProtocolError(ERROR_INVALID_PARAM, "fan_speed 必须位于 0.1-5.0")
			self.fan_speed = fan_speed
			self.is_on = True
		else:
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"空调不支持命令 {command}")

		event_type = "ac_state_changed" if command in {"turn_on", "turn_off", "set_mode"} else "ac_setting_changed"
		return [{
			"type": event_type,
			"source": self.device_id,
			"payload": {
				"is_on": self.is_on,
				"mode": self.mode,
				"target_temperature": self.target_temperature,
				"fan_speed": self.fan_speed,
			},
		}]

	def snapshot(self) -> dict[str, Any]:
		data = super().snapshot()
		data.update({
			"is_on": self.is_on,
			"mode": self.mode,
			"target_temperature": self.target_temperature,
			"fan_speed": self.fan_speed,
		})
		return data


@dataclass(slots=True)
class WashingMachine(Device):
	"""洗衣机设备，支持后台计时任务。"""

	status: str = "idle"
	program: str = "standard"
	duration_seconds: int = 1800
	remaining_seconds: int = 0
	started_at: float | None = None
	expected_finish_at: float | None = None
	paused_at: float | None = None
	completed_at: float | None = None

	def handle_command(self, command: str, params: dict[str, Any], current_time: float) -> list[dict[str, Any]]:
		self.ensure_online()
		self._refresh_remaining(current_time)
		if command == "start_wash":
			if self.status == "running":
				raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机正在运行，不能重复启动")
			program = params.get("program", "standard")
			duration_seconds = int(_require_positive_number(params, "duration_seconds", default=1800))
			self.status = "running"
			self.program = str(program)
			self.duration_seconds = duration_seconds
			self.remaining_seconds = duration_seconds
			self.started_at = current_time
			self.expected_finish_at = current_time + duration_seconds
			self.paused_at = None
			self.completed_at = None
			return [{
				"type": "washing_started",
				"source": self.device_id,
				"payload": {
					"program": self.program,
					"duration_seconds": self.duration_seconds,
					"expected_finish_at": _format_timestamp(self.expected_finish_at),
				},
			}]
		if command == "pause":
			if self.status != "running":
				raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机当前不在运行，无法暂停")
			self.status = "paused"
			self.paused_at = current_time
			self.expected_finish_at = None
			return [{
				"type": "washing_paused",
				"source": self.device_id,
				"payload": {"remaining_seconds": self.remaining_seconds},
			}]
		if command == "resume":
			if self.status != "paused":
				raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机当前不在暂停状态，无法继续")
			self.status = "running"
			self.paused_at = None
			self.expected_finish_at = current_time + self.remaining_seconds
			return [{
				"type": "washing_resumed",
				"source": self.device_id,
				"payload": {
					"remaining_seconds": self.remaining_seconds,
					"expected_finish_at": _format_timestamp(self.expected_finish_at),
				},
			}]
		if command == "cancel":
			if self.status not in {"running", "paused", "completed"}:
				raise ProtocolError(ERROR_INVALID_PARAM, "洗衣机当前没有可取消的任务")
			self.status = "cancelled"
			self.remaining_seconds = 0
			self.expected_finish_at = None
			self.paused_at = None
			return [{
				"type": "washing_cancelled",
				"source": self.device_id,
				"payload": {"program": self.program},
			}]
		raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"洗衣机不支持命令 {command}")

	def sync_time(self, current_time: float) -> list[dict[str, Any]]:
		if self.status != "running":
			return []
		self._refresh_remaining(current_time)
		if self.expected_finish_at is not None and current_time >= self.expected_finish_at:
			self.status = "completed"
			self.remaining_seconds = 0
			self.completed_at = self.expected_finish_at
			self.expected_finish_at = None
			self.paused_at = None
			return [{
				"type": "washing_completed",
				"source": self.device_id,
				"payload": {
					"program": self.program,
					"completed_at": _format_timestamp(self.completed_at),
				},
			}]
		return []

	def snapshot(self) -> dict[str, Any]:
		data = super().snapshot()
		data.update({
			"status": self.status,
			"program": self.program,
			"duration_seconds": self.duration_seconds,
			"remaining_seconds": self.remaining_seconds,
			"started_at": _format_timestamp(self.started_at),
			"expected_finish_at": _format_timestamp(self.expected_finish_at),
			"paused_at": _format_timestamp(self.paused_at),
			"completed_at": _format_timestamp(self.completed_at),
		})
		return data

	def _refresh_remaining(self, current_time: float) -> None:
		if self.status != "running" or self.expected_finish_at is None:
			return
		remaining = max(0, ceil(self.expected_finish_at - current_time))
		self.remaining_seconds = remaining


def _require_number(params: dict[str, Any], key: str) -> float:
	value = params.get(key)
	if not isinstance(value, (int, float)):
		raise ProtocolError(ERROR_INVALID_PARAM, f"{key} 必须是数值")
	return float(value)


def _require_positive_number(params: dict[str, Any], key: str, default: float | None = None) -> float:
	if key not in params:
		if default is None:
			raise ProtocolError(ERROR_INVALID_PARAM, f"{key} 必须是正数")
		return default
	value = _require_number(params, key)
	if value <= 0:
		raise ProtocolError(ERROR_INVALID_PARAM, f"{key} 必须大于 0")
	return value


def _format_timestamp(timestamp: float | None) -> str | None:
	if timestamp is None:
		return None
	return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
