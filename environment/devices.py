"""设备模型与基础物理规则。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from environment.actions import (
	ERROR_DEVICE_OFFLINE,
	ERROR_INVALID_PARAM,
	ERROR_TARGET_UNREACHABLE,
	ERROR_UNSUPPORTED_COMMAND,
	ProtocolError,
)


@dataclass(slots=True)
class Room:
	"""房间布局与简单障碍定义。"""

	room_id: str
	name: str
	width: float
	height: float
	blocked_zones: list[dict[str, float]] = field(default_factory=list)

	def contains(self, x: float, y: float) -> bool:
		return 0.0 <= x <= self.width and 0.0 <= y <= self.height

	def is_blocked(self, x: float, y: float) -> bool:
		for zone in self.blocked_zones:
			if zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]:
				return True
		return False

	def snapshot(self) -> dict[str, Any]:
		return {
			"room_id": self.room_id,
			"name": self.name,
			"width": self.width,
			"height": self.height,
			"blocked_zones": self.blocked_zones,
		}


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

	def handle_discrete(self, command: str, params: dict[str, Any]) -> list[dict[str, Any]]:
		raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"{self.device_type} 不支持离散命令 {command}")

	def handle_continuous(self, command: str, params: dict[str, Any], room: Room) -> list[dict[str, Any]]:
		raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"{self.device_type} 不支持连续命令 {command}")

	def advance(self, room: Room) -> list[dict[str, Any]]:
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

	def handle_discrete(self, command: str, params: dict[str, Any]) -> list[dict[str, Any]]:
		self.ensure_online()
		if command == "turn_on":
			self.is_on = True
			self.brightness = max(self.brightness, 100)
		elif command == "turn_off":
			self.is_on = False
			self.brightness = 0
		else:
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"灯光不支持命令 {command}")

		return [{
			"type": "light_state_changed",
			"source": self.device_id,
			"payload": {"is_on": self.is_on, "brightness": self.brightness},
		}]

	def handle_continuous(self, command: str, params: dict[str, Any], room: Room) -> list[dict[str, Any]]:
		self.ensure_online()
		if command != "set_brightness":
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"灯光不支持连续命令 {command}")

		brightness = int(_require_number(params, "brightness"))
		if brightness < 0 or brightness > 100:
			raise ProtocolError(ERROR_INVALID_PARAM, "brightness 必须位于 0-100")

		self.brightness = brightness
		self.is_on = brightness > 0
		return [{
			"type": "light_brightness_changed",
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

	def handle_discrete(self, command: str, params: dict[str, Any]) -> list[dict[str, Any]]:
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
		else:
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"空调不支持命令 {command}")

		return [{
			"type": "ac_state_changed",
			"source": self.device_id,
			"payload": {"is_on": self.is_on, "mode": self.mode},
		}]

	def handle_continuous(self, command: str, params: dict[str, Any], room: Room) -> list[dict[str, Any]]:
		self.ensure_online()
		if command == "set_temperature":
			temperature = round(_require_number(params, "temperature"), 1)
			if temperature < 16.0 or temperature > 30.0:
				raise ProtocolError(ERROR_INVALID_PARAM, "temperature 必须位于 16-30")
			self.target_temperature = temperature
		elif command == "set_fan_speed":
			fan_speed = round(_require_number(params, "fan_speed"), 2)
			if fan_speed < 0.1 or fan_speed > 5.0:
				raise ProtocolError(ERROR_INVALID_PARAM, "fan_speed 必须位于 0.1-5.0")
			self.fan_speed = fan_speed
		else:
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"空调不支持连续命令 {command}")

		self.is_on = True
		return [{
			"type": "ac_setting_changed",
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
class RobotVacuum(Device):
	"""扫地机器人，支持目标点移动与简单避障。"""

	position_x: float = 0.0
	position_y: float = 0.0
	default_speed: float = 0.5
	status: str = "idle"
	target_x: float | None = None
	target_y: float | None = None
	current_speed: float = 0.5

	def handle_discrete(self, command: str, params: dict[str, Any]) -> list[dict[str, Any]]:
		self.ensure_online()
		if command == "start_cleaning":
			self.status = "cleaning"
		elif command == "stop":
			self.status = "idle"
			self.target_x = None
			self.target_y = None
		elif command == "dock":
			self.status = "returning"
			self.target_x = 0.0
			self.target_y = 0.0
		else:
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"扫地机器人不支持命令 {command}")

		return [{
			"type": "robot_state_changed",
			"source": self.device_id,
			"payload": {"status": self.status},
		}]

	def handle_continuous(self, command: str, params: dict[str, Any], room: Room) -> list[dict[str, Any]]:
		self.ensure_online()
		if command != "move_to":
			raise ProtocolError(ERROR_UNSUPPORTED_COMMAND, f"扫地机器人不支持连续命令 {command}")

		target_x = _require_number(params, "x")
		target_y = _require_number(params, "y")
		speed = _require_number(params, "speed") if "speed" in params else self.default_speed
		if speed <= 0:
			raise ProtocolError(ERROR_INVALID_PARAM, "speed 必须大于 0")
		if not room.contains(target_x, target_y):
			raise ProtocolError(ERROR_TARGET_UNREACHABLE, "目标点超出房间边界")
		if room.is_blocked(target_x, target_y):
			raise ProtocolError(ERROR_TARGET_UNREACHABLE, "目标点位于障碍区")

		self.target_x = target_x
		self.target_y = target_y
		self.current_speed = speed
		self.status = "moving"
		return [{
			"type": "robot_target_updated",
			"source": self.device_id,
			"payload": {"target_x": self.target_x, "target_y": self.target_y, "speed": self.current_speed},
		}]

	def advance(self, room: Room) -> list[dict[str, Any]]:
		if self.target_x is None or self.target_y is None:
			return []

		distance = hypot(self.target_x - self.position_x, self.target_y - self.position_y)
		if distance <= self.current_speed:
			self.position_x = self.target_x
			self.position_y = self.target_y
			self.target_x = None
			self.target_y = None
			self.status = "idle"
			return [{
				"type": "robot_arrived",
				"source": self.device_id,
				"payload": {"x": self.position_x, "y": self.position_y},
			}]

		ratio = self.current_speed / distance
		next_x = self.position_x + (self.target_x - self.position_x) * ratio
		next_y = self.position_y + (self.target_y - self.position_y) * ratio

		if not room.contains(next_x, next_y):
			self.status = "blocked"
			return [{
				"type": "robot_boundary_blocked",
				"source": self.device_id,
				"payload": {"x": next_x, "y": next_y},
			}]
		if room.is_blocked(next_x, next_y):
			self.status = "blocked"
			return [{
				"type": "robot_obstacle_detected",
				"source": self.device_id,
				"payload": {"x": next_x, "y": next_y},
			}]

		self.position_x = next_x
		self.position_y = next_y
		return [{
			"type": "robot_position_updated",
			"source": self.device_id,
			"payload": {"x": self.position_x, "y": self.position_y, "status": self.status},
		}]

	def snapshot(self) -> dict[str, Any]:
		data = super().snapshot()
		data.update({
			"position": {"x": self.position_x, "y": self.position_y},
			"target": None if self.target_x is None else {"x": self.target_x, "y": self.target_y},
			"status": self.status,
			"current_speed": self.current_speed,
		})
		return data


def _require_number(params: dict[str, Any], key: str) -> float:
	value = params.get(key)
	if not isinstance(value, (int, float)):
		raise ProtocolError(ERROR_INVALID_PARAM, f"{key} 必须是数值")
	return float(value)
