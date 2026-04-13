"""智能家居控制模拟环境。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from environment.actions import (
	ERROR_DEVICE_NOT_FOUND,
	ERROR_INVALID_PARAM,
	ERROR_SESSION_NOT_FOUND,
	ProtocolError,
	build_error_response,
	build_success_response,
	parse_step_request,
)
from environment.devices import Device, Room
from environment.scenarios import build_default_devices, build_default_room


@dataclass(slots=True)
class SessionState:
	"""会话级环境状态。"""

	session_id: str
	room: Room
	devices: dict[str, Device]
	sim_time: int = 0
	last_user_intent: str | None = None
	state_cache: dict[str, Any] = field(default_factory=dict)
	history: list[dict[str, Any]] = field(default_factory=list)
	unread_events: list[dict[str, Any]] = field(default_factory=list)
	event_counter: int = 0

	def emit_event(self, event_type: str, source: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
		self.event_counter += 1
		event = {
			"event_id": f"{self.session_id}-evt-{self.event_counter}",
			"sim_time": self.sim_time,
			"type": event_type,
			"source": source,
			"payload": payload,
		}
		self.unread_events.append(event)
		return [event]

	def observation(self) -> dict[str, Any]:
		return {
			"sim_time": self.sim_time,
			"room": self.room.snapshot(),
			"devices": {device_id: device.snapshot() for device_id, device in self.devices.items()},
			"last_user_intent": self.last_user_intent,
			"state_cache": self.state_cache,
		}

	def drain_events(self) -> list[dict[str, Any]]:
		events = list(self.unread_events)
		self.unread_events.clear()
		return events


class SmartHomeEnv:
	"""提供 reset/step/get_state/get_events 四个最小接口。"""

	def __init__(self) -> None:
		self.sessions: dict[str, SessionState] = {}

	def reset(self, session_id: str) -> dict[str, Any]:
		"""重置指定会话，返回初始状态。"""

		room = build_default_room()
		devices = build_default_devices()
		state = SessionState(session_id=session_id, room=room, devices=devices)
		self.sessions[session_id] = state
		events = state.emit_event("session_reset", "system", {"session_id": session_id})
		state.state_cache["last_request_id"] = None
		state.history.append({"type": "reset", "session_id": session_id})
		return state.observation()

	def step(self, request: dict[str, Any]) -> dict[str, Any]:
		"""执行一次环境交互，并推进模拟时间。"""

		request_id = request.get("request_id", "unknown")
		session_id = request.get("session_id", "unknown")
		try:
			parsed = parse_step_request(request)
			state = self.sessions.get(parsed.session_id) or self._create_session(parsed.session_id)
			state.last_user_intent = parsed.intent
			state.state_cache["last_request_id"] = parsed.request_id

			generated_events: list[dict[str, Any]] = []
			if parsed.action.device == "system":
				generated_events.extend(self._handle_system_action(state, parsed.action.command, parsed.action.params))
			else:
				device = self._get_device(state, parsed.action.target)
				generated_events.extend(
					self._dispatch_device_action(state, device, parsed.action.mode, parsed.action.command, parsed.action.params)
				)

			advance_ticks = int(parsed.options.get("advance_ticks", 1))
			if advance_ticks < 0:
				raise ProtocolError(ERROR_INVALID_PARAM, "advance_ticks 不能小于 0")
			generated_events.extend(self._advance_time(state, advance_ticks))

			state.history.append({
				"request_id": parsed.request_id,
				"sim_time": state.sim_time,
				"intent": parsed.intent,
				"action": {
					"mode": parsed.action.mode,
					"device": parsed.action.device,
					"target": parsed.action.target,
					"command": parsed.action.command,
					"params": parsed.action.params,
				},
			})

			metrics = {
				"device_count": len(state.devices),
				"history_length": len(state.history),
				"unread_event_count": len(state.unread_events),
			}
			return build_success_response(
				parsed.request_id,
				parsed.session_id,
				state.observation(),
				generated_events,
				metrics=metrics,
			)
		except ProtocolError as error:
			observation = {}
			if session_id in self.sessions:
				observation = self.sessions[session_id].observation()
			return build_error_response(request_id, session_id, error, observation)

	def get_state(self, session_id: str) -> dict[str, Any]:
		"""读取当前状态快照。"""

		state = self._require_session(session_id)
		return state.observation()

	def get_events(self, session_id: str) -> list[dict[str, Any]]:
		"""轮询未读事件。"""

		state = self._require_session(session_id)
		return state.drain_events()

	def _create_session(self, session_id: str) -> SessionState:
		self.reset(session_id)
		return self.sessions[session_id]

	def _require_session(self, session_id: str) -> SessionState:
		state = self.sessions.get(session_id)
		if state is None:
			raise ProtocolError(ERROR_SESSION_NOT_FOUND, f"session {session_id} 不存在")
		return state

	def _get_device(self, state: SessionState, target: str) -> Device:
		device = state.devices.get(target)
		if device is None:
			raise ProtocolError(ERROR_DEVICE_NOT_FOUND, f"找不到设备 {target}")
		return device

	def _dispatch_device_action(
		self,
		state: SessionState,
		device: Device,
		mode: str,
		command: str,
		params: dict[str, Any],
	) -> list[dict[str, Any]]:
		device_events = device.handle_discrete(command, params) if mode == "discrete" else device.handle_continuous(command, params, state.room)
		return [state.emit_event(raw["type"], raw["source"], raw["payload"])[0] for raw in device_events]

	def _handle_system_action(self, state: SessionState, command: str, params: dict[str, Any]) -> list[dict[str, Any]]:
		if command != "advance":
			raise ProtocolError(ERROR_INVALID_PARAM, f"system 不支持命令 {command}")
		ticks = int(params.get("ticks", 1))
		if ticks <= 0:
			raise ProtocolError(ERROR_INVALID_PARAM, "ticks 必须大于 0")
		return self._advance_time(state, ticks)

	def _advance_time(self, state: SessionState, ticks: int) -> list[dict[str, Any]]:
		raw_events: list[dict[str, Any]] = []
		for _ in range(ticks):
			state.sim_time += 1
			for device in state.devices.values():
				for raw in device.advance(state.room):
					raw_events.append(state.emit_event(raw["type"], raw["source"], raw["payload"])[0])
		return raw_events
