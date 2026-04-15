"""智能家居控制模拟环境。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from environment.actions import (
	ERROR_DEVICE_NOT_FOUND,
	ERROR_SESSION_NOT_FOUND,
	ProtocolError,
	build_error_response,
	build_success_response,
	parse_step_request,
)
from environment.devices import Device
from environment.scenarios import build_default_devices


@dataclass(slots=True)
class SessionState:
	"""会话级环境状态。"""

	session_id: str
	devices: dict[str, Device]
	created_at: float = field(default_factory=time.time)
	last_user_intent: str | None = None
	state_cache: dict[str, Any] = field(default_factory=dict)
	history: list[dict[str, Any]] = field(default_factory=list)
	unread_events: list[dict[str, Any]] = field(default_factory=list)
	event_counter: int = 0

	def emit_event(self, event_type: str, source: str, payload: dict[str, Any], current_time: float) -> dict[str, Any]:
		self.event_counter += 1
		event = {
			"event_id": f"{self.session_id}-evt-{self.event_counter}",
			"occurred_at": _format_timestamp(current_time),
			"type": event_type,
			"source": source,
			"payload": payload,
		}
		self.unread_events.append(event)
		return event

	def observation(self, current_time: float) -> dict[str, Any]:
		return {
			"observed_at": _format_timestamp(current_time),
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

		current_time = time.time()
		devices = build_default_devices()
		state = SessionState(session_id=session_id, devices=devices, created_at=current_time)
		self.sessions[session_id] = state
		state.emit_event("session_reset", "system", {"session_id": session_id}, current_time)
		state.state_cache["last_request_id"] = None
		state.state_cache["last_synced_at"] = _format_timestamp(current_time)
		state.history.append({"type": "reset", "session_id": session_id})
		return state.observation(current_time)

	def step(self, request: dict[str, Any]) -> dict[str, Any]:
		"""执行一次环境交互，并同步后台计时任务。"""

		request_id = request.get("request_id", "unknown")
		session_id = request.get("session_id", "unknown")
		try:
			parsed = parse_step_request(request)
			state = self.sessions.get(parsed.session_id) or self._create_session(parsed.session_id)
			current_time = time.time()
			generated_events = self._sync_timed_devices(state, current_time)
			state.last_user_intent = parsed.intent
			state.state_cache["last_request_id"] = parsed.request_id
			device = self._get_device(state, parsed.action.target)
			generated_events.extend(self._dispatch_device_action(state, device, parsed.action.command, parsed.action.params, current_time))

			state.history.append({
				"request_id": parsed.request_id,
				"observed_at": _format_timestamp(current_time),
				"intent": parsed.intent,
				"action": {
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
				state.observation(current_time),
				generated_events,
				metrics=metrics,
			)
		except ProtocolError as error:
			observation = {}
			if session_id in self.sessions:
				current_time = time.time()
				state = self.sessions[session_id]
				self._sync_timed_devices(state, current_time)
				observation = state.observation(current_time)
			return build_error_response(request_id, session_id, error, observation)

	def get_state(self, session_id: str) -> dict[str, Any]:
		"""读取当前状态快照。"""

		state = self._require_session(session_id)
		current_time = time.time()
		self._sync_timed_devices(state, current_time)
		return state.observation(current_time)

	def get_events(self, session_id: str) -> list[dict[str, Any]]:
		"""轮询未读事件。"""

		state = self._require_session(session_id)
		current_time = time.time()
		self._sync_timed_devices(state, current_time)
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
		command: str,
		params: dict[str, Any],
		current_time: float,
	) -> list[dict[str, Any]]:
		device_events = device.handle_command(command, params, current_time)
		return [state.emit_event(raw["type"], raw["source"], raw["payload"], current_time) for raw in device_events]

	def _sync_timed_devices(self, state: SessionState, current_time: float) -> list[dict[str, Any]]:
		generated_events: list[dict[str, Any]] = []
		for device in state.devices.values():
			for raw in device.sync_time(current_time):
				generated_events.append(state.emit_event(raw["type"], raw["source"], raw["payload"], current_time))
		state.state_cache["last_synced_at"] = _format_timestamp(current_time)
		return generated_events


def _format_timestamp(timestamp: float) -> str:
	return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
