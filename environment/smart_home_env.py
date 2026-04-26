"""智能家居模拟环境核心实现。"""

from __future__ import annotations

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
from environment.clock import Clock, SystemClock
from environment.devices import Device
from environment.scenarios import DeviceFactory, build_default_devices


@dataclass(slots=True)
class SessionState:
    """保存单个会话的环境状态。"""

    session_id: str
    devices: dict[str, Device]
    created_at: float
    last_user_intent: str | None = None
    state_cache: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    unread_events: list[dict[str, Any]] = field(default_factory=list)
    event_counter: int = 0

    def emit_event(
        self,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        current_time: float,
    ) -> dict[str, Any]:
        """写入一条事件并返回标准事件对象。"""

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
        """构造当前会话的观测快照。"""

        return {
            "observed_at": _format_timestamp(current_time),
            "devices": {device_id: device.snapshot() for device_id, device in self.devices.items()},
            "last_user_intent": self.last_user_intent,
            "state_cache": self.state_cache,
        }

    def drain_events(self) -> list[dict[str, Any]]:
        """读取并清空未读事件。"""

        events = list(self.unread_events)
        self.unread_events.clear()
        return events


class SmartHomeEnv:
    """提供 `reset`、`step`、`get_state` 和 `get_events` 四个基础接口。"""

    def __init__(
        self,
        clock: Clock | None = None,
        device_factory: DeviceFactory = build_default_devices,
    ) -> None:
        # 注入时间源后，带计时逻辑的设备状态流转可以在测试中保持确定性。
        self.clock = clock or SystemClock()
        # 设备工厂必须返回新设备实例，避免不同会话共享可变状态。
        self.device_factory = device_factory
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str) -> dict[str, Any]:
        """重置会话并返回初始观测。"""

        normalized_session_id = _normalize_session_id(session_id)
        current_time = self.clock.now()
        state = SessionState(
            session_id=normalized_session_id,
            devices=self.device_factory(),
            created_at=current_time,
        )
        self.sessions[normalized_session_id] = state
        state.emit_event("session_reset", "system", {"session_id": normalized_session_id}, current_time)
        state.state_cache["last_request_id"] = None
        state.state_cache["last_synced_at"] = _format_timestamp(current_time)
        state.history.append({"type": "reset", "session_id": normalized_session_id})
        return state.observation(current_time)

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        """执行一次环境交互。"""

        request_id = str(request.get("request_id", "unknown"))
        session_id = str(request.get("session_id", "unknown"))

        try:
            parsed = parse_step_request(request)
            state = self.sessions.get(parsed.session_id) or self._create_session(parsed.session_id)
            current_time = self.clock.now()

            generated_events = self._sync_timed_devices(state, current_time)
            state.last_user_intent = parsed.intent
            state.state_cache["last_request_id"] = parsed.request_id

            device = self._get_device(state, parsed.action.target)
            generated_events.extend(
                self._dispatch_device_action(
                    state=state,
                    device=device,
                    command=parsed.action.command,
                    params=parsed.action.params,
                    current_time=current_time,
                )
            )

            state.history.append(
                {
                    "request_id": parsed.request_id,
                    "observed_at": _format_timestamp(current_time),
                    "intent": parsed.intent,
                    "action": {
                        "device": parsed.action.device,
                        "target": parsed.action.target,
                        "command": parsed.action.command,
                        "params": parsed.action.params,
                    },
                }
            )

            metrics = {
                "device_count": len(state.devices),
                "history_length": len(state.history),
                "unread_event_count": len(state.unread_events),
            }
            return build_success_response(
                request_id=parsed.request_id,
                session_id=parsed.session_id,
                observation=state.observation(current_time),
                events=generated_events,
                metrics=metrics,
            )
        except ProtocolError as error:
            observation: dict[str, Any] = {}
            if session_id in self.sessions:
                current_time = self.clock.now()
                state = self.sessions[session_id]
                self._sync_timed_devices(state, current_time)
                observation = state.observation(current_time)
            return build_error_response(request_id, session_id, error, observation)

    def get_state(self, session_id: str) -> dict[str, Any]:
        """获取当前会话状态。"""

        state = self._require_session(session_id)
        current_time = self.clock.now()
        self._sync_timed_devices(state, current_time)
        return state.observation(current_time)

    def get_events(self, session_id: str) -> list[dict[str, Any]]:
        """获取并清空当前会话的未读事件。"""

        state = self._require_session(session_id)
        current_time = self.clock.now()
        self._sync_timed_devices(state, current_time)
        return state.drain_events()

    def _create_session(self, session_id: str) -> SessionState:
        """按需创建缺失会话。"""

        self.reset(session_id)
        return self.sessions[session_id]

    def _require_session(self, session_id: str) -> SessionState:
        """获取已存在的会话，不存在时抛出协议异常。"""

        normalized_session_id = _normalize_session_id(session_id)
        state = self.sessions.get(normalized_session_id)
        if state is None:
            raise ProtocolError(ERROR_SESSION_NOT_FOUND, f"会话 `{normalized_session_id}` 不存在。")
        return state

    @staticmethod
    def _get_device(state: SessionState, target: str) -> Device:
        """获取目标设备。"""

        device = state.devices.get(target)
        if device is None:
            raise ProtocolError(ERROR_DEVICE_NOT_FOUND, f"找不到设备 `{target}`。")
        return device

    @staticmethod
    def _dispatch_device_action(
        state: SessionState,
        device: Device,
        command: str,
        params: dict[str, Any],
        current_time: float,
    ) -> list[dict[str, Any]]:
        """执行设备动作并转换为标准事件列表。"""

        raw_events = device.handle_command(command, params, current_time)
        return [
            state.emit_event(
                event_type=raw_event["type"],
                source=raw_event["source"],
                payload=raw_event["payload"],
                current_time=current_time,
            )
            for raw_event in raw_events
        ]

    @staticmethod
    def _sync_timed_devices(state: SessionState, current_time: float) -> list[dict[str, Any]]:
        """推进所有设备的时间相关状态。"""

        generated_events: list[dict[str, Any]] = []
        for device in state.devices.values():
            for raw_event in device.sync_time(current_time):
                generated_events.append(
                    state.emit_event(
                        event_type=raw_event["type"],
                        source=raw_event["source"],
                        payload=raw_event["payload"],
                        current_time=current_time,
                    )
                )
        state.state_cache["last_synced_at"] = _format_timestamp(current_time)
        return generated_events


def _normalize_session_id(session_id: str) -> str:
    """归一化并校验会话 ID。"""

    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        raise ProtocolError(ERROR_SESSION_NOT_FOUND, "会话 ID 不能为空。")
    return normalized_session_id


def _format_timestamp(timestamp: float) -> str:
    """将时间戳格式化为 ISO 字符串。"""

    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
