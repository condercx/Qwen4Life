"""环境动作协议定义与请求校验工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ERROR_INVALID_PARAM = 1001
ERROR_DEVICE_OFFLINE = 1002
ERROR_UNSUPPORTED_COMMAND = 1003
ERROR_SESSION_NOT_FOUND = 1005
ERROR_DEVICE_NOT_FOUND = 1006
ERROR_INVALID_REQUEST = 1007


@dataclass(slots=True)
class Action:
    """统一描述一次设备动作。"""

    device: str
    target: str
    command: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepRequest:
    """环境 `step` 接口的标准输入结构。"""

    request_id: str
    session_id: str
    intent: str | None
    action: Action
    timestamp: str | None = None


class ProtocolError(ValueError):
    """协议层统一异常，便于构造标准错误响应。"""

    def __init__(self, code: int, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def parse_step_request(payload: dict[str, Any]) -> StepRequest:
    """校验并解析环境动作请求。"""

    if not isinstance(payload, dict):
        raise ProtocolError(ERROR_INVALID_REQUEST, "请求体必须是字典。")

    request_id = _require_non_empty_string(payload, "request_id")
    session_id = _require_non_empty_string(payload, "session_id")
    action_payload = payload.get("action")
    if not isinstance(action_payload, dict):
        raise ProtocolError(ERROR_INVALID_REQUEST, "`action` 字段缺失或格式错误。")

    action = Action(
        device=_require_non_empty_string(action_payload, "device"),
        target=_require_non_empty_string(action_payload, "target"),
        command=_require_non_empty_string(action_payload, "command"),
        params=_optional_dict(action_payload.get("params")),
    )

    intent = payload.get("intent")
    if intent is not None and not isinstance(intent, str):
        raise ProtocolError(ERROR_INVALID_REQUEST, "`intent` 必须是字符串或 null。")

    timestamp = payload.get("timestamp")
    if timestamp is not None and not isinstance(timestamp, str):
        raise ProtocolError(ERROR_INVALID_REQUEST, "`timestamp` 必须是字符串或 null。")

    return StepRequest(
        request_id=request_id,
        session_id=session_id,
        intent=intent,
        action=action,
        timestamp=timestamp,
    )


def build_success_response(
    request_id: str,
    session_id: str,
    observation: dict[str, Any],
    events: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
    done: bool = False,
    reward: float = 0.0,
) -> dict[str, Any]:
    """构造统一的成功响应。"""

    return {
        "request_id": request_id,
        "session_id": session_id,
        "success": True,
        "observation": observation,
        "events": events,
        "metrics": metrics or {},
        "done": done,
        "reward": reward,
        "error": None,
    }


def build_error_response(
    request_id: str,
    session_id: str,
    error: ProtocolError,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一的错误响应。"""

    return {
        "request_id": request_id,
        "session_id": session_id,
        "success": False,
        "observation": observation or {},
        "events": [],
        "metrics": {},
        "done": False,
        "reward": 0.0,
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
    }


def _require_non_empty_string(payload: dict[str, Any], key: str) -> str:
    """读取并校验非空字符串字段。"""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(ERROR_INVALID_REQUEST, f"`{key}` 字段缺失或不是非空字符串。")
    return value.strip()


def _optional_dict(value: Any) -> dict[str, Any]:
    """读取可选字典字段。"""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolError(ERROR_INVALID_REQUEST, "`params` 必须是字典。")
    return value
