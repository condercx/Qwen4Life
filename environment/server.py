"""智能家居环境的独立 FastAPI 服务。"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from environment.smart_home_env import SmartHomeEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ENV SERVER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Home Environment Emulator")
env = SmartHomeEnv()
REQUEST_COUNTER = itertools.count(1)


class ActionRequest(BaseModel):
    """环境服务动作请求模型。"""

    action: dict[str, Any]
    intent: str | None = None
    request_id: str | None = Field(default=None)


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, Any]:
    """初始化或重置会话。"""

    logger.info("重置会话：%s", session_id)
    return env.reset(session_id)


@app.get("/session/{session_id}/state")
def get_state(session_id: str) -> dict[str, Any]:
    """获取当前设备状态。"""

    logger.info("查询会话状态：%s", session_id)
    return {"state": env.get_state(session_id)}


@app.get("/session/{session_id}/events")
def get_events(session_id: str) -> dict[str, Any]:
    """获取当前未读事件。"""

    logger.info("查询会话事件：%s", session_id)
    return {"events": env.get_events(session_id)}


@app.post("/session/{session_id}/action")
def execute_action(session_id: str, request: ActionRequest) -> dict[str, Any]:
    """执行一次环境动作。"""

    request_id = request.request_id or f"{session_id}-req-{next(REQUEST_COUNTER)}"
    action_info = _normalize_action_payload(request.action)
    logger.info(
        "收到动作：command=%s, target=%s, params=%s, request_id=%s",
        action_info["command"],
        action_info["target"],
        action_info["params"],
        request_id,
    )

    payload = {
        "request_id": request_id,
        "session_id": session_id,
        "intent": request.intent,
        "action": action_info,
    }
    try:
        result = env.step(payload)
        logger.info("动作执行完成：success=%s, request_id=%s", result.get("success"), request_id)
        return result
    except Exception as exc:
        logger.exception("动作执行异常：request_id=%s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _normalize_action_payload(action: dict[str, Any]) -> dict[str, Any]:
    """把 HTTP 层动作请求归一化为环境 step 协议需要的结构。"""

    return {
        "device": str(action.get("device", "unknown")),
        "target": str(action.get("target", "unknown")),
        "command": str(action.get("command", action.get("name", "unknown"))),
        "params": action.get("params", action.get("args", {})),
    }


if __name__ == "__main__":
    logger.info("启动智能家居环境服务。")
    uvicorn.run("environment.server:app", host="0.0.0.0", port=6666, reload=False)
