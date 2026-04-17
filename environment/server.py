"""智能家居环境的独立服务进程 (FastAPI)。"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from environment.smart_home_env import SmartHomeEnv

# 配置日志追踪
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ENV SERVER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Home Environment Emulator")
env = SmartHomeEnv()

# 线程安全的请求计数器（CPython 下 next() 受 GIL 保护）
_request_counter = itertools.count(1)


class ActionRequest(BaseModel):
    action: dict[str, Any]
    intent: str | None = None
    request_id: str | None = None


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, Any]:
    """初始化或重置环境会话。"""
    logger.info(f"重置会话: {session_id}")
    return env.reset(session_id)


@app.get("/session/{session_id}/state")
def get_state(session_id: str) -> dict[str, Any]:
    """获取当前全部设备状态。"""
    logger.info(f"查询会话状态: {session_id}")
    return {"state": env.get_state(session_id)}


@app.get("/session/{session_id}/events")
def get_events(session_id: str) -> dict[str, Any]:
    """获取会话未消费事件。"""
    return {"events": env.get_events(session_id)}


@app.post("/session/{session_id}/action")
def execute_action(session_id: str, req: ActionRequest) -> dict[str, Any]:
    """执行动作调用控制设备/查询等。"""
    counter = next(_request_counter)
    req_id = req.request_id or f"{session_id}-req-{counter}"

    action_info = req.action
    # ReAct agent sends 'command' and 'target' via tool execute
    command = action_info.get("command", action_info.get("name", "unknown"))
    target = action_info.get("target", "unknown")
    params = action_info.get("params", action_info.get("args", {}))

    logger.info(f"收到动作 -> {command} (目标: {target}, 参数: {params}) [req_id={req_id}]")

    # 构造兼容原生 SmartHomeEnv 格式的 payload
    payload = {
        "request_id": req_id,
        "session_id": session_id,
        "intent": req.intent,
        "action": action_info,
    }

    try:
        result = env.step(payload)
        logger.info(f"动作执行结果 -> {result.get('action_status')}")
        return result
    except Exception as e:
        logger.error(f"动作执行异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("启动智能家居独立环境服务...")
    uvicorn.run("environment.server:app", host="0.0.0.0", port=6666, reload=False)
