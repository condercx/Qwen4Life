"""面向 agent 的远程环境适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class RemoteEnvironmentAdapter:
    """通过 HTTP 连接负责调用独立的 environment server。"""

    server_url: str = "http://localhost:6666"
    timeout: int = 10

    def _post(self, path: str, payload: dict | None = None) -> dict[str, Any]:
        """封装 HTTP POST 请求。"""
        url = f"{self.server_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload or {})
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as e:
            raise RuntimeError(f"无法连接到环境服务({url}): {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"环境服务返回错误({url}): HTTP {e.response.status_code}") from e

    def _get(self, path: str) -> dict[str, Any]:
        """封装 HTTP GET 请求。"""
        url = f"{self.server_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as e:
            raise RuntimeError(f"无法连接到环境服务({url}): {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"环境服务返回错误({url}): HTTP {e.response.status_code}") from e

    def create_session(self, session_id: str) -> dict[str, Any]:
        """初始化会话。"""
        return self._post(f"/session/{session_id}/reset")

    def send_action(
        self,
        session_id: str,
        action: dict[str, Any],
        intent: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """使用语义动作驱动环境。"""
        payload = {
            "action": action,
            "intent": intent,
            "request_id": request_id
        }
        return self._post(f"/session/{session_id}/action", payload)

    def fetch_state(self, session_id: str) -> dict[str, Any]:
        """读取当前会话状态。"""
        resp = self._get(f"/session/{session_id}/state")
        return resp.get("state", {})

    def fetch_events(self, session_id: str) -> list[dict[str, Any]]:
        """读取当前会话未消费事件。"""
        resp = self._get(f"/session/{session_id}/events")
        return resp.get("events", [])
