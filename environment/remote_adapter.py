"""供 Agent 调用的环境 HTTP 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class RemoteEnvironmentAdapter:
    """通过 HTTP 与独立环境服务通信。"""

    server_url: str = "http://localhost:6666"
    timeout: int = 10

    def create_session(self, session_id: str) -> dict[str, Any]:
        """初始化指定会话。"""

        return self._post(f"/session/{session_id}/reset")

    def send_action(
        self,
        session_id: str,
        action: dict[str, Any],
        intent: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """向环境发送动作请求。"""

        payload = {
            "action": action,
            "intent": intent,
            "request_id": request_id,
        }
        return self._post(f"/session/{session_id}/action", payload)

    def fetch_state(self, session_id: str) -> dict[str, Any]:
        """获取会话当前状态。"""

        response = self._get(f"/session/{session_id}/state")
        return response.get("state", {})

    def fetch_events(self, session_id: str) -> list[dict[str, Any]]:
        """获取会话当前未读事件。"""

        response = self._get(f"/session/{session_id}/events")
        return response.get("events", [])

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """发送 POST 请求并返回 JSON 响应。"""

        url = self._build_url(path)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload or {})
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(f"无法连接环境服务：{url}。") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"环境服务返回错误：{url}，HTTP {exc.response.status_code}。"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"环境服务请求失败：{url}。") from exc

    def _get(self, path: str) -> dict[str, Any]:
        """发送 GET 请求并返回 JSON 响应。"""

        url = self._build_url(path)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(f"无法连接环境服务：{url}。") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"环境服务返回错误：{url}，HTTP {exc.response.status_code}。"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"环境服务请求失败：{url}。") from exc

    def _build_url(self, path: str) -> str:
        """拼接完整 URL。"""

        return f"{self.server_url.rstrip('/')}/{path.lstrip('/')}"
