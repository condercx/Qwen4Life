"""环境适配器接口和内存实现。"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from environment.smart_home_env import SmartHomeEnv


class EnvironmentAdapter(Protocol):
    """Agent 工具访问环境时依赖的接口。"""

    def create_session(self, session_id: str) -> dict[str, Any]:
        """初始化或重置指定会话。"""
        ...

    def send_action(
        self,
        session_id: str,
        action: dict[str, Any],
        intent: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """向目标环境发送一次动作。"""
        ...

    def fetch_state(self, session_id: str) -> dict[str, Any]:
        """返回当前会话观测。"""
        ...

    def fetch_events(self, session_id: str) -> list[dict[str, Any]]:
        """返回并清空当前会话的未读事件。"""
        ...


@dataclass(slots=True)
class InMemoryEnvironmentAdapter:
    """单元测试使用的内存适配器，直接调用 SmartHomeEnv。"""

    env: SmartHomeEnv = field(default_factory=SmartHomeEnv)
    _request_counter: Iterator[int] = field(default_factory=lambda: itertools.count(1), init=False)

    def create_session(self, session_id: str) -> dict[str, Any]:
        """在本地环境中初始化或重置会话。"""

        return self.env.reset(session_id)

    def send_action(
        self,
        session_id: str,
        action: dict[str, Any],
        intent: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """不经过 HTTP 边界，直接执行一次动作。"""

        # 与 environment.server 的请求 ID 生成方式保持一致，避免测试和生产语义分叉。
        resolved_request_id = request_id or f"{session_id}-req-{next(self._request_counter)}"
        return self.env.step(
            {
                "request_id": resolved_request_id,
                "session_id": session_id,
                "intent": intent,
                "action": action,
            }
        )

    def fetch_state(self, session_id: str) -> dict[str, Any]:
        """返回当前会话观测。"""

        return self.env.get_state(session_id)

    def fetch_events(self, session_id: str) -> list[dict[str, Any]]:
        """返回并清空当前会话的未读事件。"""

        return self.env.get_events(session_id)
