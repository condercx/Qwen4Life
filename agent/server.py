"""Web demo gateway for the smart home Agent."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.controller import SimpleSmartHomeAgent

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    """Browser chat request."""

    message: str = Field(min_length=1)
    verbose: bool = False


class DeviceActionRequest(BaseModel):
    """Manual device action request from the dashboard."""

    device: str
    target: str
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None


@dataclass(slots=True)
class DemoGateway:
    """Thin facade that keeps the frontend decoupled from Agent internals."""

    agent: SimpleSmartHomeAgent | None = None

    def reset_session(self, session_id: str) -> dict[str, Any]:
        """Reset both Agent short context and environment state."""

        self._agent().create_session(session_id)
        return self.fetch_state(session_id)

    def fetch_state(self, session_id: str) -> dict[str, Any]:
        """Return the current environment observation."""

        return self._agent().tools.adapter.fetch_state(session_id)

    def fetch_events(self, session_id: str) -> list[dict[str, Any]]:
        """Return and drain unread environment events."""

        return self._agent().tools.adapter.fetch_events(session_id)

    def execute_action(self, session_id: str, request: DeviceActionRequest) -> dict[str, Any]:
        """Run a manual device action through the same environment adapter."""

        return self._agent().tools.adapter.send_action(
            session_id=session_id,
            action={
                "device": request.device,
                "target": request.target,
                "command": request.command,
                "params": request.params,
            },
            intent=request.intent or f"手动控制 {request.target}: {request.command}",
        )

    def stream_chat(self, session_id: str, request: ChatRequest) -> Iterator[str]:
        """Stream Agent events as server-sent events."""

        try:
            for chunk in self._agent().handle_user_input_stream(session_id, request.message):
                event_type = str(chunk.get("type", "message"))
                content = str(chunk.get("content", ""))
                yield _sse(event_type, {"type": event_type, "content": content})

                if event_type in {"observation", "final_reply"}:
                    yield _sse("state", {"state": self.fetch_state(session_id)})
                if event_type == "observation":
                    events = self.fetch_events(session_id)
                    if events:
                        yield _sse("events", {"events": events})
        except Exception as exc:
            logger.exception("Agent stream failed for session %s", session_id)
            yield _sse("error", {"type": "error", "content": f"Agent 服务异常：{exc}"})
        finally:
            yield _sse("done", {"type": "done"})

    def _agent(self) -> SimpleSmartHomeAgent:
        """Create the production Agent only when the first request needs it."""

        if self.agent is None:
            self.agent = SimpleSmartHomeAgent()
        return self.agent


def create_app(gateway: DemoGateway | None = None) -> FastAPI:
    """Create the web demo FastAPI app."""

    resolved_gateway = gateway or DemoGateway()
    app = FastAPI(title="Qwen4Life Agent Demo")
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        """Serve the dashboard."""

        return FileResponse(FRONTEND_DIR / "index.html")

    @app.post("/api/session/{session_id}/reset")
    def reset_session(session_id: str) -> dict[str, Any]:
        """Reset the demo session."""

        return {"state": resolved_gateway.reset_session(session_id)}

    @app.get("/api/session/{session_id}/state")
    def get_state(session_id: str) -> dict[str, Any]:
        """Return environment state for the dashboard."""

        try:
            return {"state": resolved_gateway.fetch_state(session_id)}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/session/{session_id}/events")
    def get_events(session_id: str) -> dict[str, Any]:
        """Return unread environment events."""

        try:
            return {"events": resolved_gateway.fetch_events(session_id)}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/session/{session_id}/action")
    def execute_action(session_id: str, request: DeviceActionRequest) -> dict[str, Any]:
        """Run a manual dashboard action and return the resulting state."""

        result = resolved_gateway.execute_action(session_id, request)
        return {
            "result": result,
            "state": resolved_gateway.fetch_state(session_id),
            "events": resolved_gateway.fetch_events(session_id),
        }

    @app.post("/api/agent/{session_id}/chat/stream")
    def chat_stream(session_id: str, request: ChatRequest) -> StreamingResponse:
        """Stream Agent execution events to the browser."""

        return StreamingResponse(
            resolved_gateway.stream_chat(session_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Serialize one server-sent event."""

    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


app = create_app()


if __name__ == "__main__":
    uvicorn.run("agent.server:app", host="0.0.0.0", port=7860, reload=False)
