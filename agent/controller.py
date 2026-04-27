"""智能家居 ReAct Agent 控制器。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

from agent.llm_client import LLMClient, create_default_llm_client
from agent.llm_config import LLMConfig
from agent.memory import AgentMemory, create_default_agent_memory
from agent.memory_config import MemoryConfig
from agent.memory_decision import request_memory_decision
from agent.parser import parse_react_output
from agent.prompts import build_system_prompt, build_user_prompt
from agent.schema import AgentResult, ReactStep
from agent.tools import ToolRegistry

Message: TypeAlias = dict[str, str]
StreamEvent: TypeAlias = dict[str, str]
logger = logging.getLogger(__name__)

MAX_REACT_STEPS = 8
MAX_HISTORY_TURNS = 20
CONTINUE_PROMPT = "请继续，并给出 Action 或 Answer。"
FALLBACK_REPLY = "抱歉，我在处理这个请求时遇到了一些问题。请稍后再试，或换一种说法。"


@dataclass(slots=True)
class SimpleSmartHomeAgent:
    """通过 ReAct 循环协调模型推理与工具执行。"""

    tools: ToolRegistry = field(default_factory=ToolRegistry)
    client: LLMClient | None = None
    config: LLMConfig | None = None
    memory: AgentMemory | None = None
    _session_histories: dict[str, list[Message]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """延迟创建默认 LLM 客户端。"""

        if self.client is None:
            self.client = create_default_llm_client(self.config or LLMConfig.from_env())
        if self.memory is None:
            memory_config = MemoryConfig.from_env()
            if memory_config.enabled:
                try:
                    self.memory = create_default_agent_memory(memory_config)
                except Exception as exc:
                    logger.warning("长期记忆初始化失败，本轮运行将不启用 memory：%s", exc)
        if self.memory is not None:
            self.tools.set_memory(self.memory)

    def create_session(self, session_id: str) -> None:
        """创建或重置会话。"""

        normalized_session_id = self._validate_session_id(session_id)
        self.tools.adapter.create_session(normalized_session_id)
        self._session_histories[normalized_session_id] = []

    def handle_user_input(self, session_id: str, user_input: str) -> AgentResult:
        """以非流式方式处理单次用户输入。"""

        normalized_session_id = self._validate_session_id(session_id)
        normalized_user_input = self._validate_user_input(user_input)
        history, all_messages = self._prepare_conversation(
            normalized_session_id,
            normalized_user_input,
        )

        steps: list[ReactStep] = []
        for _ in range(MAX_REACT_STEPS):
            try:
                raw_output = self.client.chat_completion(all_messages)
            except Exception:
                return self._build_fallback_result(
                    session_id=normalized_session_id,
                    user_input=normalized_user_input,
                    history=history,
                    steps=steps,
                    raw_messages=all_messages,
                )

            parsed_step = parse_react_output(raw_output)
            steps.append(parsed_step)

            if parsed_step.type == "answer":
                self._store_assistant_message(history, raw_output)
                self._trim_history(normalized_session_id)
                self._save_memory_turn(
                    session_id=normalized_session_id,
                    user_input=normalized_user_input,
                    assistant_reply=parsed_step.content,
                )
                return AgentResult(
                    session_id=normalized_session_id,
                    user_input=normalized_user_input,
                    reply=parsed_step.content,
                    steps=steps,
                    raw_messages=list(all_messages) + [{"role": "assistant", "content": raw_output}],
                )

            if parsed_step.type == "action" and parsed_step.tool_name:
                self._execute_action_step(
                    session_id=normalized_session_id,
                    parsed_step=parsed_step,
                    all_messages=all_messages,
                    steps=steps,
                    raw_output=raw_output,
                )
                continue

            self._continue_reasoning(all_messages, raw_output)

        return self._build_fallback_result(
            session_id=normalized_session_id,
            user_input=normalized_user_input,
            history=history,
            steps=steps,
            raw_messages=all_messages,
        )

    def handle_user_input_stream(
        self,
        session_id: str,
        user_input: str,
    ) -> Iterator[StreamEvent]:
        """以流式方式处理单次用户输入。"""

        normalized_session_id = self._validate_session_id(session_id)
        normalized_user_input = self._validate_user_input(user_input)
        history, all_messages = self._prepare_conversation(
            normalized_session_id,
            normalized_user_input,
        )

        for _ in range(MAX_REACT_STEPS):
            raw_output = ""
            try:
                for chunk in self.client.chat_completion_stream(all_messages):
                    chunk_type = str(chunk.get("type", ""))
                    content = str(chunk.get("content", ""))
                    if chunk_type in {"reasoning", "content"}:
                        yield {"type": chunk_type, "content": content}
                    if chunk_type == "content":
                        raw_output += content
            except Exception as exc:
                yield {"type": "error", "content": f"\n[请求异常]：{exc}\n"}
                break

            parsed_step = parse_react_output(raw_output)
            if parsed_step.type == "answer":
                self._store_assistant_message(history, raw_output)
                self._trim_history(normalized_session_id)
                self._save_memory_turn(
                    session_id=normalized_session_id,
                    user_input=normalized_user_input,
                    assistant_reply=parsed_step.content,
                )
                yield {"type": "final_reply", "content": parsed_step.content}
                return

            if parsed_step.type == "action" and parsed_step.tool_name:
                yield {
                    "type": "action_start",
                    "content": f"\n[调用工具: {parsed_step.tool_name}({parsed_step.tool_args})]\n",
                }
                observation = self.tools.execute(
                    session_id=normalized_session_id,
                    tool_name=parsed_step.tool_name,
                    args=parsed_step.tool_args or {},
                )
                yield {"type": "observation", "content": f"[工具返回: {observation}]\n"}
                self._append_action_messages(all_messages, raw_output, observation)
                continue

            self._continue_reasoning(all_messages, raw_output)

        self._store_assistant_message(history, FALLBACK_REPLY)
        self._trim_history(normalized_session_id)
        yield {"type": "error", "content": FALLBACK_REPLY}

    def _prepare_conversation(
        self,
        session_id: str,
        user_input: str,
    ) -> tuple[list[Message], list[Message]]:
        """构造当前轮次的系统消息和上下文。"""

        self._ensure_session(session_id)
        history = self._session_histories[session_id]
        history.append({"role": "user", "content": build_user_prompt(user_input)})
        memory_prompt = self._search_memory_context(session_id, user_input)
        system_message = {
            "role": "system",
            "content": build_system_prompt(self.tools.get_tools_prompt(), memory_prompt=memory_prompt),
        }
        return history, [system_message, *history]

    def _search_memory_context(self, session_id: str, user_input: str) -> str:
        """检索长期记忆上下文，失败时不影响主流程。"""

        if self.memory is None:
            return ""
        try:
            return self.memory.search_context(
                user_id=session_id,
                session_id=session_id,
                query=user_input,
            )
        except Exception as exc:
            logger.debug("长期记忆检索失败，已忽略：%s", exc)
            return ""

    def _save_memory_turn(self, session_id: str, user_input: str, assistant_reply: str) -> None:
        """由 agent 判断并保存成功轮次的长期记忆，失败时不影响主流程。"""

        if self.memory is None or not assistant_reply.strip():
            return
        try:
            decision = request_memory_decision(self.client, user_input, assistant_reply)
            if not decision.should_save:
                return
            self.memory.save_memory(
                user_id=session_id,
                session_id=session_id,
                memory_text=decision.memory_text,
                memory_type=decision.memory_type,
            )
        except Exception as exc:
            logger.debug("长期记忆保存失败，已忽略：%s", exc)

    def _execute_action_step(
        self,
        session_id: str,
        parsed_step: ReactStep,
        all_messages: list[Message],
        steps: list[ReactStep],
        raw_output: str,
    ) -> None:
        """执行动作步骤并写回 Observation。"""

        if not parsed_step.tool_name:
            return

        observation = self.tools.execute(
            session_id=session_id,
            tool_name=parsed_step.tool_name,
            args=parsed_step.tool_args or {},
        )
        steps.append(ReactStep(type="observation", content=observation))
        self._append_action_messages(all_messages, raw_output, observation)

    @staticmethod
    def _append_action_messages(
        all_messages: list[Message],
        raw_output: str,
        observation: str,
    ) -> None:
        """把动作输出和观测结果追加到上下文。"""

        all_messages.append({"role": "assistant", "content": raw_output})
        all_messages.append({"role": "user", "content": f"Observation: {observation}"})

    @staticmethod
    def _continue_reasoning(all_messages: list[Message], raw_output: str) -> None:
        """要求模型继续补全动作或最终回答。"""

        all_messages.append({"role": "assistant", "content": raw_output})
        all_messages.append({"role": "user", "content": CONTINUE_PROMPT})

    @staticmethod
    def _store_assistant_message(history: list[Message], content: str) -> None:
        """把助手消息写入会话历史。"""

        history.append({"role": "assistant", "content": content})

    def _build_fallback_result(
        self,
        session_id: str,
        user_input: str,
        history: list[Message],
        steps: list[ReactStep],
        raw_messages: list[Message],
    ) -> AgentResult:
        """构造统一的兜底返回。"""

        self._store_assistant_message(history, FALLBACK_REPLY)
        self._trim_history(session_id)
        return AgentResult(
            session_id=session_id,
            user_input=user_input,
            reply=FALLBACK_REPLY,
            steps=steps,
            raw_messages=list(raw_messages) + [{"role": "assistant", "content": FALLBACK_REPLY}],
        )

    def _ensure_session(self, session_id: str) -> None:
        """按需初始化会话。"""

        if session_id not in self._session_histories:
            self.create_session(session_id)

    def _trim_history(self, session_id: str, max_turns: int = MAX_HISTORY_TURNS) -> None:
        """裁剪会话历史，避免上下文无限增长。"""

        history = self._session_histories.get(session_id)
        if history and len(history) > max_turns:
            self._session_histories[session_id] = history[-max_turns:]

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        """校验会话 ID。"""

        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("session_id 不能为空。")
        return normalized_session_id

    @staticmethod
    def _validate_user_input(user_input: str) -> str:
        """校验用户输入。"""

        normalized_user_input = user_input.strip()
        if not normalized_user_input:
            raise ValueError("user_input 不能为空。")
        return normalized_user_input
