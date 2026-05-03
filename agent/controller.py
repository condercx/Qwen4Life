"""智能家居 ReAct Agent 控制器。"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

from agent.knowledge_base import AgentKnowledgeBase, create_default_knowledge_base
from agent.knowledge_config import KnowledgeConfig
from agent.llm_client import LLMClient, create_default_llm_client
from agent.llm_config import LLMConfig
from agent.memory import AgentMemory, create_default_agent_memory
from agent.memory_config import MemoryConfig
from agent.parser import parse_react_output
from agent.prompts import build_system_prompt, build_user_prompt
from agent.schema import ReactStep
from agent.tools import ToolRegistry

Message: TypeAlias = dict[str, str]
StreamEvent: TypeAlias = dict[str, str]
logger = logging.getLogger(__name__)

MAX_REACT_STEPS = 8
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 6000
MAX_HISTORY_MESSAGE_CHARS = 1200
CONTINUE_PROMPT = "请继续，并给出 Action 或 Answer。"
EMPTY_OUTPUT_CONTINUE_PROMPT = "上一轮没有给出有效内容。请严格按格式继续，并给出一个 Action 或非空 Answer。"
FALLBACK_REPLY = "抱歉，我在处理这个请求时遇到了一些问题。请稍后再试，或换一种说法。"


@dataclass(slots=True)
class SimpleSmartHomeAgent:
    """通过 ReAct 循环协调模型推理与工具执行。"""

    tools: ToolRegistry = field(default_factory=ToolRegistry)
    client: LLMClient | None = None
    config: LLMConfig | None = None
    memory: AgentMemory | None = None
    knowledge_base: AgentKnowledgeBase | None = None
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
        if self.knowledge_base is None:
            knowledge_config = KnowledgeConfig.from_env()
            if knowledge_config.enabled:
                try:
                    self.knowledge_base = create_default_knowledge_base(knowledge_config)
                except Exception as exc:
                    logger.warning("知识库初始化失败，本轮运行将不启用 knowledge base：%s", exc)
        if self.knowledge_base is not None:
            self.tools.set_knowledge_base(self.knowledge_base)

    def create_session(self, session_id: str) -> None:
        """创建或重置会话。"""

        normalized_session_id = self._validate_session_id(session_id)
        self.tools.adapter.create_session(normalized_session_id)
        self._session_histories[normalized_session_id] = []

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
            if self._is_valid_answer(parsed_step):
                self._store_assistant_message(history, parsed_step.content)
                self._trim_history(normalized_session_id)
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
                self._append_action_messages(all_messages, _action_message_for(parsed_step, raw_output), observation)
                continue

            self._continue_reasoning(
                all_messages,
                raw_output,
                prompt=_continue_prompt_for(parsed_step),
            )

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
    def _continue_reasoning(
        all_messages: list[Message],
        raw_output: str,
        *,
        prompt: str = CONTINUE_PROMPT,
    ) -> None:
        """要求模型继续补全动作或最终回答。"""

        if raw_output.strip():
            all_messages.append({"role": "assistant", "content": raw_output})
        all_messages.append({"role": "user", "content": prompt})

    @staticmethod
    def _is_valid_answer(step: ReactStep) -> bool:
        """判断模型是否给出了可展示给用户的最终回答。"""

        return step.type == "answer" and bool(step.content.strip())

    @staticmethod
    def _store_assistant_message(history: list[Message], content: str) -> None:
        """把助手消息写入会话历史。"""

        history.append({"role": "assistant", "content": content})

    def _ensure_session(self, session_id: str) -> None:
        """按需初始化会话。"""

        if session_id not in self._session_histories:
            self.create_session(session_id)

    def _trim_history(
        self,
        session_id: str,
        max_messages: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        """裁剪会话历史，避免上下文无限增长。"""

        history = self._session_histories.get(session_id)
        if not history:
            return

        resolved_max_messages = max_messages or _get_positive_int_env(
            "AGENT_MAX_HISTORY_MESSAGES",
            MAX_HISTORY_MESSAGES,
        )
        resolved_max_chars = max_chars or _get_positive_int_env(
            "AGENT_MAX_HISTORY_CHARS",
            MAX_HISTORY_CHARS,
        )
        resolved_max_message_chars = _get_positive_int_env(
            "AGENT_MAX_HISTORY_MESSAGE_CHARS",
            MAX_HISTORY_MESSAGE_CHARS,
        )

        trimmed_history = [
            _compact_history_message(message, resolved_max_message_chars)
            for message in history[-resolved_max_messages:]
        ]
        while len(trimmed_history) > 1 and _messages_chars(trimmed_history) > resolved_max_chars:
            trimmed_history = trimmed_history[1:]
        self._session_histories[session_id] = trimmed_history

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


def _continue_prompt_for(step: ReactStep) -> str:
    """根据解析结果选择继续提示。"""

    if step.type == "empty":
        return EMPTY_OUTPUT_CONTINUE_PROMPT
    return CONTINUE_PROMPT


def _action_message_for(step: ReactStep, raw_output: str) -> str:
    """只把已执行的单个 Action 写回上下文，避免多 Action 输出污染下一轮。"""

    if step.raw_action_text:
        if step.content.strip():
            return f"Thought: {step.content.strip()}\n{step.raw_action_text}"
        return step.raw_action_text
    return raw_output


def _messages_chars(messages: list[Message]) -> int:
    """粗略计算历史消息字符数，用于控制上下文长度。"""

    return sum(len(message.get("content", "")) for message in messages)


def _compact_history_message(message: Message, max_chars: int) -> Message:
    """裁剪单条历史消息，避免长回答拖慢后续轮次。"""

    content = message.get("content", "")
    if len(content) <= max_chars:
        return dict(message)

    marker = "\n[历史消息过长，已截断]"
    keep_chars = max(1, max_chars - len(marker))
    return {
        **message,
        "content": content[:keep_chars].rstrip() + marker,
    }


def _get_positive_int_env(name: str, default: int) -> int:
    """读取正整数环境变量。"""

    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
