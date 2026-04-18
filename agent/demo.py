"""Agent 命令行演示入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agent import SimpleSmartHomeAgent

THOUGHT_PREFIX = "\n[思考中] "
ACTION_PREFIX = "[执行动作]"
OBSERVATION_PREFIX = "观察结果"


def main() -> None:
    """运行单轮或交互式 Agent 演示。"""

    _configure_utf8_stdout()
    args = _parse_args()
    agent = SimpleSmartHomeAgent()
    session_id = args.session_id

    if args.user_input:
        _process_stream(agent, session_id, args.user_input, args.verbose)
        return

    print(f"已进入交互模式，session_id={session_id}，输入 exit 结束。")
    while True:
        try:
            user_input = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        _process_stream(agent, session_id, user_input, args.verbose)


class _VerboseRenderer:
    """在 verbose 模式下按关键字切分流式输出。"""

    keywords: dict[str, tuple[str, str]] = {
        "Thought:": (THOUGHT_PREFIX, "thought"),
        "Answer:": ("", "answer"),
        "Action:": ("", "action"),
    }

    def __init__(self, showed_reasoning: bool = False) -> None:
        self.state = "detect"
        self.buffer = ""
        self.showed_reasoning = showed_reasoning

    def feed(self, text: str) -> None:
        """接收一个增量文本块。"""

        self.buffer += text
        self._drain()

    def flush(self) -> None:
        """在轮次结束时清空缓冲区。"""

        if not self.buffer:
            self.state = "detect"
            return

        if self.state != "action":
            print(self.buffer, end="", flush=True)
        self.buffer = ""
        self.state = "detect"

    def _drain(self) -> None:
        """持续消费缓冲区中的内容。"""

        while self.buffer:
            previous_length = len(self.buffer)
            if self.state == "detect":
                self._handle_detect()
            elif self.state in {"thought", "passthrough"}:
                self._handle_streaming()
            elif self.state == "answer":
                print(self.buffer, end="", flush=True)
                self.buffer = ""
            elif self.state == "action":
                self.buffer = ""

            if len(self.buffer) == previous_length:
                break

    def _handle_detect(self) -> None:
        """识别当前缓冲区是否以关键字开头。"""

        stripped_text = self.buffer.lstrip("\n\r\t ")
        if not stripped_text:
            return

        for keyword, (prefix, next_state) in self.keywords.items():
            if stripped_text.startswith(keyword):
                if next_state == "thought" and not self.showed_reasoning:
                    print(prefix, end="", flush=True)
                elif next_state == "thought" and self.showed_reasoning:
                    print(flush=True)
                elif next_state == "answer":
                    print("\n", end="", flush=True)

                keyword_index = self.buffer.find(keyword) + len(keyword)
                self.buffer = self.buffer[keyword_index:].lstrip(" ")
                self.state = next_state
                return

        if any(keyword.startswith(stripped_text) for keyword in self.keywords):
            return
        self.state = "passthrough"

    def _handle_streaming(self) -> None:
        """在流式输出中查找下一段关键字边界。"""

        position = 0
        while position < len(self.buffer):
            if self.buffer[position] != "\n":
                position += 1
                continue

            next_start = position + 1
            while next_start < len(self.buffer) and self.buffer[next_start] in "\n\r\t ":
                next_start += 1

            trailing_text = self.buffer[next_start:]
            if not trailing_text:
                if position > 0:
                    print(self.buffer[:position], end="", flush=True)
                    self.buffer = self.buffer[position:]
                return

            for keyword in self.keywords:
                if keyword == "Thought:":
                    continue
                if trailing_text.startswith(keyword):
                    print(self.buffer[:position], end="", flush=True)
                    self.buffer = self.buffer[next_start:]
                    self.state = "detect"
                    return
                if keyword.startswith(trailing_text):
                    print(self.buffer[:position], end="", flush=True)
                    self.buffer = self.buffer[position:]
                    return

            position = next_start

        print(self.buffer, end="", flush=True)
        self.buffer = ""


def _process_stream(
    agent: SimpleSmartHomeAgent,
    session_id: str,
    user_input: str,
    verbose: bool,
) -> None:
    """消费 Agent 流式输出并渲染到终端。"""

    print("\n助手> ", end="", flush=True)
    content_buffer = ""
    printing_answer = False
    showed_reasoning = False
    renderer = _VerboseRenderer(showed_reasoning=False) if verbose else None

    for chunk in agent.handle_user_input_stream(session_id, user_input):
        chunk_type = chunk["type"]
        text = chunk["content"]

        if chunk_type == "reasoning":
            if verbose:
                if not showed_reasoning:
                    print(THOUGHT_PREFIX, end="", flush=True)
                    showed_reasoning = True
                    assert renderer is not None
                    renderer.showed_reasoning = True
                print(text, end="", flush=True)
            continue

        if chunk_type == "content":
            if verbose:
                assert renderer is not None
                renderer.feed(text)
                continue

            content_buffer += text
            if not printing_answer:
                answer_index = content_buffer.find("Answer:")
                if answer_index != -1:
                    printing_answer = True
                    answer_text = content_buffer[answer_index + len("Answer:") :].lstrip()
                    if answer_text:
                        print(answer_text, end="", flush=True)
            else:
                print(text, end="", flush=True)
            continue

        if chunk_type == "action_start":
            if verbose:
                assert renderer is not None
                renderer.flush()
                print(ACTION_PREFIX)
                print(text.strip())
                showed_reasoning = False
                renderer = _VerboseRenderer(showed_reasoning=False)
            content_buffer = ""
            continue

        if chunk_type == "observation":
            if verbose:
                print(f"{OBSERVATION_PREFIX}: {text.strip()}\n\n助手> ", end="", flush=True)
            continue

        if chunk_type == "final_reply":
            if verbose and renderer is not None:
                renderer.flush()
            elif not printing_answer and text.strip():
                print(text.strip(), end="", flush=True)
            print("\n")
            content_buffer = ""
            printing_answer = False
            showed_reasoning = False
            if verbose:
                renderer = _VerboseRenderer(showed_reasoning=False)
            continue

        if chunk_type == "error":
            print(f"{text}\n")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="智能家居 ReAct Agent demo")
    parser.add_argument("user_input", nargs="?", help="单轮用户输入")
    parser.add_argument("--session-id", default="agent-demo-session", help="会话 ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示完整推理过程")
    return parser.parse_args()


def _configure_utf8_stdout() -> None:
    """尽量保证 Windows 终端中文输出稳定。"""

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
