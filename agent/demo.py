"""Agent 命令行演示入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agent import SimpleSmartHomeAgent


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


def _process_stream(
    agent: SimpleSmartHomeAgent,
    session_id: str,
    user_input: str,
    verbose: bool,
) -> None:
    """消费 Agent 流式输出并渲染到终端。"""

    print("\n助手> ", end="", flush=True)
    renderer = _NormalAnswerRenderer()
    verbose_renderer = _VerboseLogRenderer() if verbose else None

    for chunk in agent.handle_user_input_stream(session_id, user_input):
        chunk_type = chunk["type"]
        text = chunk["content"]

        if verbose:
            assert verbose_renderer is not None
            verbose_renderer.render(chunk_type, text)
            continue

        if chunk_type in {"reasoning", "content"}:
            renderer.feed(chunk_type, text)
            continue

        if chunk_type == "action_start":
            renderer.reset()
            continue

        if chunk_type == "observation":
            continue

        if chunk_type == "final_reply":
            if not renderer.has_printed_answer and text.strip():
                print(text.strip(), end="", flush=True)
            print("\n")
            renderer.reset()
            continue

        if chunk_type == "error":
            print(f"{text}\n")


class _NormalAnswerRenderer:
    """普通模式只展示最终 Answer 文本。"""

    def __init__(self) -> None:
        self.buffer = ""
        self.has_printed_answer = False

    def feed(self, chunk_type: str, text: str) -> None:
        """消费模型流式片段，隐藏 Thought/Action，只输出 Answer 后的内容。"""

        if chunk_type != "content":
            return

        if self.has_printed_answer:
            print(text, end="", flush=True)
            return

        self.buffer += text
        answer_index = self.buffer.find("Answer:")
        if answer_index == -1:
            return

        self.has_printed_answer = True
        answer_text = self.buffer[answer_index + len("Answer:") :].lstrip()
        if answer_text:
            print(answer_text, end="", flush=True)
        self.buffer = ""

    def reset(self) -> None:
        """开始下一次模型调用时重置普通输出状态。"""

        self.buffer = ""
        self.has_printed_answer = False


class _VerboseLogRenderer:
    """详细模式按事件类型聚合输出，避免每个 token 都换行。"""

    def __init__(self) -> None:
        self.current_stream_type = ""

    def render(self, chunk_type: str, text: str) -> None:
        """渲染一个流式事件，模型连续片段会合并到同一段。"""

        if chunk_type in {"reasoning", "content"}:
            self._render_model_chunk(chunk_type, text)
            return

        self.current_stream_type = ""
        if chunk_type == "final_reply":
            print(f"\n[Agent事件/final_reply]\n{text}\n", flush=True)
            return
        if chunk_type == "error":
            print(f"\n[Agent事件/error]\n{text}", flush=True)
            return
        print(f"\n[Agent事件/{chunk_type}]\n{text.strip()}", flush=True)

    def _render_model_chunk(self, chunk_type: str, text: str) -> None:
        if self.current_stream_type != chunk_type:
            self.current_stream_type = chunk_type
            print(f"\n[模型返回/{chunk_type}]\n", end="", flush=True)
        print(text, end="", flush=True)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="智能家居 ReAct Agent demo")
    parser.add_argument("user_input", nargs="?", help="单轮用户输入")
    parser.add_argument("--session-id", default="agent-demo-session", help="会话 ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示完整流式调试事件和模型原始片段")
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
