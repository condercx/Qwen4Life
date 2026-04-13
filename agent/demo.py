"""最小 agent 命令行演示。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from agent import SimpleSmartHomeAgent


def main() -> None:
	"""支持单轮和交互式的最小命令行调用。"""

	_ensure_stdout_encoding()
	args = _parse_args()
	agent = SimpleSmartHomeAgent()
	session_id = args.session_id

	if args.user_input:
		result = agent.handle_user_input(session_id, args.user_input)
		_print_result(result)
		return

	print(f"已进入交互模式，session_id={session_id}，输入 exit 结束。")
	while True:
		user_input = input("用户> ").strip()
		if not user_input:
			continue
		if user_input.lower() in {"exit", "quit"}:
			break
		result = agent.handle_user_input(session_id, user_input)
		_print_result(result)


def _parse_args() -> argparse.Namespace:
	"""解析命令行参数。"""

	parser = argparse.ArgumentParser(description="智能家居最小 agent demo")
	parser.add_argument("user_input", nargs="?", help="单轮用户输入")
	parser.add_argument("--session-id", default="agent-demo-session", help="会话 ID")
	return parser.parse_args()


def _print_result(result: object) -> None:
	"""打印 agent 结果摘要。"""

	print("\n[模型计划]")
	print(json.dumps(asdict(result.plan), ensure_ascii=False, indent=2))
	print("[Agent 回复]")
	print(result.reply)


def _ensure_stdout_encoding() -> None:
	"""尽量保证 Windows 终端中文输出稳定。"""

	if hasattr(sys.stdout, "reconfigure"):
		try:
			sys.stdout.reconfigure(encoding="utf-8", errors="replace")
		except ValueError:
			pass


if __name__ == "__main__":
	main()
