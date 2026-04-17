"""最小 agent 命令行演示（ReAct 模式）。"""

from __future__ import annotations

import argparse
import sys

from agent import SimpleSmartHomeAgent


def main() -> None:
	"""支持单轮和交互式的最小命令行调用。"""

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
	"""流式关键字检测渲染器。

	在段落开头缓冲少量字符检测 Thought:/Answer:/Action: 关键字，
	检测完成后立即切换为逐字符流式输出，保持打字机效果。
	"""

	_KEYWORDS: dict[str, tuple[str, str]] = {
		"Thought:": ("\n[💭 思考中...] ", "thought"),
		"Answer:":  ("", "answer"),
		"Action:":  ("", "action"),
	}
	_MAX_KW_LEN = 8  # max(len(k) for k in _KEYWORDS) == len("Thought:")

	def __init__(self, showed_reasoning: bool = False) -> None:
		self.state = "detect"	 # detect | thought | answer | action | passthrough
		self.buf = ""
		self.showed_reasoning = showed_reasoning

	def feed(self, text: str) -> None:
		"""喂入一个 content chunk，实时检测关键字并输出。"""
		self.buf += text
		self._drain()

	def flush(self) -> None:
		"""轮次结束时，把剩余缓冲全部输出。"""
		if self.buf:
			if self.state == "detect":
				# 缓冲中没凑够关键字长度就结束了，直接输出
				print(self.buf, end="", flush=True)
			elif self.state != "action":
				print(self.buf, end="", flush=True)
			self.buf = ""
		self.state = "detect"

	# ── 内部实现 ──────────────────────────────────────

	def _drain(self) -> None:
		"""循环处理缓冲区，尽可能多地输出内容。"""
		while self.buf:
			prev_len = len(self.buf)

			if self.state == "detect":
				self._handle_detect()
			elif self.state in ("thought", "passthrough"):
				self._handle_streaming()
			elif self.state == "answer":
				# answer 段落直接全速输出
				print(self.buf, end="", flush=True)
				self.buf = ""
			elif self.state == "action":
				# action 内容由 action_start 事件处理，丢弃
				self.buf = ""

			if len(self.buf) == prev_len:
				break  # 没有进展，等更多数据

	def _handle_detect(self) -> None:
		"""在段落开头检测关键字。"""
		stripped = self.buf.lstrip("\n\r\t ")
		if not stripped:
			return

		# 完整匹配
		for kw, (prefix, state) in self._KEYWORDS.items():
			if stripped.startswith(kw):
				if state == "thought" and not self.showed_reasoning:
					print(prefix, end="", flush=True)
				elif state == "thought" and self.showed_reasoning:
					print(flush=True)  # 结束 reasoning 行
				elif state == "answer":
					print("\n", end="", flush=True)  # 换行分隔思考与回复
				idx = self.buf.find(kw) + len(kw)
				self.buf = self.buf[idx:].lstrip(" ")
				self.state = state
				return

		# 部分匹配（可能凑出关键字）
		if any(kw.startswith(stripped) and len(stripped) < len(kw) for kw in self._KEYWORDS):
			return  # 等更多数据

		# 不匹配任何关键字
		self.state = "passthrough"

	def _handle_streaming(self) -> None:
		"""流式输出内容，同时扫描段落分隔符检测下一个关键字。"""
		pos = 0
		while pos < len(self.buf):
			if self.buf[pos] != "\n":
				pos += 1
				continue

			# 找到换行符，检查后面是否跟着关键字
			after_start = pos + 1
			while after_start < len(self.buf) and self.buf[after_start] in "\n\r\t ":
				after_start += 1

			after = self.buf[after_start:]

			if not after:
				# 换行后没有更多内容，先输出换行前的部分，保留换行符等后续判断
				if pos > 0:
					print(self.buf[:pos], end="", flush=True)
					self.buf = self.buf[pos:]
				return

			# 检查是否有关键字
			for kw in self._KEYWORDS:
				if kw == "Thought:":
					continue  # Thought 不会出现在段落中间
				if after.startswith(kw):
					# 找到下一个关键字！输出之前的内容，重新进入检测
					print(self.buf[:pos], end="", flush=True)
					self.buf = self.buf[after_start:]
					self.state = "detect"
					return
				if kw.startswith(after) and len(after) < len(kw):
					# 部分匹配，先输出前面的，保留后面等待
					print(self.buf[:pos], end="", flush=True)
					self.buf = self.buf[pos:]
					return

			# 不是关键字，继续扫描
			pos = after_start

		# 扫完全部内容，没有关键字边界，全部输出
		print(self.buf, end="", flush=True)
		self.buf = ""


def _process_stream(agent: SimpleSmartHomeAgent, session_id: str, user_input: str, verbose: bool) -> None:
	print(f"\n助手> ", end="", flush=True)

	content_buffer = ""		   # 非 verbose：累积内容用于检测 Answer:
	printing_answer = False	   # 非 verbose：是否已开始输出 Answer 文本
	showed_reasoning = False   # 当前轮是否已有 native reasoning 输出
	renderer: _VerboseRenderer | None = None

	if verbose:
		renderer = _VerboseRenderer(showed_reasoning=False)

	for chunk in agent.handle_user_input_stream(session_id, user_input):
		type_ = chunk["type"]
		text = chunk["content"]

		if type_ == "reasoning":
			if verbose:
				if not showed_reasoning:
					print("\n[💭 思考中...] ", end="", flush=True)
					showed_reasoning = True
					renderer.showed_reasoning = True
				print(text, end="", flush=True)

		elif type_ == "content":
			if verbose:
				renderer.feed(text)
			else:
				content_buffer += text
				if not printing_answer:
					answer_idx = content_buffer.find("Answer:")
					if answer_idx != -1:
						printing_answer = True
						after = content_buffer[answer_idx + len("Answer:"):].lstrip()
						if after:
							print(after, end="", flush=True)
				else:
					print(text, end="", flush=True)

		elif type_ == "action_start":
			if verbose:
				renderer.flush()
				print("[🎯 确定行动...]")
				print(f"🔧 {text.strip()}")
				# 重置 renderer 用于下一轮
				showed_reasoning = False
				renderer = _VerboseRenderer(showed_reasoning=False)
			content_buffer = ""

		elif type_ == "observation":
			if verbose:
				print(f"📋 观测结果: {text.strip()}\n\n助手> ", end="", flush=True)

		elif type_ == "final_reply":
			if verbose:
				renderer.flush()
			elif not printing_answer and text.strip():
				print(text.strip(), end="", flush=True)
			print("\n")
			content_buffer = ""
			printing_answer = False
			showed_reasoning = False
			if verbose:
				renderer = _VerboseRenderer(showed_reasoning=False)

		elif type_ == "error":
			print(f"{text}\n")


def _parse_args() -> argparse.Namespace:
	"""解析命令行参数。"""

	parser = argparse.ArgumentParser(description="智能家居 ReAct agent demo")
	parser.add_argument("user_input", nargs="?", help="单轮用户输入")
	parser.add_argument("--session-id", default="agent-demo-session", help="会话 ID")
	parser.add_argument("--verbose", "-v", action="store_true", help="显示完整推理链")
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
