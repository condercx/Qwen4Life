"""demo 终端输出与详细日志辅助工具。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def ensure_stdout_encoding() -> None:
	"""尽量将标准输出切换到 UTF-8，避免 Windows 终端打印中文失败。"""

	if hasattr(sys.stdout, "reconfigure"):
		try:
			sys.stdout.reconfigure(encoding="utf-8", errors="replace")
		except ValueError:
			# 某些嵌入式或重定向场景下 stdout 不允许重新配置，此时保持默认行为。
			pass


def initialize_log_file(log_path: Path, chosen_demos: list[str]) -> None:
	"""初始化日志文件，确保每次运行都从新文件开始。"""

	log_path.parent.mkdir(parents=True, exist_ok=True)
	header = {
		"created_at": datetime.now().isoformat(timespec="seconds"),
		"chosen_demos": chosen_demos,
	}
	log_path.write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")


def append_log(log_path: Path, demo_name: str, stage: str, payload: dict | list[dict]) -> None:
	"""将复杂原始返回体追加写入日志文件。"""

	entry = {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"demo": demo_name,
		"stage": stage,
		"payload": payload,
	}
	with log_path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(entry, ensure_ascii=False, indent=2))
		handle.write("\n\n")


def print_runtime_overview(chosen_demos: list[str], log_path: Path) -> None:
	"""输出本次运行概览。"""

	print("将运行的示例: " + ", ".join(chosen_demos))
	print(f"详细日志文件: {log_path}")


def print_title(title: str) -> None:
	"""输出 demo 标题。"""

	print("\n" + "=" * 16)
	print(title)
	print("=" * 16)


def print_observation_summary(label: str, session_id: str, observation: dict) -> None:
	"""打印环境状态摘要，避免终端被 JSON 淹没。"""

	print(f"[{label}] session={session_id} | observed_at={observation['observed_at']}")
	for device in observation["devices"].values():
		print("  - " + describe_device(device))


def print_step_summary(request: dict, response: dict) -> None:
	"""打印单次 step 的人类可读摘要。"""

	action = request["action"]
	print(f"[请求] {request['request_id']} | 意图: {request.get('intent') or '无'}")
	print(f"  动作: {action['target']} -> {action['command']} | params={action['params']}")
	if response["success"]:
		metrics = response.get("metrics", {})
		print(
			"  结果: 成功"
			f" | observed_at={response['observation']['observed_at']}"
			f" | 新事件={len(response['events'])}"
			f" | 未读事件={metrics.get('unread_event_count', 0)}"
		)
		if response["events"]:
			print("  事件:")
			for event in response["events"]:
				print("    * " + describe_event(event))
		else:
			print("  事件: 无")
	else:
		error = response["error"]
		print(f"  结果: 失败 | code={error['code']} | message={error['message']}")


def print_events_summary(events: list[dict]) -> None:
	"""打印累计事件摘要。"""

	print(f"[事件汇总] 共 {len(events)} 条")
	if not events:
		print("  - 无未读事件")
		return
	for event in events:
		print("  - " + describe_event(event))


def print_wait_summary(label: str, seconds: float) -> None:
	"""打印等待后台计时的摘要。"""

	print(f"[等待] {label} | {seconds:.1f}s")


def describe_device(device: dict) -> str:
	"""将设备状态格式化为简洁中文描述。"""

	device_type = device["device_type"]
	if device_type == "light":
		state = "开" if device["is_on"] else "关"
		return f"灯光 {device['name']}: {state}，亮度={device['brightness']}"
	if device_type == "ac":
		state = "开" if device["is_on"] else "关"
		return (
			f"空调 {device['name']}: {state}，模式={device['mode']}，"
			f"目标温度={device['target_temperature']}，风速={device['fan_speed']}"
		)
	if device_type == "washing_machine":
		remaining = format_duration(device["remaining_seconds"])
		return (
			f"洗衣机 {device['name']}: 状态={device['status']}，程序={device['program']}，"
			f"剩余时间={remaining}"
		)
	return f"设备 {device['name']}: {json.dumps(device, ensure_ascii=False)}"


def describe_event(event: dict) -> str:
	"""将事件转换为更容易扫读的一行文本。"""

	payload = event["payload"]
	event_type = event["type"]
	if event_type == "session_reset":
		return f"t={event['occurred_at']} 会话已重置"
	if event_type in {"light_state_changed", "light_brightness_changed"}:
		state = "开" if payload["is_on"] else "关"
		return f"t={event['occurred_at']} 灯光状态更新: {state}，亮度={payload['brightness']}"
	if event_type == "ac_state_changed":
		state = "开" if payload["is_on"] else "关"
		return f"t={event['occurred_at']} 空调状态更新: {state}，模式={payload['mode']}"
	if event_type == "ac_setting_changed":
		return (
			f"t={event['occurred_at']} 空调参数更新: 模式={payload['mode']}，"
			f"温度={payload['target_temperature']}，风速={payload['fan_speed']}"
		)
	if event_type == "washing_started":
		return (
			f"t={event['occurred_at']} 洗衣已启动: 程序={payload['program']}，"
			f"预计结束={payload['expected_finish_at']}"
		)
	if event_type == "washing_paused":
		return f"t={event['occurred_at']} 洗衣已暂停: 剩余时间={format_duration(payload['remaining_seconds'])}"
	if event_type == "washing_resumed":
		return f"t={event['occurred_at']} 洗衣已继续: 剩余时间={format_duration(payload['remaining_seconds'])}"
	if event_type == "washing_completed":
		return f"t={event['occurred_at']} 洗衣已完成: 程序={payload['program']}"
	if event_type == "washing_cancelled":
		return f"t={event['occurred_at']} 洗衣任务已取消: 程序={payload['program']}"
	return f"t={event['occurred_at']} {event_type}: {json.dumps(payload, ensure_ascii=False)}"
def format_duration(seconds: int) -> str:
	"""将剩余时间格式化为便于阅读的中文。"""

	if seconds <= 0:
		return "0 秒"
	if seconds < 60:
		return f"{seconds} 秒"
	minutes, remain = divmod(seconds, 60)
	if remain == 0:
		return f"{minutes} 分钟"
	return f"{minutes} 分 {remain} 秒"
