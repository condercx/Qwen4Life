"""默认场景与演示动作序列。"""

from __future__ import annotations

from environment.devices import AirConditioner, Light, WashingMachine


def build_default_devices() -> dict[str, object]:
	"""构造最小设备集合。"""

	light = Light(
		device_id="living_room_light_1",
		device_type="light",
		name="客厅主灯",
	)
	ac = AirConditioner(
		device_id="living_room_ac_1",
		device_type="ac",
		name="客厅空调",
	)
	washing_machine = WashingMachine(
		device_id="washing_machine_1",
		device_type="washing_machine",
		name="阳台洗衣机",
	)
	return {light.device_id: light, ac.device_id: ac, washing_machine.device_id: washing_machine}


def build_discrete_demo_requests(session_id: str) -> list[dict]:
	"""离散控制示例：灯光与空调即时控制。"""

	return [
		{
			"request_id": "demo-discrete-1",
			"session_id": session_id,
			"intent": "打开客厅的灯",
			"action": {
				"device": "light",
				"target": "living_room_light_1",
				"command": "turn_on",
				"params": {},
			},
		},
		{
			"request_id": "demo-discrete-2",
			"session_id": session_id,
			"intent": "把客厅灯调到 60 亮度",
			"action": {
				"device": "light",
				"target": "living_room_light_1",
				"command": "set_brightness",
				"params": {"brightness": 60},
			},
		},
		{
			"request_id": "demo-discrete-3",
			"session_id": session_id,
			"intent": "把空调设成制冷",
			"action": {
				"device": "ac",
				"target": "living_room_ac_1",
				"command": "set_mode",
				"params": {"mode": "cool"},
			},
		},
		{
			"request_id": "demo-discrete-4",
			"session_id": session_id,
			"intent": "把空调调到 24 度",
			"action": {
				"device": "ac",
				"target": "living_room_ac_1",
				"command": "set_temperature",
				"params": {"temperature": 24.0},
			},
		},
	]


def build_timed_demo_requests(session_id: str) -> list[dict]:
	"""计时任务示例：启动洗衣并在后续轮询查看状态。"""

	return [
		{
			"request_id": "demo-timed-1",
			"session_id": session_id,
			"intent": "开始标准洗衣",
			"action": {
				"device": "washing_machine",
				"target": "washing_machine_1",
				"command": "start_wash",
				"params": {"program": "standard", "duration_seconds": 6},
			},
		},
		{"kind": "wait", "seconds": 2, "label": "等待 2 秒，模拟后台计时"},
		{"kind": "poll", "label": "2 秒后查询洗衣机状态"},
		{
			"request_id": "demo-timed-2",
			"session_id": session_id,
			"intent": "洗衣过程中打开客厅灯",
			"action": {
				"device": "light",
				"target": "living_room_light_1",
				"command": "turn_on",
				"params": {},
			},
		},
		{"kind": "wait", "seconds": 5, "label": "再等待 5 秒，观察洗衣完成"},
		{"kind": "poll", "label": "洗衣完成后查询状态"},
	]


def build_mixed_demo_requests(session_id: str) -> list[dict]:
	"""混合示例：回家联动加洗衣任务。"""

	return [
		{
			"request_id": "demo-mixed-1",
			"session_id": session_id,
			"intent": "我到家了，帮我准备客厅",
			"action": {
				"device": "light",
				"target": "living_room_light_1",
				"command": "turn_on",
				"params": {},
			},
		},
		{
			"request_id": "demo-mixed-2",
			"session_id": session_id,
			"intent": "把空调设成制冷并调到 24 度",
			"action": {
				"device": "ac",
				"target": "living_room_ac_1",
				"command": "set_mode",
				"params": {"mode": "cool"},
			},
		},
		{
			"request_id": "demo-mixed-3",
			"session_id": session_id,
			"intent": "继续把空调调到 24 度",
			"action": {
				"device": "ac",
				"target": "living_room_ac_1",
				"command": "set_temperature",
				"params": {"temperature": 24.0},
			},
		},
		{
			"request_id": "demo-mixed-4",
			"session_id": session_id,
			"intent": "顺便开始洗衣服",
			"action": {
				"device": "washing_machine",
				"target": "washing_machine_1",
				"command": "start_wash",
				"params": {"program": "quick", "duration_seconds": 8},
			},
		},
		{"kind": "wait", "seconds": 3, "label": "等待 3 秒，查看洗衣剩余时间"},
		{"kind": "poll", "label": "混合场景中间状态"},
	]
