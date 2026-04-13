"""默认场景与演示动作序列。"""

from __future__ import annotations

from environment.devices import AirConditioner, Light, RobotVacuum, Room


def build_default_room() -> Room:
	"""构造一个带障碍区的客厅场景。"""

	return Room(
		room_id="living_room",
		name="客厅",
		width=10.0,
		height=8.0,
		blocked_zones=[{"x1": 4.0, "y1": 2.0, "x2": 5.5, "y2": 5.5}],
	)


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
	robot = RobotVacuum(
		device_id="robot_vacuum_1",
		device_type="robot_vacuum",
		name="扫地机器人",
		position_x=0.5,
		position_y=0.5,
	)
	return {light.device_id: light, ac.device_id: ac, robot.device_id: robot}


def build_discrete_demo_requests(session_id: str) -> list[dict]:
	"""离散控制示例：开灯、关灯、查询状态。"""

	return [
		{
			"request_id": "demo-discrete-1",
			"session_id": session_id,
			"intent": "打开客厅的灯",
			"action": {
				"mode": "discrete",
				"device": "light",
				"target": "living_room_light_1",
				"command": "turn_on",
				"params": {},
			},
			"options": {"advance_ticks": 1},
		},
		{
			"request_id": "demo-discrete-2",
			"session_id": session_id,
			"intent": "关闭客厅的灯",
			"action": {
				"mode": "discrete",
				"device": "light",
				"target": "living_room_light_1",
				"command": "turn_off",
				"params": {},
			},
			"options": {"advance_ticks": 0},
		},
	]


def build_continuous_demo_requests(session_id: str) -> list[dict]:
	"""连续控制示例：下发机器人目标点并推进时间。"""

	return [
		{
			"request_id": "demo-continuous-1",
			"session_id": session_id,
			"intent": "让扫地机器人移动到沙发旁边",
			"action": {
				"mode": "continuous",
				"device": "robot_vacuum",
				"target": "robot_vacuum_1",
				"command": "move_to",
				"params": {"x": 2.0, "y": 6.5, "speed": 0.8},
			},
			"options": {"advance_ticks": 3},
		},
		{
			"request_id": "demo-continuous-2",
			"session_id": session_id,
			"intent": "继续推进环境，观察机器人位置",
			"action": {
				"mode": "discrete",
				"device": "system",
				"target": "clock",
				"command": "advance",
				"params": {"ticks": 5},
			},
			"options": {"advance_ticks": 0},
		},
	]


def build_mixed_demo_requests(session_id: str) -> list[dict]:
	"""混合示例：夜间回家联动。"""

	return [
		{
			"request_id": "demo-mixed-1",
			"session_id": session_id,
			"intent": "我到家了，帮我准备客厅",
			"action": {
				"mode": "discrete",
				"device": "light",
				"target": "living_room_light_1",
				"command": "turn_on",
				"params": {},
			},
			"options": {"advance_ticks": 0},
		},
		{
			"request_id": "demo-mixed-2",
			"session_id": session_id,
			"intent": "把空调设成制冷并调到 24 度",
			"action": {
				"mode": "discrete",
				"device": "ac",
				"target": "living_room_ac_1",
				"command": "set_mode",
				"params": {"mode": "cool"},
			},
			"options": {"advance_ticks": 0},
		},
		{
			"request_id": "demo-mixed-3",
			"session_id": session_id,
			"intent": "继续把空调调到 24 度",
			"action": {
				"mode": "continuous",
				"device": "ac",
				"target": "living_room_ac_1",
				"command": "set_temperature",
				"params": {"temperature": 24.0},
			},
			"options": {"advance_ticks": 0},
		},
		{
			"request_id": "demo-mixed-4",
			"session_id": session_id,
			"intent": "让扫地机器人移到门口待命",
			"action": {
				"mode": "continuous",
				"device": "robot_vacuum",
				"target": "robot_vacuum_1",
				"command": "move_to",
				"params": {"x": 1.0, "y": 7.0, "speed": 0.6},
			},
			"options": {"advance_ticks": 5},
		},
	]
