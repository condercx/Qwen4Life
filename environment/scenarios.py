"""默认场景与演示请求构造器。"""

from __future__ import annotations

from typing import Any

from environment.devices import AirConditioner, Device, Light, WashingMachine


def build_default_devices() -> dict[str, Device]:
    """构造默认设备集合。"""

    light = Light(
        device_id="living_room_light_1",
        device_type="light",
        name="客厅主灯",
    )
    air_conditioner = AirConditioner(
        device_id="living_room_ac_1",
        device_type="ac",
        name="客厅空调",
    )
    washing_machine = WashingMachine(
        device_id="washing_machine_1",
        device_type="washing_machine",
        name="阳台洗衣机",
    )
    return {
        light.device_id: light,
        air_conditioner.device_id: air_conditioner,
        washing_machine.device_id: washing_machine,
    }


def build_discrete_demo_requests(session_id: str) -> list[dict[str, Any]]:
    """构造离散控制演示请求。"""

    return [
        _build_step_request(
            request_id="demo-discrete-1",
            session_id=session_id,
            intent="打开客厅的灯",
            device="light",
            target="living_room_light_1",
            command="turn_on",
        ),
        _build_step_request(
            request_id="demo-discrete-2",
            session_id=session_id,
            intent="把客厅灯调到 60 亮度",
            device="light",
            target="living_room_light_1",
            command="set_brightness",
            params={"brightness": 60},
        ),
        _build_step_request(
            request_id="demo-discrete-3",
            session_id=session_id,
            intent="把空调设成制冷",
            device="ac",
            target="living_room_ac_1",
            command="set_mode",
            params={"mode": "cool"},
        ),
        _build_step_request(
            request_id="demo-discrete-4",
            session_id=session_id,
            intent="把空调调到 24 度",
            device="ac",
            target="living_room_ac_1",
            command="set_temperature",
            params={"temperature": 24.0},
        ),
    ]


def build_timed_demo_requests(session_id: str) -> list[dict[str, Any]]:
    """构造计时任务演示请求。"""

    return [
        _build_step_request(
            request_id="demo-timed-1",
            session_id=session_id,
            intent="开始标准洗衣",
            device="washing_machine",
            target="washing_machine_1",
            command="start_wash",
            params={"program": "standard", "duration_seconds": 6},
        ),
        {"kind": "wait", "seconds": 2, "label": "等待 2 秒，模拟后台计时"},
        {"kind": "poll", "label": "2 秒后查询洗衣机状态"},
        _build_step_request(
            request_id="demo-timed-2",
            session_id=session_id,
            intent="洗衣过程中打开客厅灯",
            device="light",
            target="living_room_light_1",
            command="turn_on",
        ),
        {"kind": "wait", "seconds": 5, "label": "再等待 5 秒，观察洗衣完成"},
        {"kind": "poll", "label": "洗衣完成后查询状态"},
    ]


def build_mixed_demo_requests(session_id: str) -> list[dict[str, Any]]:
    """构造多设备联动演示请求。"""

    return [
        _build_step_request(
            request_id="demo-mixed-1",
            session_id=session_id,
            intent="我到家了，帮我准备客厅",
            device="light",
            target="living_room_light_1",
            command="turn_on",
        ),
        _build_step_request(
            request_id="demo-mixed-2",
            session_id=session_id,
            intent="把空调设成制冷",
            device="ac",
            target="living_room_ac_1",
            command="set_mode",
            params={"mode": "cool"},
        ),
        _build_step_request(
            request_id="demo-mixed-3",
            session_id=session_id,
            intent="继续把空调调到 24 度",
            device="ac",
            target="living_room_ac_1",
            command="set_temperature",
            params={"temperature": 24.0},
        ),
        _build_step_request(
            request_id="demo-mixed-4",
            session_id=session_id,
            intent="顺便开始洗衣服",
            device="washing_machine",
            target="washing_machine_1",
            command="start_wash",
            params={"program": "quick", "duration_seconds": 8},
        ),
        {"kind": "wait", "seconds": 3, "label": "等待 3 秒，查看洗衣剩余时间"},
        {"kind": "poll", "label": "混合场景中间状态"},
    ]


def _build_step_request(
    request_id: str,
    session_id: str,
    intent: str,
    device: str,
    target: str,
    command: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一格式的环境请求。"""

    return {
        "request_id": request_id,
        "session_id": session_id,
        "intent": intent,
        "action": {
            "device": device,
            "target": target,
            "command": command,
            "params": params or {},
        },
    }
