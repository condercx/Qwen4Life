"""测试和模拟复用的环境设备 fixture。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from environment.devices import (
    AirConditioner,
    Curtain,
    Device,
    Light,
    SmartPlug,
    TemperatureHumiditySensor,
    WashingMachine,
)

DeviceFactory = Callable[[], dict[str, Device]]
DeviceT = TypeVar("DeviceT", bound=Device)


def build_default_devices() -> dict[str, Device]:
    """构造新会话默认使用的基础设备集合。"""

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
    curtain = Curtain(
        device_id="living_room_curtain_1",
        device_type="curtain",
        name="客厅窗帘",
    )
    sensor = TemperatureHumiditySensor(
        device_id="living_room_sensor_1",
        device_type="temperature_humidity_sensor",
        name="客厅温湿度传感器",
    )
    smart_plug = SmartPlug(
        device_id="desk_plug_1",
        device_type="smart_plug",
        name="书房插座",
    )
    return {
        light.device_id: light,
        air_conditioner.device_id: air_conditioner,
        washing_machine.device_id: washing_machine,
        curtain.device_id: curtain,
        sensor.device_id: sensor,
        smart_plug.device_id: smart_plug,
    }


def build_evening_home_devices() -> dict[str, Device]:
    """构造晚间到家场景的设备 fixture。"""

    devices = build_default_devices()
    light = _require_device(devices, "living_room_light_1", Light)
    air_conditioner = _require_device(devices, "living_room_ac_1", AirConditioner)
    curtain = _require_device(devices, "living_room_curtain_1", Curtain)
    sensor = _require_device(devices, "living_room_sensor_1", TemperatureHumiditySensor)

    light.is_on = True
    light.brightness = 60
    air_conditioner.is_on = True
    air_conditioner.mode = "cool"
    air_conditioner.target_temperature = 24.0
    curtain.position_percent = 35
    sensor.temperature = 26.0
    sensor.humidity = 50.0
    return devices


def build_all_offline_devices() -> dict[str, Device]:
    """构造所有已知设备均离线的 fixture。"""

    devices = build_default_devices()
    for device in devices.values():
        device.online = False
    return devices


def build_washing_running_devices(
    current_time: float = 0.0,
    duration_seconds: int = 300,
    program: str = "standard",
) -> dict[str, Device]:
    """构造洗衣机已经运行中的 fixture。"""

    devices = build_default_devices()
    washing_machine = _require_device(devices, "washing_machine_1", WashingMachine)
    washing_machine.status = "running"
    washing_machine.program = program
    washing_machine.duration_seconds = duration_seconds
    washing_machine.remaining_seconds = duration_seconds
    washing_machine.started_at = current_time
    washing_machine.expected_finish_at = current_time + duration_seconds
    washing_machine.paused_at = None
    washing_machine.completed_at = None
    return devices


def build_washing_paused_devices(
    current_time: float = 300.0,
    remaining_seconds: int = 120,
    duration_seconds: int = 300,
    program: str = "standard",
) -> dict[str, Device]:
    """构造洗衣机在中途暂停的 fixture。"""

    devices = build_default_devices()
    washing_machine = _require_device(devices, "washing_machine_1", WashingMachine)
    elapsed_seconds = max(0, duration_seconds - remaining_seconds)
    washing_machine.status = "paused"
    washing_machine.program = program
    washing_machine.duration_seconds = duration_seconds
    washing_machine.remaining_seconds = remaining_seconds
    # Windows 上负 Unix 时间戳可能无法序列化，因此这里保证生成的时间戳非负。
    washing_machine.started_at = max(0.0, current_time - elapsed_seconds)
    washing_machine.expected_finish_at = None
    washing_machine.paused_at = current_time
    washing_machine.completed_at = None
    return devices


DEVICE_FIXTURES: dict[str, DeviceFactory] = {
    "default": build_default_devices,
    "evening_home": build_evening_home_devices,
    "all_offline": build_all_offline_devices,
    "washing_running": build_washing_running_devices,
    "washing_paused": build_washing_paused_devices,
}


def get_device_fixture_names() -> list[str]:
    """返回已注册的 fixture 名称，供测试和诊断使用。"""

    return list(DEVICE_FIXTURES)


def _require_device(
    devices: dict[str, Device],
    device_id: str,
    expected_type: type[DeviceT],
) -> DeviceT:
    """返回带类型的 fixture 设备；fixture 配置错误时尽早失败。"""

    device = devices[device_id]
    if not isinstance(device, expected_type):
        raise TypeError(f"{device_id} 必须是 {expected_type.__name__}。")
    return device
