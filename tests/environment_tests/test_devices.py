"""测试新增设备模型的命令和快照行为。"""

from __future__ import annotations

import unittest

from environment.actions import (
    ERROR_DEVICE_OFFLINE,
    ERROR_UNSUPPORTED_COMMAND,
    ProtocolError,
)
from environment.devices import Curtain, SmartPlug, TemperatureHumiditySensor


class ExtendedDeviceTests(unittest.TestCase):
    """验证窗帘、传感器和智能插座的设备级规则。"""

    def test_curtain_supports_position_commands(self) -> None:
        curtain = Curtain(
            device_id="living_room_curtain_1",
            device_type="curtain",
            name="客厅窗帘",
        )

        events = curtain.handle_command(
            "set_position",
            {"position_percent": 65},
            current_time=0.0,
        )

        self.assertEqual(curtain.snapshot()["position_percent"], 65)
        self.assertEqual(events[0]["type"], "curtain_position_changed")

    def test_sensor_is_read_only(self) -> None:
        sensor = TemperatureHumiditySensor(
            device_id="living_room_sensor_1",
            device_type="temperature_humidity_sensor",
            name="客厅温湿度传感器",
        )

        with self.assertRaises(ProtocolError) as context:
            sensor.handle_command("turn_on", {}, current_time=0.0)

        self.assertEqual(context.exception.code, ERROR_UNSUPPORTED_COMMAND)

    def test_sensor_reports_offline_before_read_only_error(self) -> None:
        sensor = TemperatureHumiditySensor(
            device_id="living_room_sensor_1",
            device_type="temperature_humidity_sensor",
            name="客厅温湿度传感器",
            online=False,
        )

        with self.assertRaises(ProtocolError) as context:
            sensor.handle_command("turn_on", {}, current_time=0.0)

        self.assertEqual(context.exception.code, ERROR_DEVICE_OFFLINE)

    def test_smart_plug_supports_switching_and_power_reading(self) -> None:
        plug = SmartPlug(
            device_id="desk_plug_1",
            device_type="smart_plug",
            name="书房插座",
        )

        plug.handle_command("turn_on", {"power_watts": 12.5}, current_time=0.0)
        self.assertTrue(plug.snapshot()["is_on"])
        self.assertEqual(plug.snapshot()["power_watts"], 12.5)

        plug.handle_command("turn_off", {}, current_time=1.0)
        self.assertFalse(plug.snapshot()["is_on"])
        self.assertEqual(plug.snapshot()["power_watts"], 0.0)


if __name__ == "__main__":
    unittest.main()
