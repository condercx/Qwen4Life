"""Tests for ReAct output parsing."""

from __future__ import annotations

import unittest

from agent.parser import parse_react_output


class ReactParserTests(unittest.TestCase):
    """Verify model output is parsed into one executable step."""

    def test_multiple_action_lines_parse_only_first_complete_action(self) -> None:
        output = (
            "Thought: 用户要求多个设备控制，需要逐个执行。\n"
            'Action: control_device(device_id="living_room_light_1", command="turn_on")\n'
            'Action: control_device(device_id="living_room_curtain_1", command="close")\n'
            'Action: control_device(device_id="living_room_ac_1", command="set_temperature", params={"temperature": 24})'
        )

        step = parse_react_output(output)

        self.assertEqual(step.type, "action")
        self.assertEqual(step.tool_name, "control_device")
        self.assertEqual(
            step.tool_args,
            {"device_id": "living_room_light_1", "command": "turn_on"},
        )
        self.assertEqual(
            step.raw_action_text,
            'Action: control_device(device_id="living_room_light_1", command="turn_on")',
        )

    def test_action_parser_keeps_json_params_for_first_action(self) -> None:
        output = (
            "Thought: 需要调空调。\n"
            'Action: control_device(device_id="living_room_ac_1", command="set_temperature", params={"temperature": 24})\n'
            'Action: query_all_devices()'
        )

        step = parse_react_output(output)

        self.assertEqual(
            step.tool_args,
            {
                "device_id": "living_room_ac_1",
                "command": "set_temperature",
                "params": {"temperature": 24},
            },
        )


if __name__ == "__main__":
    unittest.main()
