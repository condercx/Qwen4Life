"""演示流程编排器。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from environment.demo_catalog import DEMO_BUILDERS, get_demo_names
from environment.demo_output import (
    append_log,
    ensure_stdout_encoding,
    initialize_log_file,
    print_events_summary,
    print_observation_summary,
    print_runtime_overview,
    print_step_summary,
    print_title,
    print_wait_summary,
)
from environment.smart_home_env import SmartHomeEnv


def run(selected_demos: list[str] | None = None, log_path: Path | None = None) -> None:
    """运行指定的演示集合。"""

    ensure_stdout_encoding()
    chosen_demos = selected_demos or get_demo_names()
    resolved_log_path = log_path or Path(__file__).with_name("environment.log")
    initialize_log_file(resolved_log_path, chosen_demos)
    print_runtime_overview(chosen_demos, resolved_log_path)

    environment = SmartHomeEnv()
    for demo_name in chosen_demos:
        run_single_demo(environment, demo_name, resolved_log_path)


def run_single_demo(environment: SmartHomeEnv, demo_name: str, log_path: Path) -> None:
    """运行单个演示流程。"""

    title, session_id, builder = DEMO_BUILDERS[demo_name]
    print_title(title)

    reset_state = environment.reset(session_id)
    append_log(log_path, demo_name, "reset", reset_state)
    print_observation_summary("初始化状态", session_id, reset_state)

    for step in builder(session_id):
        kind = step.get("kind", "request")
        if kind == "wait":
            _handle_wait_step(step, log_path, demo_name)
            continue
        if kind == "poll":
            _handle_poll_step(environment, session_id, step, log_path, demo_name)
            continue

        response = environment.step(step)
        append_log(log_path, demo_name, "step", {"request": step, "response": response})
        print_step_summary(step, response)

    final_state = environment.get_state(session_id)
    append_log(log_path, demo_name, "state", final_state)
    print_observation_summary("最终状态", session_id, final_state)

    events = environment.get_events(session_id)
    append_log(log_path, demo_name, "events", events)
    print_events_summary(events)


def _handle_wait_step(step: dict[str, Any], log_path: Path, demo_name: str) -> None:
    """处理等待步骤。"""

    seconds = float(step["seconds"])
    print_wait_summary(step.get("label", "等待后台任务推进"), seconds)
    time.sleep(seconds)
    append_log(log_path, demo_name, "wait", step)


def _handle_poll_step(
    environment: SmartHomeEnv,
    session_id: str,
    step: dict[str, Any],
    log_path: Path,
    demo_name: str,
) -> None:
    """处理轮询步骤。"""

    label = step.get("label", "状态轮询")
    state = environment.get_state(session_id)
    events = environment.get_events(session_id)
    payload = {"label": label, "state": state, "events": events}
    append_log(log_path, demo_name, "poll", payload)
    print_observation_summary(label, session_id, state)
    print_events_summary(events)
