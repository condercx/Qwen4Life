"""演示目录与元数据定义。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from environment.scenarios import (
    build_discrete_demo_requests,
    build_mixed_demo_requests,
    build_timed_demo_requests,
)

DemoBuilder: TypeAlias = Callable[[str], list[dict[str, Any]]]
DemoMeta: TypeAlias = tuple[str, str, DemoBuilder]

DEMO_BUILDERS: dict[str, DemoMeta] = {
    "discrete": ("离散控制示例", "demo-discrete-session", build_discrete_demo_requests),
    "timed": ("计时任务示例", "demo-timed-session", build_timed_demo_requests),
    "mixed": ("混合联动示例", "demo-mixed-session", build_mixed_demo_requests),
}


def get_demo_names() -> list[str]:
    """返回所有可用的演示名称。"""

    return list(DEMO_BUILDERS.keys())
