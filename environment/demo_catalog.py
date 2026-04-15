"""demo 示例目录与元数据定义。"""

from __future__ import annotations

from typing import Callable

from environment.scenarios import (
	build_discrete_demo_requests,
	build_mixed_demo_requests,
	build_timed_demo_requests,
)

DEMO_BUILDERS: dict[str, tuple[str, str, Callable[[str], list[dict]]]] = {
	"discrete": ("离散控制示例", "demo-discrete-session", build_discrete_demo_requests),
	"timed": ("计时任务示例", "demo-timed-session", build_timed_demo_requests),
	"mixed": ("混合联动示例", "demo-mixed-session", build_mixed_demo_requests),
}


def get_demo_names() -> list[str]:
	"""返回所有可用 demo 名称。"""

	return list(DEMO_BUILDERS.keys())
