"""智能家居环境使用的时间源抽象。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    """用于确定性环境测试的最小时间源接口。"""

    def now(self) -> float:
        """返回当前 Unix 时间戳，单位为秒。"""
        ...


@dataclass(slots=True)
class SystemClock:
    """生产环境使用的系统真实时间源。"""

    def now(self) -> float:
        """返回当前真实时间戳。"""

        return time.time()


@dataclass(slots=True)
class FakeClock:
    """测试和确定性模拟使用的可变时间源。"""

    current_time: float = 0.0

    def now(self) -> float:
        """返回受控时间戳。"""

        return self.current_time

    def advance(self, seconds: float) -> float:
        """向前推进时间，并返回推进后的时间戳。"""

        if seconds < 0:
            raise ValueError("seconds 必须为非负数。")
        self.current_time += seconds
        return self.current_time
