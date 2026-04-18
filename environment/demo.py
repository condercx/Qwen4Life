"""环境演示脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from environment.demo_catalog import DEMO_BUILDERS, get_demo_names
from environment.demo_runner import run


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="智能家居模拟环境 demo")
    parser.add_argument(
        "--demo",
        dest="demos",
        action="append",
        choices=get_demo_names(),
        help="选择要运行的 demo，可重复传入多次。",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=str(Path(__file__).with_name("environment.log")),
        help="详细日志输出路径。",
    )
    parser.add_argument(
        "--list",
        dest="show_list",
        action="store_true",
        help="只显示可用 demo 列表，不实际执行。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.show_list:
        print("可用 demo：")
        for demo_name, (title, _, _) in DEMO_BUILDERS.items():
            print(f"- {demo_name}: {title}")
        raise SystemExit(0)
    run(arguments.demos, Path(arguments.log_file))
