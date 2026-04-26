# Environment

`environment/` 目录实现了一个独立的智能家居模拟环境。

它的职责是：

- 管理会话级环境状态。
- 模拟设备动作和状态变化。
- 生成设备事件。
- 通过 FastAPI 暴露 HTTP 接口，供 Agent 调用。

## 当前模块

- `actions.py`：动作协议、请求解析、统一错误响应。
- `adapter.py`：环境适配器协议和内存适配器，供 Agent 单元测试绕过 HTTP 服务。
- `clock.py`：环境时间源抽象，支持生产环境真实时间和测试环境可控时间。
- `devices.py`：灯光、空调、洗衣机、窗帘、传感器、智能插座等设备模型和命令处理逻辑。
- `scenarios.py`：设备 fixture 构造器和 fixture 注册表。
- `smart_home_env.py`：环境核心，负责 `reset`、`step`、`get_state`、`get_events`。
- `server.py`：FastAPI 服务入口。
- `remote_adapter.py`：给 Agent 端用的 HTTP 适配器。

## 当前设备

默认场景中包含 6 类设备：

- `living_room_light_1`：客厅主灯。
- `living_room_ac_1`：客厅空调。
- `washing_machine_1`：阳台洗衣机。
- `living_room_curtain_1`：客厅窗帘。
- `living_room_sensor_1`：客厅温湿度传感器。
- `desk_plug_1`：书房插座。

## 测试设备场景

`environment.scenarios` 提供多组可复用设备 fixture：

- `default`：默认六设备初始状态。
- `evening_home`：客厅灯已开启，空调处于制冷 24 度，窗帘部分打开，传感器读数较高。
- `all_offline`：所有默认设备离线。
- `washing_running`：洗衣机已在运行中。
- `washing_paused`：洗衣机已暂停。

`SmartHomeEnv` 支持注入 `device_factory`，每次 `reset()` 都会创建一组新的设备实例，避免测试之间共享可变状态。
计时类 fixture 会避免生成负 Unix 时间戳，保证在 Windows 和 Unix-like 系统上都能稳定序列化快照。

```python
from environment.scenarios import build_all_offline_devices
from environment.smart_home_env import SmartHomeEnv

env = SmartHomeEnv(device_factory=build_all_offline_devices)
state = env.reset("offline-test")
```

## 当前接口

环境服务默认运行在 `http://127.0.0.1:6666`，提供以下接口：

- `POST /session/{session_id}/reset`
- `GET /session/{session_id}/state`
- `GET /session/{session_id}/events`
- `POST /session/{session_id}/action`

其中 `action` 接口用于执行设备动作，请求示例：

```json
{
  "request_id": "req-1",
  "intent": "打开客厅灯",
  "action": {
    "device": "light",
    "target": "living_room_light_1",
    "command": "turn_on",
    "params": {}
  }
}
```

动作字段也兼容 `name` / `args` 别名，服务层会归一化为 `command` / `params` 后再交给环境核心处理。

## 测试友好的时间控制

`SmartHomeEnv` 默认使用真实系统时间，不影响 HTTP 服务。单元测试或确定性模拟可以注入 `FakeClock`，这样带计时逻辑的设备不需要真实等待。

```python
from environment.clock import FakeClock
from environment.scenarios import build_washing_running_devices
from environment.smart_home_env import SmartHomeEnv

clock = FakeClock(current_time=1_700_000_000)
env = SmartHomeEnv(
    clock=clock,
    device_factory=lambda: build_washing_running_devices(current_time=clock.now(), duration_seconds=5),
)
env.reset("test-session")
clock.advance(5)
events = env.get_events("test-session")
```

## 内存环境适配器

Agent 生产运行时通过 `RemoteEnvironmentAdapter` 访问 HTTP 服务。单元测试可以使用 `InMemoryEnvironmentAdapter`，它复用同一个 `SmartHomeEnv` 接口，但不发网络请求。

```python
from environment.adapter import InMemoryEnvironmentAdapter
from environment.smart_home_env import SmartHomeEnv

adapter = InMemoryEnvironmentAdapter(env=SmartHomeEnv())
adapter.create_session("test-session")
state = adapter.fetch_state("test-session")
```

## 运行方式

启动服务：

```bash
python -m environment.server
```

快速本地验证可直接实例化 `SmartHomeEnv` 或 `InMemoryEnvironmentAdapter`。仓库不再保留 environment 命令行 demo，避免测试路径依赖真实等待、终端输出和日志文件。

## 测试

环境测试位于 `tests/environment_tests/`，使用标准库 `unittest`：

```bash
python -m unittest discover -s tests
```

当前覆盖设备 fixture、新增设备模型、可控时间推进、离线设备错误响应、session 状态隔离和 FastAPI 服务边界。

## 当前限制

- 目前主要是最小可运行环境，不是完整 IoT 平台。
- 事件类型仍然比较少，暂未模拟复杂联动、延迟执行和设备上报噪声。
