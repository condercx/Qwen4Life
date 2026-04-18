# Environment

`environment/` 目录实现了一个独立的智能家居模拟环境。

它的职责是：

- 管理会话级环境状态。
- 模拟设备动作和状态变化。
- 生成设备事件。
- 通过 FastAPI 暴露 HTTP 接口，供 Agent 调用。

## 当前模块

- `actions.py`：动作协议、请求解析、统一错误响应。
- `devices.py`：灯光、空调、洗衣机等设备模型和命令处理逻辑。
- `scenarios.py`：默认设备和 demo 场景构造器。
- `smart_home_env.py`：环境核心，负责 `reset`、`step`、`get_state`、`get_events`。
- `server.py`：FastAPI 服务入口。
- `remote_adapter.py`：给 Agent 端用的 HTTP 适配器。
- `demo.py`：命令行 demo 入口。
- `demo_runner.py`：演示流程编排。
- `demo_output.py`：日志与终端输出格式化。
- `demo_catalog.py`：demo 注册表。

## 当前设备

默认场景中包含 3 类设备：

- `living_room_light_1`：客厅主灯。
- `living_room_ac_1`：客厅空调。
- `washing_machine_1`：阳台洗衣机。

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

## 运行方式

启动服务：

```bash
python -m environment.server
```

查看可用 demo：

```bash
python environment/demo.py --list
```

运行全部 demo：

```bash
python environment/demo.py
```

运行指定 demo：

```bash
python environment/demo.py --demo discrete
python environment/demo.py --demo timed
python environment/demo.py --demo mixed
```

## 当前 demo

- `discrete`：离散控制示例。
- `timed`：计时任务示例。
- `mixed`：多设备联动示例。

## 当前限制

- 目前主要是最小可运行环境，不是完整 IoT 平台。
- 事件类型和设备种类都比较少。
- 还没有正式测试目录。
