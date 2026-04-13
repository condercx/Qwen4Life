# 智能家居模拟环境

这个目录提供一个最小可运行的 Python 模拟环境，用于后续 agent 在同进程内调用。当前版本目标是先把“离散动作 + 连续动作 + 会话状态 + 事件轮询”这条链路打通。

## 目录结构

- `actions.py`：动作请求协议、错误码、返回结构。
- `agent_adapter.py`：面向 agent 的调用适配层，隔离环境核心与后续协议传输。
- `demo.py`：命令行入口，只负责参数解析和分发。
- `demo_catalog.py`：demo 列表与示例元数据。
- `demo_runner.py`：demo 执行流程编排。
- `demo_output.py`：终端摘要输出与详细日志写入。
- `devices.py`：房间模型、灯光、空调、扫地机器人。
- `smart_home_env.py`：环境入口，暴露 `reset`、`step`、`get_state`、`get_events`。
- `scenarios.py`：默认设备与演示动作序列。
- `requirements.txt`：依赖说明。

## 最小 API

### 1. reset(session_id)

重置指定会话，返回初始 observation。

### 2. step(request_json)

执行一次动作请求，并按 `options.advance_ticks` 推进模拟时间。

### 3. get_state(session_id)

读取当前状态快照。

### 4. get_events(session_id)

读取并清空当前未读事件。

## Agent 调用方式

推荐不要让 agent 直接依赖 `SmartHomeEnv` 内部细节，而是通过 `AgentEnvironmentAdapter` 调用。

### 初始化环境

```python
from environment import AgentEnvironmentAdapter

adapter = AgentEnvironmentAdapter()
initial_state = adapter.create_session("agent-session-001")
```

这里有两层初始化：

- 环境初始化：`AgentEnvironmentAdapter()` 内部创建 `SmartHomeEnv()`，通常整个进程只做一次。
- 会话初始化：`create_session(session_id)`，每个 agent 会话开始时调用一次。

### 发送动作

```python
response = adapter.send_action(
  "agent-session-001",
  intent="打开客厅灯",
  action={
    "mode": "discrete",
    "device": "light",
    "target": "living_room_light_1",
    "command": "turn_on",
    "params": {}
  },
  options={"advance_ticks": 1},
)
```

### 读取状态与事件

```python
state = adapter.fetch_state("agent-session-001")
events = adapter.fetch_events("agent-session-001")
```

### 如果上层已经生成了完整协议请求

```python
request = adapter.build_request(
  session_id="agent-session-001",
  intent="让扫地机器人去门口",
  action={
    "mode": "continuous",
    "device": "robot_vacuum",
    "target": "robot_vacuum_1",
    "command": "move_to",
    "params": {"x": 1.0, "y": 7.0, "speed": 0.6}
  },
  options={"advance_ticks": 5},
)
response = adapter.send_request(request)
```

## 演变路径

推荐按下面的路径演进，而不是一开始就把环境和传输协议耦合在一起：

### v0：本地直连

- Agent -> Adapter -> SmartHomeEnv
- 优点：开发快、调试简单、最适合先验证环境逻辑和 agent 规划链路。

### v1：统一 JSON 协议

- Agent -> Adapter -> JSON 请求体 -> SmartHomeEnv
- 目标：把本地函数调用和协议请求格式统一起来，便于日志记录、回放和测试。

### v2：接入传输层

- Agent -> Adapter -> HTTP / MQTT / 串口 / WebSocket -> 协议解析 -> SmartHomeEnv
- 目标：把同一套动作请求挂到不同传输层上，模拟或接入真实嵌入式设备。

### 为什么先做 Adapter

- 环境核心只关心状态机和动作执行，不关心数据是怎么传进来的。
- Agent 侧只依赖稳定的 `create_session/send_action/fetch_state/fetch_events` 接口。
- 后续接协议时，主要改 adapter 和 transport 层，不需要重写环境核心。

## 请求格式

```json
{
  "request_id": "r-001",
  "session_id": "s-home-01",
  "intent": "打开客厅灯",
  "action": {
    "mode": "discrete",
    "device": "light",
    "target": "living_room_light_1",
    "command": "turn_on",
    "params": {}
  },
  "options": {
    "advance_ticks": 1
  }
}
```

## 连续动作示例

```json
{
  "request_id": "r-002",
  "session_id": "s-home-01",
  "intent": "让扫地机器人去门口",
  "action": {
    "mode": "continuous",
    "device": "robot_vacuum",
    "target": "robot_vacuum_1",
    "command": "move_to",
    "params": {
      "x": 1.0,
      "y": 7.0,
      "speed": 0.6
    }
  },
  "options": {
    "advance_ticks": 5
  }
}
```

## 当前支持的动作

### 灯光

- 离散：`turn_on`、`turn_off`
- 连续：`set_brightness`

### 空调

- 离散：`turn_on`、`turn_off`、`set_mode`
- 连续：`set_temperature`、`set_fan_speed`

### 扫地机器人

- 离散：`start_cleaning`、`stop`、`dock`
- 连续：`move_to`

### 系统时钟

- 离散：`advance`

## 运行方式

从仓库根目录执行：

```bash
python -m environment.demo
```

只运行单个示例：

```bash
python -m environment.demo --demo discrete
```

一次运行多个示例：

```bash
python -m environment.demo --demo discrete --demo mixed
```

查看可用示例列表：

```bash
python -m environment.demo --list
```

指定详细日志输出文件：

```bash
python -m environment.demo --demo continuous --log-file environment/custom-demo.log
```

## 输出说明

- 终端输出：显示精简摘要，便于直接看懂每一步动作是否成功、产生了哪些关键事件、最终设备处于什么状态。
- 详细日志：完整原始返回体会默认写入 `environment/environment.log`，适合排查协议字段、事件 payload 和状态变化细节。

### 终端输出建议重点关注

- `初始化状态`：检查默认设备是否正确加载。
- `请求`：查看当前执行的意图、动作、参数。
- `结果`：确认本次 step 是否成功，以及当前 `sim_time` 和事件数量。
- `事件`：查看灯光、空调、机器人是否产生了预期变化。
- `最终状态`：查看整组 demo 执行后的完整设备摘要。
- `事件汇总`：查看本轮累计发生过的关键事件。

## 扩展建议

1. 增加传感器设备，例如门磁、空气质量、人体存在检测。
2. 在 `actions.py` 增加序列号、ACK、CRC 等字段，逐步模拟嵌入式协议。
3. 将 `step` 封装到 HTTP、MQTT 或串口适配层，而不是直接修改环境核心。
4. 增加任务编排器，把“回家模式”“睡眠模式”变成宏动作。
