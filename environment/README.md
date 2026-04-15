# 智能家居模拟环境

这个目录提供一个最小可运行的 Python 模拟环境，用于后续 agent 在同进程内调用。当前版本聚焦两类能力：

- 即时离散控制：灯光、空调这类命令立即生效。
- 计时任务：洗衣机这类命令启动后会在后台按真实时间推进，下次读取状态时可以直接看到剩余时间或完成结果。

## 目录结构

- `actions.py`：动作请求协议、错误码、返回结构。
- `agent_adapter.py`：面向 agent 的调用适配层，隔离环境核心与后续协议传输。
- `demo.py`：命令行入口，只负责参数解析和分发。
- `demo_catalog.py`：demo 列表与示例元数据。
- `demo_runner.py`：demo 执行流程编排，支持等待和状态轮询。
- `demo_output.py`：终端摘要输出与详细日志写入。
- `devices.py`：灯光、空调、洗衣机设备状态机。
- `smart_home_env.py`：环境入口，暴露 `reset`、`step`、`get_state`、`get_events`。
- `scenarios.py`：默认设备与演示动作序列。
- `requirements.txt`：依赖说明。

## 最小 API

### 1. reset(session_id)

重置指定会话，返回初始 observation。

### 2. step(request_json)

执行一次动作请求。对于洗衣机这类计时设备，环境会在每次 API 调用前自动同步真实时间，因此不需要额外的推进参数。

### 3. get_state(session_id)

读取当前状态快照。这里会先同步后台计时任务，再返回最新状态。

### 4. get_events(session_id)

读取并清空当前未读事件。这里同样会先同步后台计时任务，因此洗衣完成事件会在下一次轮询时自然出现。

## Agent 调用方式

推荐不要让 agent 直接依赖 `SmartHomeEnv` 内部细节，而是通过 `AgentEnvironmentAdapter` 调用。

### 初始化环境

```python
from environment import AgentEnvironmentAdapter

adapter = AgentEnvironmentAdapter()
initial_state = adapter.create_session("agent-session-001")
```

### 发送动作

```python
response = adapter.send_action(
  "agent-session-001",
  intent="开始洗衣服",
  action={
    "device": "washing_machine",
    "target": "washing_machine_1",
    "command": "start_wash",
    "params": {"program": "standard"}
  },
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
  intent="把空调调到 24 度",
  action={
    "device": "ac",
    "target": "living_room_ac_1",
    "command": "set_temperature",
    "params": {"temperature": 24.0}
  },
)
response = adapter.send_request(request)
```

## 请求格式

```json
{
  "request_id": "r-001",
  "session_id": "s-home-01",
  "intent": "开始洗衣服",
  "action": {
    "device": "washing_machine",
    "target": "washing_machine_1",
    "command": "start_wash",
    "params": {
      "program": "standard"
    }
  }
}
```

## 当前支持的动作

### 灯光

- `turn_on`
- `turn_off`
- `set_brightness`

### 空调

- `turn_on`
- `turn_off`
- `set_mode`
- `set_temperature`
- `set_fan_speed`

### 洗衣机

- `start_wash`
- `pause`
- `resume`
- `cancel`

默认情况下，如果没有传入 `duration_seconds`，洗衣时长按 30 分钟处理。

## 运行方式

从仓库根目录执行：

```bash
python -m environment.demo
```

只运行单个示例：

```bash
python -m environment.demo --demo timed
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
python -m environment.demo --demo timed --log-file environment/custom-demo.log
```

## 输出说明

- 终端输出：显示精简摘要，便于直接看懂每一步动作是否成功、产生了哪些关键事件、最终设备处于什么状态。
- 详细日志：完整原始返回体会默认写入 `environment/environment.log`，适合排查协议字段、事件 payload 和状态变化细节。

### 终端输出建议重点关注

- `初始化状态`：检查默认设备是否正确加载。
- `请求`：查看当前执行的意图、动作、参数。
- `等待`：查看 demo 中为了模拟后台计时而产生的停顿。
- `状态轮询`：查看下一次 get_state 时洗衣机还剩多久或是否完成。
- `事件汇总`：查看本轮累计发生过的关键事件。

## 扩展建议

1. 增加更多计时设备，例如烘干机、洗碗机、烤箱。
2. 在 `actions.py` 增加序列号、ACK、CRC 等字段，逐步模拟嵌入式协议。
3. 将 `step` 封装到 HTTP、MQTT 或串口适配层，而不是直接修改环境核心。
4. 增加任务编排器，把“回家模式”“睡眠模式”变成宏动作。
