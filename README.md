# Qwen4Life

Qwen4Life 是一个面向智能家居场景的最小可运行 Agent + Environment 示例项目。

当前代码分成两个核心部分：

- `environment/`：模拟智能家居环境，负责设备状态、动作执行、事件流转和 HTTP 服务。
- `agent/`：实现基于 ReAct 的智能家居 Agent，负责理解用户输入、决定是否调用工具，并把结果整理成自然语言回复。

根目录 `README.md` 以当前代码为准。`agent/README.md` 和 `environment/README.md` 仍有历史内容和乱码，暂不作为最新说明。

## 当前功能

- 提供 3 类默认设备：客厅主灯、客厅空调、阳台洗衣机。
- 支持设备控制、状态查询和事件读取。
- 支持洗衣机这类带时间推进的设备任务。
- 提供独立的 FastAPI 环境服务，Agent 通过 HTTP 调用环境。
- 提供命令行 Agent demo，支持单轮模式和交互模式。
- 提供环境演示脚本，可直接跑离散控制、计时任务和混合联动示例。

## 项目结构

```text
Qwen4Life
├── agent
│   ├── __init__.py
│   ├── controller.py
│   ├── demo.py
│   ├── llm_client.py
│   ├── llm_config.py
│   ├── parser.py
│   ├── prompts.py
│   ├── schema.py
│   └── tools.py
├── environment
│   ├── __init__.py
│   ├── actions.py
│   ├── demo.py
│   ├── demo_catalog.py
│   ├── demo_output.py
│   ├── demo_runner.py
│   ├── devices.py
│   ├── remote_adapter.py
│   ├── scenarios.py
│   ├── server.py
│   └── smart_home_env.py
├── requirements.txt
└── README.md
```

## 核心模块说明

### `environment/`

- `actions.py`：定义环境动作协议、请求校验和统一错误响应。
- `devices.py`：定义灯光、空调、洗衣机等设备模型及其命令处理逻辑。
- `scenarios.py`：构造默认设备和 demo 请求序列。
- `smart_home_env.py`：环境核心，负责会话管理、状态观测、事件派发和 `step()` 执行。
- `server.py`：暴露 FastAPI 服务。
- `remote_adapter.py`：给 Agent 用的 HTTP 适配层。
- `demo.py`、`demo_runner.py`、`demo_output.py`：环境演示脚本和日志输出。

### `agent/`

- `controller.py`：ReAct 主循环，负责会话上下文、模型调用、工具执行和 fallback 处理。
- `llm_client.py`：OpenAI 兼容接口客户端，支持非流式和流式输出。
- `llm_config.py`：从环境变量或 `.env` 读取模型配置。
- `parser.py`：解析模型输出中的 `Thought / Action / Answer`。
- `tools.py`：注册并执行 Agent 工具，当前包含设备查询和设备控制。
- `prompts.py`：系统提示词模板。
- `demo.py`：Agent 命令行入口。

## 依赖

当前根依赖如下：

```text
fastapi
uvicorn
httpx
```

安装方式：

```bash
pip install -r requirements.txt
```

说明：

- `agent/requirements.txt` 和 `environment/requirements.txt` 是历史文件，当前实际依赖以根目录 `requirements.txt` 为准。
- 如果要运行 Agent 并连接模型，还需要本地准备一个兼容 OpenAI `/v1/chat/completions` 的模型服务，例如 Ollama。

## 运行方式

### 1. 启动环境服务

```bash
python -m environment.server
```

默认监听：

- `http://127.0.0.1:6666`

### 2. 准备模型服务

当前 `agent.llm_config.LLMConfig` 默认配置如下：

- 后端：`openai_compatible_remote`
- 地址：`http://127.0.0.1:11434/v1/chat/completions`
- 模型名：`qwen3.5:4b`
- 默认 API Key：`ollama`

如果你使用 Ollama，可以按类似方式准备：

```bash
ollama run qwen3.5:4b
```

如果使用其他兼容服务，可以通过环境变量覆盖：

- `AGENT_MODEL_CHAT_COMPLETIONS_URL`
- `AGENT_MODEL_NAME`
- `AGENT_MODEL_API_KEY`
- `AGENT_MODEL_TIMEOUT_SECONDS`
- `AGENT_MODEL_ENABLE_THINKING`
- `AGENT_MODEL_THINKING_BUDGET`

### 3. 启动 Agent demo

交互模式：

```bash
python agent/demo.py
```

显示更完整的推理过程：

```bash
python agent/demo.py -v
```

单轮模式：

```bash
python agent/demo.py "帮我打开客厅灯"
```

### 4. 查看环境 demo

列出可用 demo：

```bash
python environment/demo.py --list
```

运行全部 demo：

```bash
python environment/demo.py
```

只运行指定 demo：

```bash
python environment/demo.py --demo discrete
python environment/demo.py --demo timed
python environment/demo.py --demo mixed
```

## 环境 HTTP 接口

环境服务当前提供以下接口：

- `POST /session/{session_id}/reset`：重置会话并返回初始状态。
- `GET /session/{session_id}/state`：获取当前会话状态。
- `GET /session/{session_id}/events`：获取并清空当前未读事件。
- `POST /session/{session_id}/action`：发送设备动作请求。

`POST /session/{session_id}/action` 的请求体示例：

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

## Agent 工具

当前 Agent 内置 2 个工具：

- `query_all_devices()`：查询所有设备状态。
- `control_device(device_id, command, params)`：执行指定设备控制。

工具最终都通过 `environment.remote_adapter.RemoteEnvironmentAdapter` 转发到环境服务。

## 已验证内容

当前代码已做过以下基础验证：

- `python -m compileall agent environment`
- `python environment/demo.py --list`
- `python agent/demo.py --help`
- 环境层 `reset -> step -> state` 的基本 smoke test
- `agent.parser.parse_react_output()` 的动作解析基本校验

## 当前限制

- 目前没有正式测试目录，只有基础手工验证。
- Agent 仍依赖外部模型服务，仓库本身不包含模型权重。
- `agent/README.md` 和 `environment/README.md` 还没有同步清理。
- 当前工具较少，主要覆盖最小智能家居场景。

## 后续建议

- 补充 `tests/`，把环境协议、设备状态流转和 Agent 解析逻辑覆盖起来。
- 清理子目录 README，避免和根文档不一致。
- 为 Agent 增加更多工具和更细粒度的设备语义。
- 为环境服务补充健康检查和更完整的 API 文档示例。
