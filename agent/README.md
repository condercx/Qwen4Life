# Agent

`agent/` 目录实现了一个面向智能家居场景的最小 ReAct Agent。

它的职责是：

- 接收用户输入。
- 调用大模型生成 `Thought / Action / Answer`。
- 在需要时通过工具访问环境服务。
- 把工具结果继续反馈给模型，直到得到最终回复。

## 当前模块

- `controller.py`：ReAct 主循环，会话管理、上下文拼接、工具执行和兜底处理都在这里。
- `llm_client.py`：OpenAI 兼容接口客户端，支持非流式和流式返回，也兼容 `<think>` 标签和 `reasoning_content`。
- `llm_config.py`：读取模型相关环境变量。
- `parser.py`：解析模型输出中的 `Thought`、`Action`、`Answer`。
- `tools.py`：注册并执行 Agent 工具。
- `prompts.py`：系统提示词模板。
- `schema.py`：`ReactStep` 和 `AgentResult` 等内部结构。
- `demo.py`：命令行入口。

## 当前工具

目前注册了两个通用内置工具：

- `query_all_devices()`：查询所有设备状态，包括灯光、空调、洗衣机、窗帘、传感器和智能插座。
- `control_device(device_id, command, params)`：控制指定设备；传感器属于只读设备，只能查询状态。

这两个工具都会通过 `environment.remote_adapter.RemoteEnvironmentAdapter` 调用环境服务。

单元测试可以把 `ToolRegistry` 的 `adapter` 换成 `environment.adapter.InMemoryEnvironmentAdapter`，直接调用内存中的 `SmartHomeEnv`。这样测试 Agent 工具和控制器时不需要启动 FastAPI 环境服务。

```python
from agent.tools import ToolRegistry
from environment.adapter import InMemoryEnvironmentAdapter
from environment.scenarios import build_evening_home_devices
from environment.smart_home_env import SmartHomeEnv

env = SmartHomeEnv(device_factory=build_evening_home_devices)
tools = ToolRegistry(adapter=InMemoryEnvironmentAdapter(env=env))
tools.adapter.create_session("test-session")
```

`agent` 包的顶层导出和 `ToolRegistry` 默认 HTTP adapter 都使用懒加载，导入 `agent.tools` 时不会提前加载模型客户端或 HTTP 客户端。

## 测试

Agent 工具层测试位于 `tests/agent_tests/`，使用标准库 `unittest`：

```bash
python -m unittest discover -s tests
```

当前测试通过 `InMemoryEnvironmentAdapter` 验证工具查询、通用设备控制、新增设备控制和错误返回，并用假模型客户端验证控制器的直接回答、工具调用后回答、流式工具调用和异常兜底，不需要启动环境服务或模型服务。

## 模型配置

默认配置定义在 `agent.llm_config.LLMConfig`：

- 接口地址：`http://127.0.0.1:11434/v1/chat/completions`
- 模型名：`qwen3.5:4b`
- 后端类型：`openai_compatible_remote`

可通过环境变量覆盖，例如：

- `AGENT_MODEL_CHAT_COMPLETIONS_URL`
- `AGENT_MODEL_NAME`
- `AGENT_MODEL_API_KEY`
- `AGENT_MODEL_TIMEOUT_SECONDS`
- `AGENT_MODEL_ENABLE_THINKING`
- `AGENT_MODEL_THINKING_BUDGET`

## 运行方式

显示帮助：

```bash
python agent/demo.py --help
```

单轮模式：

```bash
python agent/demo.py "帮我打开客厅灯"
```

交互模式：

```bash
python agent/demo.py
```

详细模式：

```bash
python agent/demo.py -v
```

## 依赖前提

运行 Agent 前需要准备两部分外部依赖：

1. 环境服务已经启动。

```bash
python -m environment.server
```

2. 一个兼容 OpenAI `/v1/chat/completions` 的模型服务已经启动。

如果使用默认配置，典型方式是本地启动 Ollama 并准备：

```bash
ollama run qwen3.5:4b
```

## 当前限制

- 当前工具仍是查询和控制两个通用入口，尚未拆分成设备专用工具。
- 仍依赖外部模型服务，仓库不包含模型本体。
