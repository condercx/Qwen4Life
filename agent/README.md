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
- `memory_config.py` / `memory_client.py` / `memory_store.py` / `memory_decision.py` / `memory.py`：可选长期记忆配置、Ollama embedding、ChromaDB + BM25 + KG 检索、保存决策和 AgentMemory 门面。
- `schema.py`：`ReactStep` 和 `AgentResult` 等内部结构。
- `demo.py`：命令行入口。

## 当前工具

默认注册两个通用设备工具；启用长期记忆后会额外注册三个记忆管理工具：

- `query_all_devices()`：查询所有设备状态，包括灯光、空调、洗衣机、窗帘、传感器和智能插座。
- `control_device(device_id, command, params)`：控制指定设备；传感器属于只读设备，只能查询状态。
- `list_memories()`：列出当前用户的长期记忆和可删除 ID。
- `delete_memory(memory_id)`：删除当前用户的一条长期记忆。
- `clear_user_memory()`：清空当前用户的全部长期记忆。

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

## 可选长期记忆

`SimpleSmartHomeAgent` 可以接入长期记忆，但默认关闭。启用后，每轮用户输入会先检索相关历史记忆，并把检索结果注入 system prompt；ReAct 主循环仍照常调用 `query_all_devices` 和 `control_device`。最终成功回答后，Agent 会额外调用一次 LLM 判断本轮是否值得保存，并输出结构化 JSON。只有 Agent 判断本轮包含长期价值时才写入记忆，不会保存 `Thought`、`Action`、`Observation` 或 raw messages。fallback、异常和空回答不会写入长期记忆。

长期记忆只用于用户偏好、习惯、设备别名和历史约定。实时设备状态、设备控制结果必须以工具返回的 `Observation` 为准；如果记忆与工具结果冲突，以工具结果为准。

默认本地配置使用 Ollama embedding API 和 `bge-m3` 模型：

```text
AGENT_MEMORY_ENABLED=false
AGENT_MEMORY_EMBED_BACKEND=ollama
AGENT_MEMORY_OLLAMA_EMBED_URL=http://127.0.0.1:11434/api/embed
AGENT_MEMORY_EMBED_MODEL=bge-m3
AGENT_MEMORY_CHROMA_PATH=.agent_memory/chroma
AGENT_MEMORY_COLLECTION=agent_memory
AGENT_MEMORY_TOP_K=5
AGENT_MEMORY_MIN_SCORE=0.0
```

启用示例：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull bge-m3
$env:AGENT_MEMORY_ENABLED = "true"
.\.venv\Scripts\python.exe agent/demo.py
```

记忆存储在相对路径 `.agent_memory/chroma`，这是本地 ChromaDB 数据目录，不是模型内部记忆。不同机器测试时可以删除该目录后重建，不应提交到 git。测试仍使用 fake/in-memory memory，不会调用 Ollama、Chroma 或联网。

当前保存策略只写入 Agent 判断有长期价值的信息：用户偏好、设备别名、习惯、家庭规则和历史约定。普通寒暄、一次性设备操作、实时设备状态、fallback、异常和空回答不会写入长期记忆。

检索侧会合并向量检索、BM25 关键词检索、轻量 KG 实体/关系匹配和 query expansion。query expansion 当前是本地轻量规则扩展，不额外调用 LLM。

启用长期记忆后会额外注册 3 个工具：

- `list_memories()`：列出当前用户的长期记忆和可删除 ID。
- `delete_memory(memory_id)`：删除当前用户的一条长期记忆。
- `clear_user_memory()`：清空当前用户的全部长期记忆。
