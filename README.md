# Qwen4Life

Qwen4Life 是一个面向智能家居场景的最小可运行 Agent + Environment 示例项目。

当前代码分成两个核心部分：

- `environment/`：模拟智能家居环境，负责设备状态、动作执行、事件流转和 HTTP 服务。
- `agent/`：实现基于 ReAct 的智能家居 Agent，负责理解用户输入、决定是否调用工具，并把结果整理成自然语言回复。

根目录 `README.md`、`agent/README.md` 和 `environment/README.md` 均按当前代码同步维护。

## 当前功能

- 提供 6 类默认设备：灯光、空调、洗衣机、窗帘、温湿度传感器、智能插座。
- 支持设备控制、状态查询和事件读取。
- 支持洗衣机这类带时间推进的设备任务。
- 提供独立的 FastAPI 环境服务，Agent 通过 HTTP 调用环境。
- 提供命令行 Agent demo，支持单轮模式和交互模式。
- 环境核心支持内存适配器和可控时间，便于编写不依赖服务进程的单元测试。
- 支持可选儿童教育 RAG 知识库，当前示例语料为 Project Gutenberg 的《格林童话》。
- 支持可选 markdown 长期记忆，由 Agent 自己判断是否记录用户偏好、别名、习惯、家庭规则和历史约定。

## 项目结构

```text
Qwen4Life
├── agent
│   ├── __init__.py
│   ├── controller.py
│   ├── demo.py
│   ├── embedding_client.py
│   ├── knowledge_base.py
│   ├── knowledge_config.py
│   ├── knowledge_store.py
│   ├── llm_client.py
│   ├── llm_config.py
│   ├── memory.py
│   ├── memory_config.py
│   ├── parser.py
│   ├── prompts.py
│   ├── schema.py
│   └── tools.py
├── environment
│   ├── __init__.py
│   ├── actions.py
│   ├── adapter.py
│   ├── clock.py
│   ├── devices.py
│   ├── remote_adapter.py
│   ├── scenarios.py
│   ├── server.py
│   └── smart_home_env.py
├── data
│   └── knowledge
│       └── grimms_fairy_tales.txt
├── tests
│   ├── agent_tests
│   ├── environment_tests
│   └── README.md
├── requirements.txt
└── README.md
```

## 核心模块说明

### `environment/`

- `actions.py`：定义环境动作协议、请求校验和统一错误响应。
- `adapter.py`：定义 Agent 访问环境的适配器协议，并提供单元测试用的内存适配器。
- `clock.py`：定义环境时间源，生产环境使用真实时间，测试可注入可控时间。
- `devices.py`：定义灯光、空调、洗衣机、窗帘、传感器、智能插座等设备模型及其命令处理逻辑。
- `scenarios.py`：构造可复用设备 fixture，并提供 fixture 注册表。
- `smart_home_env.py`：环境核心，负责会话管理、状态观测、事件派发和 `step()` 执行。
- `server.py`：暴露 FastAPI 服务。
- `remote_adapter.py`：给 Agent 用的 HTTP 适配层。

### `agent/`

- `controller.py`：ReAct 主循环，负责会话上下文、模型调用、工具执行和 fallback 处理。
- `llm_client.py`：OpenAI 兼容接口客户端，支持非流式和流式输出。
- `llm_config.py`：从环境变量或 `.env` 读取模型配置。
- `parser.py`：解析模型输出中的 `Thought / Action / Answer`。
- `tools.py`：注册并执行 Agent 工具，当前包含设备查询和设备控制。
- `knowledge_config.py` / `embedding_client.py` / `knowledge_store.py` / `knowledge_base.py`：可选儿童教育 RAG 知识库配置、Ollama `bge-m3` embedding、ChromaDB + BM25 + query expansion 检索和知识库门面。
- `memory_config.py` / `memory.py`：可选 markdown 长期记忆配置和 AgentMemory 门面。
- `prompts.py`：系统提示词模板。
- `demo.py`：Agent 命令行入口。

### `tests/`

- `environment_tests/`：覆盖设备 fixture、环境状态流转、可控时间、错误响应和 FastAPI 服务边界。
- `agent_tests/`：覆盖 Agent 工具层、控制器和内存环境适配器的集成。
- `README.md`：记录测试运行方式和测试边界。

## 依赖

当前根依赖如下：

```text
fastapi
uvicorn
httpx
chromadb
```

安装方式：

```bash
pip install -r requirements.txt
```

说明：

- 当前实际依赖以根目录 `requirements.txt` 为准。
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

HTTP 动作接口也兼容 `name` / `args` 别名，服务层会归一化为环境核心使用的 `command` / `params`。

## 默认设备

- `living_room_light_1`：客厅主灯，支持开关和亮度调节。
- `living_room_ac_1`：客厅空调，支持模式、温度和风速控制。
- `washing_machine_1`：阳台洗衣机，支持启动、暂停、继续和取消。
- `living_room_curtain_1`：客厅窗帘，支持打开、关闭和开合度设置。
- `living_room_sensor_1`：客厅温湿度传感器，只支持状态查询。
- `desk_plug_1`：书房插座，支持开关和功率读数。

## Agent 工具

当前 Agent 默认内置 2 个基础工具；启用儿童教育知识库或长期记忆后会按需注册额外工具：

- `query_all_devices()`：查询所有设备状态。
- `control_device(device_id, command, params)`：执行指定设备控制。
- `search_knowledge_base(query)`：查询本地儿童教育 RAG 知识库，当前可检索《格林童话》。
- `list_memories()`：列出当前用户的长期记忆和可删除 ID。
- `delete_memory(memory_id)`：删除当前用户的一条长期记忆。
- `clear_user_memory()`：清空当前用户的全部长期记忆。

工具最终都通过 `environment.remote_adapter.RemoteEnvironmentAdapter` 转发到环境服务。

单元测试可以改用 `environment.adapter.InMemoryEnvironmentAdapter`，直接调用 `SmartHomeEnv`，不需要启动环境 HTTP 服务：

```python
from agent.tools import ToolRegistry
from environment.adapter import InMemoryEnvironmentAdapter
from environment.clock import FakeClock
from environment.scenarios import build_evening_home_devices
from environment.smart_home_env import SmartHomeEnv

clock = FakeClock(current_time=1_700_000_000)
env = SmartHomeEnv(clock=clock, device_factory=build_evening_home_devices)
tools = ToolRegistry(adapter=InMemoryEnvironmentAdapter(env=env))
tools.adapter.create_session("test-session")
result = tools.execute("test-session", "query_all_devices", {})
```

## 测试时间控制

环境核心支持注入 `environment.clock.FakeClock`，用于单元测试中推进洗衣机等计时设备，不需要真实 `sleep`：

```python
from environment.clock import FakeClock
from environment.smart_home_env import SmartHomeEnv

clock = FakeClock(current_time=1_700_000_000)
env = SmartHomeEnv(clock=clock)
env.reset("test-session")
clock.advance(10)
```

## 测试设备场景

`environment.scenarios` 当前提供 `default`、`evening_home`、`all_offline`、`washing_running`、`washing_paused` 五类设备 fixture。可以通过 `SmartHomeEnv(device_factory=...)` 注入，供 agent 工具和控制器单测复用。

## 测试

当前测试使用 Python 标准库 `unittest`，不需要新增第三方测试依赖。

```bash
python -m unittest discover -s tests
```

这些测试不会启动 uvicorn 进程，也不会连接模型服务；HTTP 服务边界通过 FastAPI `TestClient` 在进程内验证，Agent 控制器测试使用假模型客户端驱动非流式和流式 ReAct 输出。

## 本地验证

修改环境或 Agent 后建议运行以下命令，确保语法、设备行为、HTTP 边界和 Agent 工具链保持一致：

- `python -m compileall environment agent tests`
- `python -m unittest discover -s tests`
- `python agent/demo.py --help`

当前测试覆盖环境层 `reset -> step -> state`、新增设备模型、FastAPI 服务边界、`InMemoryEnvironmentAdapter` 工具调用，以及 `SimpleSmartHomeAgent` 的 fake LLM 非流式和流式控制器路径。

## 当前限制

- Agent 仍依赖外部模型服务，仓库本身不包含模型权重。
- 当前 Agent 仍只暴露查询和控制两个通用工具，尚未拆分成更细粒度的设备专用工具。

## 后续建议

- 继续扩展 `tests/`，覆盖更多异常输入和端到端 Agent 场景。
- 为 Agent 增加更多工具和更细粒度的设备语义。
- 为环境服务补充健康检查和更完整的 API 文档示例。

## 儿童教育知识库（可选）

`SimpleSmartHomeAgent` 支持可选本地 RAG 知识库，默认关闭。该能力对应 `question.md` 中的儿童教育陪伴场景：当用户询问格林童话、睡前故事、故事寓意或儿童陪伴讲解时，Agent 可以调用 `search_knowledge_base(query)` 查询本地知识库，再由 ReAct controller 生成最终回答。

当前示例语料是 Project Gutenberg 的 *Grimms' Fairy Tales*，源文本位于：

```text
data/knowledge/grimms_fairy_tales.txt
```

来源：<https://www.gutenberg.org/ebooks/2591>

知识库默认使用本地 Ollama 的 `bge-m3` embedding 模型，向量数据写入相对路径：

```text
.agent_kb/chroma
```

构建和启用方式：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull bge-m3
.\.venv\Scripts\python.exe -m agent.scripts.build_knowledge_base
$env:AGENT_KB_ENABLED = "true"
.\.venv\Scripts\python.exe agent/demo.py
```

知识库检索会合并 Chroma 向量检索、BM25 关键词检索和本地 query expansion。query expansion 不额外调用 LLM，主要用于把“小红帽、灰姑娘、睡美人”等中文查询扩展到英文原文标题和关键词。

可选环境变量：

```text
AGENT_KB_ENABLED=false
AGENT_KB_EMBED_BACKEND=ollama
AGENT_KB_OLLAMA_EMBED_URL=http://127.0.0.1:11434/api/embed
AGENT_KB_EMBED_MODEL=bge-m3
AGENT_KB_CHROMA_PATH=.agent_kb/chroma
AGENT_KB_COLLECTION=children_education
AGENT_KB_SOURCE_PATH=data/knowledge/grimms_fairy_tales.txt
AGENT_KB_TOP_K=5
AGENT_KB_MIN_SCORE=0.0
AGENT_KB_EMBED_TEXT_CHARS=120
```

`.agent_kb/` 是本地生成的向量库，可以删除后重建，不应提交到 git。

## Markdown 长期记忆（可选）

长期记忆现在使用 markdown 文件记录，默认关闭。它不再使用 ChromaDB，也不再做向量检索；它的职责是保存少量可审计的用户偏好、设备别名、习惯、家庭规则和历史约定。

默认路径：

```text
.agent_memory/profile/{user_id}.md
```

启用方式：

```powershell
$env:AGENT_MEMORY_ENABLED = "true"
.\.venv\Scripts\python.exe agent/demo.py
```

可选环境变量：

```text
AGENT_MEMORY_ENABLED=false
AGENT_MEMORY_DIR=.agent_memory/profile
AGENT_MEMORY_MAX_CONTEXT_ITEMS=20
```

保存策略：

- Agent 在 ReAct 主循环里通过 `save_memory(memory_type, memory_text)` 工具保存长期记忆，不再额外开启一次隐藏的保存决策对话。
- 只有用户明确表达长期有价值的信息时才写入 markdown。
- 可保存内容包括用户偏好、设备别名、习惯、家庭规则和历史约定，也包括儿童陪伴中的长期偏好，例如睡前故事语气或内容禁忌。
- 普通寒暄、一次性设备操作、实时设备状态、知识库故事正文、fallback、异常和空回答不会写入长期记忆。
- 实际保存的是 Agent 提炼出的简洁中文事实，不保存 `Thought`、`Action`、`Observation` 或 raw messages。

启用长期记忆后，Agent 会注册 4 个记忆管理工具：

- `save_memory(memory_type, memory_text)`：保存一条当前用户长期记忆。
- `list_memories()`：列出当前用户的长期记忆和可删除 ID。
- `delete_memory(memory_id)`：删除当前用户的一条长期记忆。
- `clear_user_memory()`：清空当前用户的全部长期记忆。

实时设备状态和控制结果始终以工具 `Observation` 为准；如果长期记忆与工具结果冲突，以工具结果为准。长期记忆是本地 markdown 文件，不是模型内部记忆。

Windows 本地测试或运行时优先使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m compileall environment agent tests
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
