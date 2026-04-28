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
- `knowledge_config.py` / `embedding_client.py` / `knowledge_store.py` / `knowledge_base.py`：可选儿童教育 RAG 知识库，使用 Ollama `bge-m3`、ChromaDB、BM25 和 query expansion。
- `memory_config.py` / `memory.py`：可选 markdown 长期记忆，由 Agent 在 ReAct 主循环里按需调用工具保存。
- `schema.py`：`ReactStep` 和 `AgentResult` 等内部结构。
- `demo.py`：命令行入口。

## 当前工具

默认注册两个通用设备工具；启用儿童教育知识库或长期记忆后会按需注册额外工具：

- `query_all_devices()`：查询所有设备状态，包括灯光、空调、洗衣机、窗帘、传感器和智能插座。
- `control_device(device_id, command, params)`：控制指定设备；传感器属于只读设备，只能查询状态。
- `search_knowledge_base(query)`：查询本地儿童教育知识库，当前语料为《格林童话》。
- `list_memories()`：列出当前用户的长期记忆和可删除 ID。
- `delete_memory(memory_id)`：删除当前用户的一条长期记忆。
- `clear_user_memory()`：清空当前用户的全部长期记忆。

设备工具会通过 `environment.remote_adapter.RemoteEnvironmentAdapter` 调用环境服务；知识库和长期记忆工具只访问本地文件或本地 ChromaDB。

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

## 儿童教育知识库

`SimpleSmartHomeAgent` 可以接入本地 RAG 知识库，但默认关闭。该功能对应儿童教育陪伴场景：用户询问格林童话、睡前故事、故事寓意或适合孩子的讲解方式时，Agent 可调用 `search_knowledge_base(query)` 检索本地知识库，再生成最终回答。

当前知识库源文本为 Project Gutenberg 的 *Grimms' Fairy Tales*：

```text
data/knowledge/grimms_fairy_tales.txt
```

来源：<https://www.gutenberg.org/ebooks/2591>

构建和启用示例：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull bge-m3
.\.venv\Scripts\python.exe -m agent.scripts.build_knowledge_base
$env:AGENT_KB_ENABLED = "true"
.\.venv\Scripts\python.exe agent/demo.py -v
```

默认配置：

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

检索侧会合并 Chroma 向量检索、BM25 和 query expansion。query expansion 是本地轻量规则，用于把“小红帽、灰姑娘、睡美人”等中文查询扩展到英文原文标题和关键词，不额外调用 LLM。

`.agent_kb/chroma` 是本地生成的 ChromaDB 数据目录，不应提交到 git，可以删除后重建。

## Markdown 长期记忆

长期记忆默认关闭，现在使用 markdown 文件记录，不再使用 ChromaDB。Agent 会在 ReAct 主循环里按需调用 `save_memory(memory_type, memory_text)`；只有用户偏好、设备别名、习惯、家庭规则和历史约定等长期有价值信息会写入记忆。

默认路径：

```text
.agent_memory/profile/{user_id}.md
```

默认配置：

```text
AGENT_MEMORY_ENABLED=false
AGENT_MEMORY_DIR=.agent_memory/profile
AGENT_MEMORY_MAX_CONTEXT_ITEMS=20
```

启用示例：

```powershell
$env:AGENT_MEMORY_ENABLED = "true"
.\.venv\Scripts\python.exe agent/demo.py -v
```

保存策略：

- 保存由 Agent 显式调用 `save_memory()` 工具完成，不再额外开启隐藏的 LLM 决策请求。
- 保存内容必须是简洁、可独立理解的中文事实。
- 可保存儿童陪伴中的长期偏好，例如“睡前故事不要太吓人”“孩子喜欢动物故事”。
- 不保存普通寒暄、一次性设备操作、实时设备状态、知识库故事正文、fallback、异常、空回答、`Thought`、`Action`、`Observation` 或 raw messages。
- 实时设备状态和控制结果必须以工具 `Observation` 为准；长期记忆不能覆盖实时状态。

启用长期记忆后会额外注册 4 个工具：

- `save_memory(memory_type, memory_text)`：保存一条当前用户长期记忆。
- `list_memories()`：列出当前用户的长期记忆和可删除 ID。
- `delete_memory(memory_id)`：删除当前用户的一条长期记忆。
- `clear_user_memory()`：清空当前用户的全部长期记忆。
