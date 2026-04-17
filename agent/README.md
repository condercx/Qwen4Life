# Agent — 智能家居 ReAct 控制器

基于 **ReAct**（Reasoning + Acting）范式的智能家居对话代理。通过本地大模型理解自然语言指令，自主推理并调用工具控制家中设备，最终以流式自然语言回复用户。

## 工作原理

Agent 在每一轮对话中执行如下循环，直到生成最终回复或达到步数上限：

```
用户输入 → Thought（思考意图）→ Action（调用工具）→ Observation（接收结果）→ … → Answer（回复用户）
```

- 用户说「我有点冷」→ 模型先查询空调状态，发现关着 → 开空调 → 调温度 → 回复「已经帮你打开空调并调到 26 度」
- 简单问候不需要工具调用，模型直接给出 Answer

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Ollama

下载 [Ollama](https://ollama.com) 并拉取模型：

```bash
ollama run qwen3.5:4b
```

Ollama 会在后台持续监听 `http://127.0.0.1:11434`。

### 3. 配置环境变量

复制配置模板并按需修改：

```bash
cp agent/.env.example agent/.env
```

默认配置即可对接本地 Ollama，无需额外修改。如需切换模型，只需改 `AGENT_MODEL_NAME`。

### 4. 启动环境服务

在一个终端中运行模拟环境：

```bash
python -m environment.server
```

### 5. 启动 Agent

另开一个终端，启动交互式对话：

```bash
# 普通模式 — 只显示最终回复
python -m agent.demo

# 详细模式 — 显示完整推理链和工具调用过程
python -m agent.demo -v
```

也支持单轮调用：

```bash
python -m agent.demo "帮我把客厅灯打开"
```

## 交互示例

```
用户> 我热了

助手>
[💭 思考中...] 用户觉得热，我先查一下空调状态。
[🎯 确定行动...]
🔧 [调用工具: query_all_devices({})]
📋 观测结果: [客厅空调] 状态：关闭，模式 制冷，目标温度 26.0°C

助手>
[💭 思考中...] 空调关着，帮用户打开并适当调低温度。
[🎯 确定行动...]
🔧 [调用工具: control_device(device_id="living_room_ac_1", command="turn_on")]
📋 观测结果: 操作成功。

助手> 空调已经帮你打开了，目前是制冷模式 26°C，如果还觉得热我再帮你调低。
```

> 以上为 `-v`（详细模式）输出。普通模式只显示最终回复。

## 目录结构

| 文件 | 职责 |
|------|------|
| `controller.py` | ReAct 循环核心，协调模型调用、工具执行和对话历史管理 |
| `llm_client.py` | OpenAI 兼容 HTTP 客户端，支持流式输出和 `<think>` 标签解析 |
| `llm_config.py` | 从 `.env` 文件和环境变量读取模型配置（采样参数、端点等） |
| `parser.py` | 从模型输出文本中提取 `Thought` / `Action` / `Answer` 结构 |
| `prompts.py` | System prompt 模板，定义 ReAct 格式约定和设备清单 |
| `tools.py` | 工具注册中心，将 `query_all_devices` / `control_device` 桥接到环境服务 |
| `schema.py` | 内部数据结构定义（`ReactStep`、`AgentResult`） |
| `demo.py` | CLI 入口，渲染流式打字机效果 |

## 扩展指南

**添加新工具**：在 `tools.py` 的 `_register_builtin_tools()` 中调用 `_register()` 注册即可。只需提供工具名、描述、参数说明和处理函数，无需修改其他文件。

**切换模型**：修改 `.env` 中的 `AGENT_MODEL_NAME` 和 `AGENT_MODEL_CHAT_COMPLETIONS_URL`。任何兼容 OpenAI `/v1/chat/completions` 接口的服务都可以直接对接。

**调整推理行为**：`.env` 中的 `AGENT_MODEL_ENABLE_THINKING` 和 `AGENT_MODEL_THINKING_BUDGET` 控制思考模式（需模型支持）。
