# 智能家居最小 Agent

这个目录提供一个最小可运行的上层 agent，用于通过硅基流动在线模型把用户自然语言转换成环境动作，然后调用 environment 中的 adapter 执行。

当前第一版默认使用硅基流动在线接口，但代码结构已经按“可替换模型后端”组织，后续新增本地模型时，优先在 `llm_client.py` 中增加新的本地实现类，而不是修改控制器主流程。

## 目录结构

- `llm_config.py`：读取通用模型配置。
- `llm_client.py`：统一模型客户端接口与当前远程实现。
- `prompts.py`：系统提示词与用户提示词。
- `schema.py`：agent 内部使用的数据结构。
- `parser.py`：解析模型输出 JSON。
- `controller.py`：主流程编排。
- `demo.py`：命令行入口。
- `requirements.txt`：依赖说明。
- `.env.example`：环境变量示例。

## 工作流

1. 用户输入自然语言。
2. `controller.py` 调用硅基流动模型。
3. 模型只输出严格 JSON 计划。
4. `parser.py` 解析计划。
5. `controller.py` 调用 `environment/agent_adapter.py`。
6. environment 返回状态与事件。
7. agent 生成最终回复。

## 配置

复制 `.env.example` 中的变量到你的本地环境变量中，至少需要：

- `AGENT_MODEL_API_KEY`
- `AGENT_MODEL_NAME`

### 推荐环境变量

- `AGENT_MODEL_PROVIDER`：模型提供方，例如 `siliconflow`
- `AGENT_MODEL_BACKEND`：后端类型，当前支持 `openai_compatible_remote`
- `AGENT_MODEL_API_KEY`
- `AGENT_MODEL_CHAT_COMPLETIONS_URL`
- `AGENT_MODEL_NAME`
- `AGENT_MODEL_TIMEOUT_SECONDS`
- `AGENT_MODEL_TEMPERATURE`
- `AGENT_MODEL_TOP_P`
- `AGENT_MODEL_MAX_TOKENS`
- `AGENT_MODEL_N`
- `AGENT_MODEL_FORCE_JSON_OUTPUT`
- `AGENT_MODEL_ENABLE_THINKING`
- `AGENT_MODEL_THINKING_BUDGET`

### 当前远程请求与文档对齐点

当前 `llm_client.py` 参考硅基流动 Chat Completions 文档发送请求，包含这些核心字段：

- `model`
- `messages`
- `stream=false`
- `temperature`
- `top_p`
- `max_tokens`
- `n`
- `response_format={"type": "json_object"}`，用于强约束模型输出 JSON
- 可选 `enable_thinking`
- 可选 `thinking_budget`

另外会读取响应头里的 `x-siliconcloud-trace-id`，便于后续排查线上请求问题。

### PowerShell 代理 Workaround

如果你的 Windows 环境依赖 PAC/WPAD 自动代理，而 Python 标准库没有自动吃到 PAC 规则，可以先在当前 PowerShell 会话里手动设置代理，再运行 demo。

当前 `wpad.dat` 对普通外网请求的兜底规则是：`PROXY proxy-shz.intel.com:912`。对 `api.siliconflow.cn` 这类地址，最简单的 workaround 就是显式设置下面三个环境变量：

```powershell
$env:HTTP_PROXY = "http://proxy-shz.intel.com:912"
$env:HTTPS_PROXY = "http://proxy-shz.intel.com:912"
$env:ALL_PROXY = "http://proxy-shz.intel.com:912"
```

注意这里使用的是 `http://` 代理地址，即使目标接口本身是 HTTPS，请求也会通过 HTTP CONNECT 代理转发。

设置完成后，直接在同一个 PowerShell 会话里运行：

```powershell
python -m agent.demo "打开客厅灯"
```

如果只想验证链路是否打通，也可以先执行：

```powershell
Invoke-WebRequest `
	-Uri "https://api.siliconflow.cn/v1/chat/completions" `
	-Method POST `
	-ContentType "application/json" `
	-Body '{"model":"test","messages":[{"role":"user","content":"hi"}]}' `
	-TimeoutSec 30
```

如果不再需要这组代理变量，可以在当前 PowerShell 会话中清理：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

## 运行方式

单轮调用：

```bash
python -m agent.demo "打开客厅灯"
```

交互模式：

```bash
python -m agent.demo
```

指定会话：

```bash
python -m agent.demo --session-id my-session "把空调调到24度"
```

## 当前限制

1. 第一版只支持单轮单动作规划。
2. 模型输出必须是严格 JSON，否则 parser 会报错。
3. 复杂多步任务还没有拆解成真正的规划链。
4. 本地模型后端的接口位置已经预留，但实现还没接入。

## 后续接本地模型的建议

1. 在 `llm_client.py` 中新增 `LocalLLMClient`。
2. 让 `create_default_llm_client()` 按 `AGENT_MODEL_BACKEND=local` 返回本地实现。
3. 保持 `controller.py` 不变，让控制器继续只依赖 `LLMClient` 接口。
