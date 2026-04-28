# 测试

测试套件使用 Python 标准库 `unittest`，不新增 pytest 依赖。

修改环境或 Agent 后，先做语法编译检查，再运行全部单元测试：

```bash
python -m compileall environment agent tests
```

```bash
python -m unittest discover -s tests
```

当前测试聚焦确定性的环境行为、新增设备模型、Agent 工具执行和 Agent 控制器主路径，不启动 uvicorn 服务，也不连接 LLM 服务。
环境 HTTP 边界使用 FastAPI `TestClient` 在进程内验证，避免测试依赖外部端口。

Agent 控制器测试使用假模型客户端按顺序返回 ReAct 文本，用于验证直接回答、工具调用后回答、流式工具调用和模型异常兜底。
儿童教育知识库测试使用 `InMemoryKnowledgeStore`，markdown 长期记忆测试使用临时相对目录；测试不会调用真实 Ollama、Chroma 或联网。

测试子包使用 `agent_tests`、`environment_tests` 这类名称，避免 `unittest discover` 遮蔽源码包 `agent` 和 `environment`。
