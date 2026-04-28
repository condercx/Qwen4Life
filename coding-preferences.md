# Coding Preferences

- Python 代码优先保持当前文件已有风格，避免在同一文件里混用缩进、命名和结构。
- 新增代码保持轻量、可测试、依赖注入友好，优先复用现有 dataclass、Protocol、Fake client、InMemory adapter/store 习惯。
- 注释和文档优先使用中文；只在复杂逻辑前写有帮助的短注释，避免解释显而易见的代码。
- 不写死绝对路径、API key、模型名或服务 URL；配置走环境变量，并提供本地默认值。
- 运行 Python 时优先使用 `.\.venv\Scripts\python.exe`，测试优先跑 `compileall` 和 `unittest discover -s tests`。
- Agent 测试不依赖真实模型、HTTP 服务、Ollama、Chroma 或联网；使用 FakeLLMClient、InMemoryEnvironmentAdapter、InMemoryKnowledgeStore 和临时 markdown memory 目录。
- 儿童教育知识库和用户长期记忆要保持职责分离：知识库是 RAG tool，长期记忆是用户画像/约定的 markdown 文件。
- 知识库默认关闭，默认 embedding 使用本地 Ollama `bge-m3`，默认 ChromaDB 相对路径是 `.agent_kb/chroma`。
- 知识库检索可以组合向量检索、BM25 和本地 query expansion；不要把知识库故事正文写入用户长期记忆。
- 长期记忆默认关闭，默认相对路径是 `.agent_memory/profile/{user_id}.md`，用 markdown 保存，方便审计、手动编辑和删除。
- 长期记忆由 Agent 在 ReAct 主循环里显式调用 `save_memory()` 工具保存；不再额外开启隐藏的 LLM 保存决策请求。
- 长期记忆只保存用户偏好、设备别名、习惯、家庭规则和历史约定等长期有价值信息。
- 长期记忆不保存 `Thought`、`Action`、`Observation` 或 raw messages；保存文本应是简洁、可独立理解的中文事实。
- 实时设备状态和控制结果以工具 Observation 为准，长期记忆不能覆盖实时状态。
- Memory-plus-plus 只作为实验参考，不直接接入它的 benchmark 主流程、硬编码 key 或最终答案生成逻辑。
- 不提交真实用户记忆、ChromaDB 数据目录或本地运行产物。
