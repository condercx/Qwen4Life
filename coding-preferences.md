# Coding Preferences

- Python 代码优先保持当前文件已有风格，避免在同一文件里混用缩进、命名和结构。
- 新增代码保持轻量、可测试、依赖注入友好，优先复用现有 dataclass、Protocol、Fake client、InMemory adapter/store 习惯。
- 注释和文档优先使用中文；只在复杂逻辑前写有帮助的短注释，避免解释显而易见的代码。
- 不写死绝对路径、API key、模型名或服务 URL；配置走环境变量，并提供本地默认值。
- 运行 Python 时优先使用 `.\.venv\Scripts\python.exe`，测试优先跑 `compileall` 和 `unittest discover -s tests`。
- Agent 测试不依赖真实模型、HTTP 服务、Ollama、Chroma 或联网；使用 FakeLLMClient、InMemoryEnvironmentAdapter、InMemoryMemoryStore。
- 长期记忆默认关闭，默认 embedding 使用本地 Ollama `bge-m3`，默认 ChromaDB 相对路径是 `.agent_memory/chroma`。
- 长期记忆由 Agent 在成功回答后通过结构化 JSON 决策是否保存；只保存用户偏好、设备别名、习惯、家庭规则和历史约定等长期有价值信息。
- 长期记忆不保存 `Thought`、`Action`、`Observation` 或 raw messages；保存文本应是简洁、可独立理解的中文事实。
- 长期记忆检索优先保持轻量本地实现，当前组合向量检索、BM25、轻量 KG 实体/关系匹配和本地 query expansion。
- 实时设备状态和控制结果以工具 Observation 为准，长期记忆不能覆盖实时状态。
- Memory-plus-plus 只作为实验参考，不直接接入它的 benchmark 主流程、硬编码 key 或最终答案生成逻辑。
- 不提交真实用户记忆、ChromaDB 数据目录或本地运行产物。
