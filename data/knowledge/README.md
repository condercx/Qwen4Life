# Knowledge Data

本目录保存儿童教育知识库的可重建源文本。

当前包含：

- `grimms_fairy_tales.txt`：Project Gutenberg 的 *Grimms' Fairy Tales* UTF-8 文本，来源为 <https://www.gutenberg.org/ebooks/2591>。

该文本用于本地 RAG 知识库构建，默认不会把向量库提交到 git。构建后的 ChromaDB 默认位于：

```text
.agent_kb/chroma
```

重建命令：

```powershell
.\.venv\Scripts\python.exe -m agent.scripts.build_knowledge_base
```

如需重新下载源文本：

```powershell
.\.venv\Scripts\python.exe -m agent.scripts.build_knowledge_base --download
```
