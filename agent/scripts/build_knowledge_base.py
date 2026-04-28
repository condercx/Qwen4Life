"""构建本地儿童教育知识库。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.request import urlopen

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agent.knowledge_config import KnowledgeConfig
from agent.knowledge_store import ChromaKnowledgeStore, KnowledgeChunk, build_grimms_chunks

GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/2591/pg2591.txt"


def main() -> None:
    """下载或读取格林童话文本，并写入本地 ChromaDB。"""

    args = _parse_args()
    config = KnowledgeConfig.from_env()
    source_path = Path(args.source_path or config.source_path)
    if source_path.is_absolute():
        raise RuntimeError("知识库源文本路径必须使用相对路径。")

    if not source_path.exists() or args.download:
        _download_source(source_path)

    raw_text = source_path.read_text(encoding="utf-8")
    chunks = build_grimms_chunks(
        raw_text,
        source=config.source_url,
        chunk_chars=config.chunk_chars,
        chunk_overlap=config.chunk_overlap,
    )
    store = ChromaKnowledgeStore(config)
    if args.clear:
        _preflight_embedding(store, chunks)
        store.clear()
    store.add_chunks(chunks)
    print(f"已写入知识库片段：{len(chunks)}")
    print(f"ChromaDB 路径：{config.chroma_path}")


def _download_source(source_path: Path) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(GUTENBERG_TEXT_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    source_path.write_text(text, encoding="utf-8")


def _preflight_embedding(store: ChromaKnowledgeStore, chunks: list[KnowledgeChunk]) -> None:
    if not chunks:
        return
    store.embedding_client.embed([store._text_for_embedding(chunks[0].text)])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建格林童话儿童教育知识库")
    parser.add_argument("--source-path", help="格林童话 UTF-8 文本路径，默认读取 AGENT_KB_SOURCE_PATH")
    parser.add_argument("--download", action="store_true", help="重新从 Project Gutenberg 下载源文本")
    parser.add_argument("--no-clear", dest="clear", action="store_false", help="写入前不清空 collection")
    parser.set_defaults(clear=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
