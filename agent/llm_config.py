"""通用 LLM 配置定义与环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_ENV_FILES_LOADED = False


@dataclass(slots=True)
class LLMConfig:
	"""模型调用配置。当前默认走集成在本地的 Ollama 接口。"""

	provider: str = "ollama"
	backend: str = "openai_compatible_remote"
	api_key: str | None = "ollama"
	chat_completions_url: str = "http://127.0.0.1:11434/v1/chat/completions"
	model: str = "qwen3.5:4b"
	timeout_seconds: int = 300
	temperature: float = 0.6
	top_p: float = 0.95
	top_k: int = 20
	min_p: float = 0.0
	presence_penalty: float = 0.0
	repetition_penalty: float = 1.0
	max_tokens: int = 4096
	n: int = 1
	force_json_output: bool = False
	enable_thinking: bool | None = None
	thinking_budget: int | None = None

	@classmethod
	def from_env(cls) -> "LLMConfig":
		"""从环境变量读取配置。"""

		_load_default_env_files()

		provider = os.getenv("AGENT_MODEL_PROVIDER", "ollama")
		backend = os.getenv("AGENT_MODEL_BACKEND", "openai_compatible_remote")
		api_key = os.getenv("AGENT_MODEL_API_KEY", "ollama")
		chat_completions_url = os.getenv(
			"AGENT_MODEL_CHAT_COMPLETIONS_URL",
			"http://127.0.0.1:11434/v1/chat/completions",
		)
		model = os.getenv("AGENT_MODEL_NAME", "qwen3.5:4b")
		return cls(
			provider=provider,
			backend=backend,
			api_key=api_key,
			chat_completions_url=chat_completions_url,
			model=model,
			timeout_seconds=int(os.getenv("AGENT_MODEL_TIMEOUT_SECONDS", "300")),
			temperature=float(os.getenv("AGENT_MODEL_TEMPERATURE", "0.6")),
			top_p=float(os.getenv("AGENT_MODEL_TOP_P", "0.95")),
			top_k=int(os.getenv("AGENT_MODEL_TOP_K", "20")),
			min_p=float(os.getenv("AGENT_MODEL_MIN_P", "0.0")),
			presence_penalty=float(os.getenv("AGENT_MODEL_PRESENCE_PENALTY", "0.0")),
			repetition_penalty=float(os.getenv("AGENT_MODEL_REPETITION_PENALTY", "1.0")),
			max_tokens=int(os.getenv("AGENT_MODEL_MAX_TOKENS", "4096")),
			n=int(os.getenv("AGENT_MODEL_N", "1")),
			force_json_output=os.getenv("AGENT_MODEL_FORCE_JSON_OUTPUT", "false").lower() in {"1", "true", "yes", "on"},
			enable_thinking=_optional_bool(os.getenv("AGENT_MODEL_ENABLE_THINKING")),
			thinking_budget=_optional_int(os.getenv("AGENT_MODEL_THINKING_BUDGET")),
		)


def _optional_bool(value: str | None) -> bool | None:
	"""解析可选布尔环境变量。"""

	if value is None or not value.strip():
		return None
	return value.lower() in {"1", "true", "yes", "on"}


def _optional_int(value: str | None) -> int | None:
	"""解析可选整型环境变量。"""

	if value is None or not value.strip():
		return None
	return int(value)


def _load_default_env_files() -> None:
	"""按约定自动加载 .env 文件，但不覆盖现有系统环境变量。"""

	global _ENV_FILES_LOADED
	if _ENV_FILES_LOADED:
		return

	candidates = [
		Path(__file__).resolve().parent / ".env",
		Path.cwd() / ".env",
	]
	loaded_paths: set[Path] = set()
	for env_path in candidates:
		resolved_path = env_path.resolve()
		if resolved_path in loaded_paths:
			continue
		loaded_paths.add(resolved_path)
		_load_env_file(resolved_path)

	_ENV_FILES_LOADED = True


def _load_env_file(env_path: Path) -> None:
	"""读取单个 .env 文件，并写入当前进程环境变量。"""

	if not env_path.is_file():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		if line.startswith("export "):
			line = line[len("export "):].strip()
		if "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if not key or key in os.environ:
			continue
		if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
			value = value[1:-1]
		os.environ[key] = value
