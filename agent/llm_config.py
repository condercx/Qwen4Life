"""通用 LLM 配置定义与环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_ENV_FILES_LOADED = False


@dataclass(slots=True)
class LLMConfig:
	"""模型调用配置。当前默认走硅基流动在线接口，后续可扩展到本地模型。"""

	provider: str = "siliconflow"
	backend: str = "openai_compatible_remote"
	api_key: str | None = None
	chat_completions_url: str = "https://api.siliconflow.cn/v1/chat/completions"
	model: str = "Qwen/Qwen3.5-4B"
	timeout_seconds: int = 60
	temperature: float = 0.1
	top_p: float = 0.7
	max_tokens: int = 10240
	n: int = 1
	force_json_output: bool = True
	enable_thinking: bool | None = None
	thinking_budget: int | None = None

	@classmethod
	def from_env(cls) -> "LLMConfig":
		"""从环境变量读取配置。"""

		_load_default_env_files()

		provider = os.getenv("AGENT_MODEL_PROVIDER", "siliconflow")
		backend = os.getenv("AGENT_MODEL_BACKEND", "openai_compatible_remote")
		api_key = os.getenv("AGENT_MODEL_API_KEY")
		chat_completions_url = os.getenv(
			"AGENT_MODEL_CHAT_COMPLETIONS_URL",
			"https://api.siliconflow.cn/v1/chat/completions",
		)
		model = os.getenv("AGENT_MODEL_NAME", "Qwen/Qwen3.5-4B")
		return cls(
			provider=provider,
			backend=backend,
			api_key=api_key,
			chat_completions_url=chat_completions_url,
			model=model,
			timeout_seconds=int(os.getenv("AGENT_MODEL_TIMEOUT_SECONDS", "60")),
			temperature=float(os.getenv("AGENT_MODEL_TEMPERATURE", "0.1")),
			top_p=float(os.getenv("AGENT_MODEL_TOP_P", "0.7")),
			max_tokens=int(os.getenv("AGENT_MODEL_MAX_TOKENS", "10240")),
			n=int(os.getenv("AGENT_MODEL_N", "1")),
			force_json_output=os.getenv("AGENT_MODEL_FORCE_JSON_OUTPUT", "true").lower() in {"1", "true", "yes", "on"},
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
