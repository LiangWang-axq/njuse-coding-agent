from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _find_dotenv(start: Path) -> Path | None:
    """Find the nearest .env in ``start`` or any ancestor (up to 6 levels)."""
    current = start.resolve()
    for _ in range(6):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 90
    max_steps: int = 24
    retries: int = 2
    context_chars: int = 16_000


def load_settings(workspace: Path | None = None) -> Settings:
    root = (workspace or Path.cwd()).resolve()
    dotenv_path = _find_dotenv(root)
    dotenv = _load_dotenv(dotenv_path) if dotenv_path else {}

    def value(name: str, default: str = "") -> str:
        return os.environ.get(name, dotenv.get(name, default))

    return Settings(
        base_url=value("AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        api_key=value("AGENT_API_KEY") or value("OPENAI_API_KEY"),
        model=value("AGENT_MODEL", "deepseek-chat"),
        timeout_seconds=int(value("AGENT_TIMEOUT_SECONDS", "90")),
        max_steps=int(value("AGENT_MAX_STEPS", "24")),
        retries=int(value("AGENT_RETRIES", "2")),
        context_chars=int(value("AGENT_CONTEXT_CHARS", "16000")),
    )
