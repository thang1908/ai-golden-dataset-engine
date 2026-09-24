from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_MODEL = "gemini-2.5-pro"


def _dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ConfigurationError(f"Environment file is not a regular file: {path}")
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ConfigurationError(f"Invalid environment file line {number}")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120.0
    max_tokens: int = 4096
    max_retries: int = 5

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigurationError("GEMINI_API_KEY is required")
        if not self.model:
            raise ConfigurationError("GEMINI_MODEL must not be empty")
        if self.timeout_seconds <= 0 or self.max_tokens < 256 or self.max_retries < 0:
            raise ConfigurationError("Gemini timeout, max tokens, or retry count is invalid")


def load_settings(
    *, dotenv_path: Path = DEFAULT_DOTENV_PATH, overrides: dict[str, object] | None = None
) -> Settings:
    dotenv, overrides = _dotenv(dotenv_path), overrides or {}

    def value(name: str, default: str = "") -> str:
        if overrides.get(name) is not None:
            return str(overrides[name]).strip()
        for source in (os.environ, dotenv):
            if name in source:
                return source[name].strip()
        return default

    try:
        return Settings(
            api_key=value("GEMINI_API_KEY"),
            model=value("GEMINI_MODEL", DEFAULT_MODEL),
            timeout_seconds=float(value("GEMINI_TIMEOUT_SECONDS", "120")),
            max_tokens=int(value("GEMINI_MAX_TOKENS", "4096")),
            max_retries=int(value("GEMINI_MAX_RETRIES", "5")),
        )
    except ValueError as exc:
        raise ConfigurationError("Gemini numeric configuration is invalid") from exc
