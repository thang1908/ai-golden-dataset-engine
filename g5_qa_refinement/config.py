"""Read shared repository VLLM settings; G5 remains Medium by default."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

DEFAULT_MODEL = "v-llm-v1-medium"
DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


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


def _boolean(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError("invalid boolean")


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str
    client_id: str
    client_secret: str
    project_id: str
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120
    max_tokens: int = 1024
    max_retries: int = 3
    max_requests_per_minute: int = 150
    enable_thinking: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("https://", "http://")):
            raise ConfigurationError("VLLM_BASE_URL must begin with https:// or http://")
        if not all((self.client_id, self.client_secret, self.project_id, self.model)):
            raise ConfigurationError(
                "VLLM_BASE_URL, VLLM_CLIENT_ID, VLLM_CLIENT_SECRET and VLLM_PROJECT_ID are required"
            )
        if self.timeout_seconds <= 0 or self.max_tokens < 64 or self.max_retries < 1 or self.max_requests_per_minute < 1:
            raise ConfigurationError("G5 timeout, max tokens, retries or request limit are invalid")

    @property
    def service_url(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")


def load_settings(
    *, dotenv_path: Path = DEFAULT_DOTENV_PATH, overrides: dict[str, object] | None = None
) -> Settings:
    values, overrides = _dotenv(dotenv_path), overrides or {}

    def value(name: str, default: str = "") -> str:
        if overrides.get(name) is not None:
            return str(overrides[name]).strip()
        for source in (os.environ, values):
            if name in source:
                return source[name].strip()
        return default

    try:
        return Settings(
            base_url=value("VLLM_BASE_URL"),
            client_id=value("VLLM_CLIENT_ID"),
            client_secret=value("VLLM_CLIENT_SECRET"),
            project_id=value("VLLM_PROJECT_ID"),
            # Shared VLLM_MODEL may target Large, so G5 never reads it.
            model=value("G5_MODEL", default=DEFAULT_MODEL),
            timeout_seconds=float(value("VLLM_TIMEOUT_SECONDS", "120")),
            max_tokens=int(value("VLLM_MAX_TOKENS", "1024")),
            max_retries=int(value("VLLM_MAX_RETRIES", "3")),
            max_requests_per_minute=int(value("VLLM_MAX_REQUESTS_PER_MINUTE", "150")),
            enable_thinking=_boolean(value("VLLM_ENABLE_THINKING", "true")),
        )
    except ValueError as exc:
        raise ConfigurationError("G5 numeric configuration is invalid") from exc

