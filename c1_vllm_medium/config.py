"""Configuration with C1 overrides and compatibility with the shared VLLM .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from c1_vllm_medium.errors import ConfigurationError

DEFAULT_MODEL = "v-llm-v1-medium"
DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def read_dotenv(path: Path) -> dict[str, str]:
    """Read a small KEY=VALUE .env file without adding a runtime dependency."""
    if not path.exists():
        return {}
    if not path.is_file():
        raise ConfigurationError(f"Environment file is not a regular file: {path}")
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ConfigurationError(f"Invalid environment file line {number}")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


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

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("https://", "http://")):
            raise ConfigurationError("C1_BASE_URL must begin with https:// or http://")
        if not all((self.client_id, self.client_secret, self.project_id, self.model)):
            raise ConfigurationError(
                "C1_BASE_URL, C1_CLIENT_ID, C1_CLIENT_SECRET and C1_PROJECT_ID are required"
            )
        if self.timeout_seconds <= 0 or self.max_tokens < 64 or self.max_retries < 1:
            raise ConfigurationError("C1 timeout, max tokens or retry settings are invalid")

    @property
    def service_url(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")


def load_settings(
    *,
    dotenv_path: Path = DEFAULT_DOTENV_PATH,
    overrides: dict[str, str | float | int | None] | None = None,
) -> Settings:
    file_values = read_dotenv(dotenv_path)
    overrides = overrides or {}

    def value(c1_name: str, shared_name: str | None = None, default: str = "") -> str:
        candidate = overrides.get(c1_name)
        if candidate is not None:
            return str(candidate).strip()
        for name in (c1_name, shared_name):
            if name and name in os.environ:
                return os.environ[name].strip()
        for name in (c1_name, shared_name):
            if name and name in file_values:
                return file_values[name].strip()
        return default

    try:
        return Settings(
            base_url=value("C1_BASE_URL", "VLLM_BASE_URL"),
            client_id=value("C1_CLIENT_ID", "VLLM_CLIENT_ID"),
            client_secret=value("C1_CLIENT_SECRET", "VLLM_CLIENT_SECRET"),
            project_id=value("C1_PROJECT_ID", "VLLM_PROJECT_ID"),
            # C1 deliberately does not inherit VLLM_MODEL: the shared value is Large.
            model=value("C1_MODEL", default=DEFAULT_MODEL),
            timeout_seconds=float(value("C1_TIMEOUT_SECONDS", "VLLM_TIMEOUT_SECONDS", "120")),
            max_tokens=int(value("C1_MAX_TOKENS", "VLLM_MAX_TOKENS", "1024")),
            max_retries=int(value("C1_MAX_RETRIES", "VLLM_MAX_RETRIES", "3")),
        )
    except ValueError as exc:
        raise ConfigurationError("C1 numeric configuration is invalid") from exc
