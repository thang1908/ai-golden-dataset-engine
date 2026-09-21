"""Environment configuration for the remote OpenAI-compatible vLLM service."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    vllm_base_url: str
    vllm_client_id: str = Field(min_length=1)
    vllm_client_secret: SecretStr
    vllm_project_id: str = Field(min_length=1)
    vllm_model: str = Field(default="v-llm-v1-large", min_length=1)
    vllm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    vllm_max_retries: int = Field(default=3, ge=1, le=10)
    vllm_max_tokens: int = Field(default=1024, ge=64, le=8192)
    # Base64 counts toward the platform's 100k message-content limit.
    vllm_max_image_bytes: int = Field(default=60_000, ge=10_000, le=70_000)
    vllm_enable_thinking: bool = False
    vllm_guardrail: Literal["off", "low", "large"] = "off"
    vllm_guard_output_mode: Literal["refuse", "collect"] = "refuse"
    vllm_token_refresh_skew_seconds: int = Field(default=60, ge=0, le=3600)

    @field_validator("vllm_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("VLLM_BASE_URL must start with http:// or https://")
        if "your-vllm-host" in value or "<" in value or ">" in value:
            raise ValueError("Replace VLLM_BASE_URL with the real service host")
        return value

    @field_validator("vllm_client_id", "vllm_project_id")
    @classmethod
    def reject_placeholder(cls, value: str) -> str:
        if value.strip().lower() in {"replace-me", "your_client_id", "your_project_id"}:
            raise ValueError("Replace placeholder credentials in golden_dataset_harness/.env")
        return value

    @field_validator("vllm_client_secret")
    @classmethod
    def reject_secret_placeholder(cls, value: SecretStr) -> SecretStr:
        if value.get_secret_value().strip().lower() in {"replace-me", "your_client_secret"}:
            raise ValueError("Replace VLLM_CLIENT_SECRET in golden_dataset_harness/.env")
        return value

    @property
    def service_base_url(self) -> str:
        return self.vllm_base_url[:-3] if self.vllm_base_url.endswith("/v1") else self.vllm_base_url

    @property
    def openai_base_url(self) -> str:
        return f"{self.service_base_url}/v1"

    @property
    def oauth_url(self) -> str:
        return f"{self.service_base_url}/oauth/token"
