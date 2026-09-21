"""Async OAuth2 Client Credentials token cache for the vLLM platform."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from golden_dataset_harness.models.provider_settings import ProviderSettings


class OAuthError(RuntimeError):
    """The platform rejected credentials or returned an invalid token response."""


@dataclass(frozen=True)
class AccessToken:
    value: str
    expires_at: float


class OAuthTokenProvider:
    def __init__(
        self,
        settings: ProviderSettings,
        http_client: httpx.AsyncClient | None = None,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings
        self._client = http_client
        self._clock = clock
        self._token: AccessToken | None = None
        self._lock = asyncio.Lock()

    async def get_token(self, force_refresh: bool = False) -> str:
        async with self._lock:
            now = self._clock()
            skew = self.settings.vllm_token_refresh_skew_seconds
            if not force_refresh and self._token and now < self._token.expires_at - skew:
                return self._token.value
            self._token = await self._request_token(now)
            return self._token.value

    async def invalidate(self) -> None:
        async with self._lock:
            self._token = None

    async def _request_token(self, now: float) -> AccessToken:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.settings.vllm_timeout_seconds)
        try:
            response = await client.post(
                self.settings.oauth_url,
                json={
                    "client_id": self.settings.vllm_client_id,
                    "client_secret": self.settings.vllm_client_secret.get_secret_value(),
                    "project_id": self.settings.vllm_project_id,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OAuthError("Unable to obtain vLLM access token") from exc
        finally:
            if owns_client:
                await client.aclose()

        token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(token, str) or not token:
            raise OAuthError("OAuth response is missing access_token")
        if not isinstance(expires_in, (int, float)) or expires_in <= 0:
            raise OAuthError("OAuth response is missing a valid expires_in")
        return AccessToken(token, now + float(expires_in))
