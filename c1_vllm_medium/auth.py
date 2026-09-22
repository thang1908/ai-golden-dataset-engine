"""OAuth2 Client Credentials token cache for C1."""

from __future__ import annotations

import time

import httpx

from c1_vllm_medium.config import Settings
from c1_vllm_medium.errors import AuthenticationError


class TokenProvider:
    def __init__(self, settings: Settings, client: httpx.Client) -> None:
        self._settings = settings
        self._client = client
        self._value: str | None = None
        self._expires_at = 0.0

    def invalidate(self) -> None:
        self._value = None
        self._expires_at = 0.0

    def get(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._value and time.monotonic() < self._expires_at - 60:
            return self._value
        try:
            response = self._client.post(
                f"{self._settings.service_url}/oauth/token",
                headers={"Content-Type": "application/json"},
                json={
                    "client_id": self._settings.client_id,
                    "client_secret": self._settings.client_secret,
                    "project_id": self._settings.project_id,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthenticationError("Unable to obtain an OAuth access token") from exc
        token, expires_in = payload.get("access_token"), payload.get("expires_in")
        if not isinstance(token, str) or not token or not isinstance(expires_in, (int, float)):
            raise AuthenticationError("OAuth response is missing a valid access token")
        self._value = token
        self._expires_at = time.monotonic() + float(expires_in)
        return token
