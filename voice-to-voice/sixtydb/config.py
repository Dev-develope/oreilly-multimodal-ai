"""Shared configuration and HTTP helpers for the 60dB plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass

import aiohttp
from livekit.agents import utils

DEFAULT_BASE_URL = "https://api.60db.ai/v1"


@dataclass
class SixtyDBConfig:
    """Resolved connection settings for the 60dB API.

    Values are pulled from explicit constructor arguments first, then from
    environment variables, mirroring how the openai/deepgram plugins resolve
    their API keys.
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def resolve(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "SixtyDBConfig":
        api_key = api_key or os.environ.get("SIXTYDB_API_KEY")
        if not api_key:
            raise ValueError(
                "60dB API key is required: pass api_key=... or set the "
                "SIXTYDB_API_KEY environment variable."
            )
        base_url = (
            base_url
            or os.environ.get("SIXTYDB_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        return cls(api_key=api_key, base_url=base_url)

    @property
    def auth_headers(self) -> dict[str, str]:
        # 60dB API: bearer-token auth. Change here if 60dB uses a custom
        # header (e.g. "x-60db-api-key") instead of Authorization.
        return {"Authorization": f"Bearer {self.api_key}"}

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def ws_url(self, path: str) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1]
        return f"{scheme}://{host}/{path.lstrip('/')}"


def http_session() -> aiohttp.ClientSession:
    """Reuse the agent's shared aiohttp session when running inside a job.

    ``utils.http_context.http_session()`` returns the session LiveKit manages
    for the worker, so we don't leak connectors per request.
    """
    return utils.http_context.http_session()
