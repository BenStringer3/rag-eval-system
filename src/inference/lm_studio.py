"""LM Studio OpenAI-compatible client factory."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LMStudioSettings:
    """Connection settings for LM Studio's /v1 API."""

    base_url: str
    api_key: str

    @classmethod
    def from_config(cls, cfg: dict) -> LMStudioSettings:
        lm = cfg["lm_studio"]
        return cls(base_url=lm["base_url"], api_key=lm["api_key"])


def openai_client(settings: LMStudioSettings) -> OpenAI:
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)
