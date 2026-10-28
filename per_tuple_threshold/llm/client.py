"""Chat clients for the supported LLM providers.

Every provider exposes ``complete(messages, max_tokens, temperature) -> str``.
Credentials are read from the environment only:

* ``anthropic``: ``ANTHROPIC_AUTH_TOKEN`` or ``ANTHROPIC_API_KEY`` (required),
  ``ANTHROPIC_BASE_URL`` (optional, an Anthropic-compatible endpoint).
* ``openai``: ``OPENAI_API_KEY`` (required), ``OPENAI_BASE_URL`` (optional, an
  OpenAI-compatible endpoint such as ``https://api.camel-hub.com/v1``).
* ``local``: an OpenAI-compatible server you run yourself, e.g. vLLM.
  ``LLM_BASE_URL`` (required, e.g. ``http://localhost:8000/v1``), ``LLM_API_KEY``
  (optional; a placeholder is sent when unset, since vLLM does not check it).
"""

from __future__ import annotations

import os
from typing import Protocol

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_LOCAL = "local"
PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_LOCAL)

ANTHROPIC_KEY_ENVS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
OPENAI_KEY_ENVS = ("OPENAI_API_KEY",)
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
LOCAL_BASE_URL_ENV = "LLM_BASE_URL"
LOCAL_KEY_ENV = "LLM_API_KEY"
LOCAL_PLACEHOLDER_KEY = "local"

_OPENAI_MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")

Messages = list[dict[str, str]]


class ChatClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: Messages, max_tokens: int, temperature: float) -> str: ...


def _api_key(names: tuple[str, ...], provider: str) -> str:
    key = next((os.environ[name] for name in names if os.environ.get(name)), None)
    if key is None:
        raise RuntimeError(f"Set {' or '.join(names)} to call the {provider} API.")
    return key


class AnthropicChatClient:
    provider = PROVIDER_ANTHROPIC

    def __init__(self, model: str):
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(
            api_key=_api_key(ANTHROPIC_KEY_ENVS, self.provider),
            base_url=os.environ.get(ANTHROPIC_BASE_URL_ENV),
        )

    def complete(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.content[0].text


class OpenAIChatClient:
    """Chat Completions API. Intended for non-reasoning chat models (e.g. GPT-4o, GPT-4.1)."""

    provider = PROVIDER_OPENAI

    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        api_key, base_url = self._credentials()
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _credentials(self) -> tuple[str, str | None]:
        return _api_key(OPENAI_KEY_ENVS, self.provider), os.environ.get(OPENAI_BASE_URL_ENV)

    def complete(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.choices[0].message.content or ""


class LocalChatClient(OpenAIChatClient):
    """A self-hosted OpenAI-compatible server (e.g. ``vllm serve``). ``model`` is its served model name."""

    provider = PROVIDER_LOCAL

    def _credentials(self) -> tuple[str, str | None]:
        base_url = os.environ.get(LOCAL_BASE_URL_ENV)
        if not base_url:
            raise RuntimeError(
                f"Set {LOCAL_BASE_URL_ENV} (e.g. http://localhost:8000/v1) to use the {self.provider} provider."
            )
        return os.environ.get(LOCAL_KEY_ENV) or LOCAL_PLACEHOLDER_KEY, base_url


def infer_provider(model: str) -> str:
    return PROVIDER_OPENAI if model.lower().startswith(_OPENAI_MODEL_PREFIXES) else PROVIDER_ANTHROPIC


def make_client(provider: str, model: str) -> ChatClient:
    if provider == PROVIDER_ANTHROPIC:
        return AnthropicChatClient(model)
    if provider == PROVIDER_OPENAI:
        return OpenAIChatClient(model)
    if provider == PROVIDER_LOCAL:
        return LocalChatClient(model)
    raise ValueError(f"Unknown LLM provider {provider!r}; choose from {PROVIDERS}.")
