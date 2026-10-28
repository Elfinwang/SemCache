import sys
import types

import pytest

from per_tuple_threshold.config import DEFAULT_LLM_MODELS, LLM_MODEL_ENV, LLM_PROVIDER_ENV, resolve_llm
from per_tuple_threshold.llm.client import (
    LOCAL_BASE_URL_ENV,
    LOCAL_KEY_ENV,
    LOCAL_PLACEHOLDER_KEY,
    PROVIDER_ANTHROPIC,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    infer_provider,
    make_client,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (LLM_MODEL_ENV, LLM_PROVIDER_ENV, LOCAL_BASE_URL_ENV, LOCAL_KEY_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-4o-mini", PROVIDER_OPENAI),
        ("GPT-4.1", PROVIDER_OPENAI),
        ("o3-mini", PROVIDER_OPENAI),
        ("claude-sonnet-4-5-20250929", PROVIDER_ANTHROPIC),
    ],
)
def test_infer_provider(model, provider):
    assert infer_provider(model) == provider


def test_resolve_llm_defaults_to_anthropic():
    assert resolve_llm(None, None) == (PROVIDER_ANTHROPIC, DEFAULT_LLM_MODELS[PROVIDER_ANTHROPIC])


def test_resolve_llm_provider_only_uses_provider_default():
    assert resolve_llm(None, PROVIDER_OPENAI) == (PROVIDER_OPENAI, DEFAULT_LLM_MODELS[PROVIDER_OPENAI])


def test_resolve_llm_infers_from_model_and_env(monkeypatch):
    assert resolve_llm("gpt-4.1", None) == (PROVIDER_OPENAI, "gpt-4.1")
    monkeypatch.setenv(LLM_PROVIDER_ENV, PROVIDER_OPENAI)
    monkeypatch.setenv(LLM_MODEL_ENV, "my-gateway-model")
    assert resolve_llm(None, None) == (PROVIDER_OPENAI, "my-gateway-model")


def test_resolve_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        resolve_llm(None, "gemini")


def _install_fake_openai(monkeypatch, captured):
    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content="YES")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class OpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))


def test_openai_client_reads_env_and_calls_chat_completions(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    client = make_client(PROVIDER_OPENAI, "gpt-4o-mini")
    text = client.complete([{"role": "user", "content": "q"}], max_tokens=16, temperature=0)
    assert text == "YES"
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://example.invalid/v1"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["max_tokens"] == 16
    assert captured["temperature"] == 0


def test_missing_openai_key_raises(monkeypatch):
    _install_fake_openai(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_client(PROVIDER_OPENAI, "gpt-4o-mini")


def test_anthropic_client_reads_env(monkeypatch):
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="NO")])

    class Anthropic:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    client = make_client(PROVIDER_ANTHROPIC, "claude-sonnet-4-5-20250929")
    assert client.complete([{"role": "user", "content": "q"}], max_tokens=16, temperature=0) == "NO"
    assert captured["api_key"] == "ak-test"
    assert captured["model"] == "claude-sonnet-4-5-20250929"


def test_resolve_llm_selects_local_when_base_url_is_set(monkeypatch):
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, "http://localhost:8000/v1")
    assert resolve_llm("llama-3.1-8b", None) == (PROVIDER_LOCAL, "llama-3.1-8b")
    assert resolve_llm("gpt-4o-mini", PROVIDER_OPENAI) == (PROVIDER_OPENAI, "gpt-4o-mini")


def test_resolve_llm_local_requires_a_model(monkeypatch):
    with pytest.raises(ValueError, match="served model name"):
        resolve_llm(None, PROVIDER_LOCAL)
    monkeypatch.setenv(LLM_MODEL_ENV, "qwen2.5-14b")
    assert resolve_llm(None, PROVIDER_LOCAL) == (PROVIDER_LOCAL, "qwen2.5-14b")


def test_local_client_uses_base_url_and_placeholder_key(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, "http://localhost:8000/v1")
    client = make_client(PROVIDER_LOCAL, "llama-3.1-8b")
    assert client.complete([{"role": "user", "content": "q"}], max_tokens=16, temperature=0) == "YES"
    assert captured["api_key"] == LOCAL_PLACEHOLDER_KEY
    assert captured["base_url"] == "http://localhost:8000/v1"
    assert captured["model"] == "llama-3.1-8b"

    monkeypatch.setenv(LOCAL_KEY_ENV, "token-123")
    make_client(PROVIDER_LOCAL, "llama-3.1-8b")
    assert captured["api_key"] == "token-123"


def test_local_client_requires_base_url(monkeypatch):
    _install_fake_openai(monkeypatch, {})
    with pytest.raises(RuntimeError, match=LOCAL_BASE_URL_ENV):
        make_client(PROVIDER_LOCAL, "llama-3.1-8b")
