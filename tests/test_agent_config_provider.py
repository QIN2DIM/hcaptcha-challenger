# -*- coding: utf-8 -*-
import pytest
from pydantic import ValidationError

from hcaptcha_challenger.agent.challenger import AgentConfig


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Provider credentials must come only from explicit kwargs in these tests.

    AgentConfig fields read os.environ via default_factory and also load .env,
    so a machine with GEMINI_API_KEY / OPENAI_API_KEY exported would otherwise
    flip the validation outcomes.
    """
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)


def test_default_is_gemini_requires_key():
    with pytest.raises(ValidationError):
        AgentConfig(GEMINI_API_KEY="", CHAT_PROVIDER="gemini")


def test_gemini_with_key_ok():
    cfg = AgentConfig(GEMINI_API_KEY="k", CHAT_PROVIDER="gemini")
    assert cfg.CHAT_PROVIDER == "gemini"


def test_openai_local_needs_only_base_url():
    cfg = AgentConfig(
        GEMINI_API_KEY="",
        CHAT_PROVIDER="openai-compatible",
        OPENAI_BASE_URL="http://localhost:11434/v1",
    )
    assert cfg.OPENAI_BASE_URL.endswith("/v1")


def test_openai_hosted_needs_only_api_key():
    cfg = AgentConfig(
        GEMINI_API_KEY="",
        CHAT_PROVIDER="openai-compatible",
        OPENAI_API_KEY="sk-x",
    )
    assert cfg.OPENAI_API_KEY.get_secret_value() == "sk-x"


def test_openai_without_any_credential_fails():
    with pytest.raises(ValidationError):
        AgentConfig(GEMINI_API_KEY="", CHAT_PROVIDER="openai-compatible")
