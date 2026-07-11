# -*- coding: utf-8 -*-
import pytest
from pydantic import ValidationError

from hcaptcha_challenger.agent.challenger import AgentConfig


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
