# -*- coding: utf-8 -*-
from hcaptcha_challenger.agent.challenger import AgentConfig, build_provider
from hcaptcha_challenger.tools.internal.providers import (
    GeminiProvider,
    OpenAICompatibleProvider,
)


def test_build_gemini_provider():
    cfg = AgentConfig(GEMINI_API_KEY="k", CHAT_PROVIDER="gemini")
    p = build_provider(cfg, model="gemini-2.5-pro")
    assert isinstance(p, GeminiProvider)


def test_build_openai_provider():
    cfg = AgentConfig(
        GEMINI_API_KEY="",
        CHAT_PROVIDER="openai-compatible",
        OPENAI_BASE_URL="http://localhost:11434/v1",
    )
    p = build_provider(cfg, model="qwen2-vl")
    assert isinstance(p, OpenAICompatibleProvider)
