# -*- coding: utf-8 -*-
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hcaptcha_challenger.tools.internal.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _CapabilityError,
    _encode_image_data_url,
    _is_capability_error,
    _is_transient,
)


class DummySchema(BaseModel):
    value: int


class FakeStatusError(Exception):
    """Stand-in for an openai APIStatusError carrying an HTTP status_code."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(Exception):
    """Name-matched stand-in for the openai transient exception."""


@pytest.fixture
def png(tmp_path) -> Path:
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return p


def test_encode_image_data_url(png):
    url = _encode_image_data_url(png)
    assert url.startswith("data:image/png;base64,")


async def test_strict_parse_path(png, monkeypatch):
    provider = OpenAICompatibleProvider(model="qwen2-vl", api_key="k")
    parsed = DummySchema(value=7)
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    mock_parse = AsyncMock(return_value=completion)
    monkeypatch.setattr(
        type(provider),
        "_client",
        property(
            lambda self: MagicMock(
                chat=MagicMock(completions=MagicMock(parse=mock_parse))
            )
        ),
    )
    result = await provider.generate_with_images(
        images=[png], response_schema=DummySchema, user_prompt="go", description="sys"
    )
    assert result.value == 7
    assert provider._supports_json_schema is True


async def test_capability_error_falls_back_without_retry(png, monkeypatch):
    provider = OpenAICompatibleProvider(
        model="llava", base_url="http://localhost:11434/v1"
    )

    parse_calls = {"n": 0}

    async def failing_parse(**kwargs):
        parse_calls["n"] += 1
        raise ValueError("response_format json_schema is not supported")

    create_completion = MagicMock()
    create_completion.choices = [MagicMock(message=MagicMock(content='{"value": 5}'))]
    mock_create = AsyncMock(return_value=create_completion)

    client = MagicMock(
        chat=MagicMock(
            completions=MagicMock(parse=failing_parse, create=mock_create)
        )
    )
    monkeypatch.setattr(type(provider), "_client", property(lambda self: client))

    result = await provider.generate_with_images(
        images=[png], response_schema=DummySchema, user_prompt="go", description="sys"
    )
    assert result.value == 5
    assert provider._supports_json_schema is False
    # capability error must not be retried: exactly one parse attempt
    assert parse_calls["n"] == 1

    # second call skips the strict path entirely
    await provider.generate_with_images(
        images=[png], response_schema=DummySchema, user_prompt="go", description="sys"
    )
    assert parse_calls["n"] == 1


def test_missing_sdk_message(monkeypatch):
    provider = OpenAICompatibleProvider(model="m", api_key="k")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no module named openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"hcaptcha-challenger\[openai\]"):
        _ = provider._client


# --- error classification (fixes false-positive capability + retry gaps) ---


def test_capability_marker_400_is_capability_not_transient():
    exc = FakeStatusError("response_format json_schema not supported", 400)
    assert _is_capability_error(exc) is True
    assert _is_transient(exc) is False


def test_vllm_unrecognized_argument_is_capability():
    exc = FakeStatusError("unrecognized request argument: response_format", 400)
    assert _is_capability_error(exc) is True


def test_auth_error_is_not_capability():
    # A 401 body typically contains "invalid_api_key"; must NOT be a capability miss.
    exc = FakeStatusError("Error code: 401 - invalid_api_key", 401)
    assert _is_capability_error(exc) is False
    assert _is_transient(exc) is False


def test_model_not_found_is_not_capability():
    exc = FakeStatusError("The model does not exist", 404)
    assert _is_capability_error(exc) is False


def test_invalid_image_400_is_not_capability():
    # No structured-output marker -> ordinary 400, surfaces immediately.
    exc = FakeStatusError("invalid image data", 400)
    assert _is_capability_error(exc) is False
    assert _is_transient(exc) is False


def test_old_sdk_attribute_error_is_capability():
    exc = AttributeError("'AsyncCompletions' object has no attribute 'parse'")
    assert _is_capability_error(exc) is True
    assert _is_transient(exc) is False


def test_internal_capability_error_is_capability():
    assert _is_capability_error(_CapabilityError("empty parsed payload")) is True
    assert _is_transient(_CapabilityError("empty parsed payload")) is False


def test_transient_statuses_and_names_are_retried():
    assert _is_transient(APITimeoutError("timeout")) is True
    assert _is_transient(FakeStatusError("rate limited", 429)) is True
    assert _is_transient(FakeStatusError("server boom", 500)) is True


def test_parse_failure_is_retried():
    # Fallback json parse failures must be retried like GeminiProvider.
    assert _is_transient(ValueError("Failed to parse JSON response: 'oops'")) is True


async def test_transient_error_is_retried_then_succeeds(png, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    provider = OpenAICompatibleProvider(model="qwen2-vl", api_key="k")

    calls = {"n": 0}
    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(parsed=DummySchema(value=42)))]

    async def flaky_parse(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise APITimeoutError("temporary")
        return ok

    client = MagicMock(chat=MagicMock(completions=MagicMock(parse=flaky_parse)))
    monkeypatch.setattr(type(provider), "_client", property(lambda self: client))

    result = await provider.generate_with_images(
        images=[png], response_schema=DummySchema, user_prompt="go", description="sys"
    )
    assert result.value == 42
    assert calls["n"] == 3  # two transient failures, third attempt succeeds


async def test_fallback_parse_failure_is_retried_then_succeeds(png, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    provider = OpenAICompatibleProvider(model="llava", base_url="http://localhost:11434/v1")
    provider._supports_json_schema = False  # force fallback path

    calls = {"n": 0}

    async def flaky_create(**kwargs):
        calls["n"] += 1
        content = "not json" if calls["n"] < 3 else '{"value": 7}'
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content=content))]
        return completion

    client = MagicMock(chat=MagicMock(completions=MagicMock(create=flaky_create)))
    monkeypatch.setattr(type(provider), "_client", property(lambda self: client))

    result = await provider.generate_with_images(
        images=[png], response_schema=DummySchema, user_prompt="go", description="sys"
    )
    assert result.value == 7
    assert calls["n"] == 3
