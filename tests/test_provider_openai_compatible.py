# -*- coding: utf-8 -*-
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hcaptcha_challenger.tools.internal.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _encode_image_data_url,
)


class DummySchema(BaseModel):
    value: int


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
