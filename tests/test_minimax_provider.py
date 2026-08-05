from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from hcaptcha_challenger.tools.internal.providers import minimax
from hcaptcha_challenger.tools.internal.providers.minimax import (
    MINIMAX_ENDPOINTS,
    MINIMAX_MODEL_SPECS,
    MiniMaxProvider,
)


class Answer(BaseModel):
    answer: str


class StubAsyncClient:
    def __init__(self, response_payload: dict[str, Any]):
        self.response_payload = response_payload
        self.request: dict[str, Any] | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def post(self, url: str, *, headers: dict, json: dict):
        self.request = {"url": url, "headers": headers, "json": json}
        return httpx.Response(
            200,
            json=self.response_payload,
            request=httpx.Request("POST", url),
        )


def install_stub_client(monkeypatch, response_payload: dict[str, Any]) -> StubAsyncClient:
    client = StubAsyncClient(response_payload)
    monkeypatch.setattr(minimax.httpx, "AsyncClient", lambda **kwargs: client)
    return client


@pytest.mark.parametrize(
    ("region", "protocol", "expected_url"),
    [
        ("global_en", "openai", "https://api.minimax.io/v1/chat/completions"),
        ("global_en", "anthropic", "https://api.minimax.io/anthropic/v1/messages"),
        ("cn_zh", "openai", "https://api.minimaxi.com/v1/chat/completions"),
        ("cn_zh", "anthropic", "https://api.minimaxi.com/anthropic/v1/messages"),
    ],
)
async def test_endpoint_matrix_uses_the_derived_request_url(
    monkeypatch, region: str, protocol: str, expected_url: str
):
    response_payload = (
        {"choices": [{"message": {"content": '{"answer":"ok"}'}}]}
        if protocol == "openai"
        else {"content": [{"type": "text", "text": '{"answer":"ok"}'}]}
    )
    client = install_stub_client(monkeypatch, response_payload)
    provider = MiniMaxProvider("test-key", region=region, protocol=protocol)

    result = await provider.generate_with_images(
        images=[], response_schema=Answer, user_prompt="Return an answer."
    )

    assert result == Answer(answer="ok")
    assert client.request["url"] == expected_url


async def test_openai_request_uses_supported_multimodal_fields(monkeypatch, tmp_path):
    image = tmp_path / "challenge.png"
    image.write_bytes(b"image-data")
    client = install_stub_client(
        monkeypatch,
        {"choices": [{"message": {"content": '{"answer":"openai"}'}}]},
    )
    provider = MiniMaxProvider("test-key", protocol="openai")

    result = await provider.generate_with_images(
        images=[image],
        response_schema=Answer,
        user_prompt="Inspect this image.",
        description="Solve the task.",
    )

    payload = client.request["json"]
    assert result == Answer(answer="openai")
    assert "response_format" not in payload
    assert "thinking" not in payload
    assert "JSON Schema" in payload["messages"][0]["content"]
    image_part = payload["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


async def test_anthropic_request_uses_native_content_blocks(monkeypatch, tmp_path):
    image = tmp_path / "challenge.webp"
    image.write_bytes(b"image-data")
    client = install_stub_client(
        monkeypatch,
        {
            "content": [
                {"type": "thinking", "thinking": "Internal reasoning"},
                {"type": "text", "text": '```json\n{"answer":"anthropic"}\n```'},
            ]
        },
    )
    provider = MiniMaxProvider("test-key", region="cn_zh", protocol="anthropic")

    result = await provider.generate_with_images(
        images=[image],
        response_schema=Answer,
        user_prompt="Inspect this image.",
        description="Solve the task.",
    )

    payload = client.request["json"]
    assert result == Answer(answer="anthropic")
    assert payload["max_tokens"] == 4096
    assert "JSON Schema" in payload["system"]
    image_part = payload["messages"][0]["content"][1]
    assert image_part == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/webp",
            "data": "aW1hZ2UtZGF0YQ==",
        },
    }


async def test_m27_accepts_text_and_rejects_image_input(monkeypatch, tmp_path):
    client = install_stub_client(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": '<think>Reasoning</think>\n```json\n{"answer":"text"}\n```'
                    }
                }
            ]
        },
    )
    provider = MiniMaxProvider("test-key", model="MiniMax-M2.7")

    result = await provider.generate_with_images(
        images=[], response_schema=Answer, user_prompt="Return an answer."
    )

    assert result == Answer(answer="text")
    assert client.request["json"]["model"] == "MiniMax-M2.7"
    assert "thinking" not in client.request["json"]

    image = tmp_path / "challenge.png"
    image.write_bytes(b"image-data")
    with pytest.raises(ValueError, match="does not support image input"):
        await provider.generate_with_images(
            images=[image], response_schema=Answer, user_prompt="Inspect this image."
        )


def test_model_registry_matches_target_facts():
    assert tuple(MINIMAX_MODEL_SPECS) == ("MiniMax-M3", "MiniMax-M2.7")
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["context_window"] == 1_000_000
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["input_modalities"] == (
        "text",
        "image",
        "video",
    )
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["thinking"] == ("adaptive", "disabled")
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["pricing_usd_per_million_tokens"] == (
        {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
    )
    assert MINIMAX_MODEL_SPECS["MiniMax-M2.7"] == {
        "context_window": 204_800,
        "input_modalities": ("text",),
        "thinking": ("always_on",),
        "pricing_usd_per_million_tokens": (
            {
                "input": 0.3,
                "output": 1.2,
                "cache_read": 0.06,
                "cache_write": 0.375,
            },
        ),
    }
    assert MINIMAX_ENDPOINTS["global_en"]["anthropic"].endswith("/anthropic")
    assert MINIMAX_ENDPOINTS["cn_zh"]["anthropic"].endswith("/anthropic")


def test_anthropic_override_requires_an_api_root():
    with pytest.raises(ValueError, match="must end with /anthropic"):
        MiniMaxProvider(
            "test-key",
            protocol="anthropic",
            base_url="https://gateway.example/v1",
        )
