import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

# Load the standalone HTTP adapter without importing browser/vision agents.
_provider_path = (
    Path(__file__).resolve().parents[1]
    / "src/hcaptcha_challenger/tools/internal/providers/minimax.py"
)
_spec = importlib.util.spec_from_file_location("minimax", _provider_path)
assert _spec is not None and _spec.loader is not None
minimax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(minimax)
MINIMAX_ENDPOINTS = minimax.MINIMAX_ENDPOINTS
MINIMAX_MODEL_SPECS = minimax.MINIMAX_MODEL_SPECS
MiniMaxProvider = minimax.MiniMaxProvider


class Answer(BaseModel):
    answer: str


class StubAsyncClient:
    def __init__(self, response_payload: dict[str, Any]):
        self.response_payload = response_payload
        self.request: dict[str, Any] | None = None

    def handle(self, request: httpx.Request):
        self.request = {
            "url": str(request.url),
            "headers": dict(request.headers),
            "json": json.loads(request.content),
        }
        return httpx.Response(200, json=self.response_payload)


def install_stub_client(monkeypatch, response_payload: dict[str, Any]) -> StubAsyncClient:
    client = StubAsyncClient(response_payload)
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        minimax.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(client.handle), **kwargs),
    )
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
    assert payload["model"] == "MiniMax-M3"
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
    assert payload["model"] == "MiniMax-M3"
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
    assert tuple(MINIMAX_MODEL_SPECS) == ("MiniMax-M3.1-Flash", "MiniMax-M3")
    flash = MINIMAX_MODEL_SPECS["MiniMax-M3.1-Flash"]
    assert flash["context_window"] == 1_000_000
    assert flash["max_output_tokens"] == 524_288
    assert flash["thinking"] == ("adaptive",)
    assert flash["pricing_usd_per_million_tokens"] == (
        {"input": 0.15, "output": 0.6, "cache_read": 0.015, "cache_write": None},
    )
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["context_window"] == 1_000_000
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["input_modalities"] == (
        "text",
        "image",
        "video",
    )
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["thinking"] == ("adaptive", "disabled")
    assert MINIMAX_MODEL_SPECS["MiniMax-M3"]["pricing_usd_per_million_tokens"] == (
        {
            "context_tier": "<=512k",
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": None,
        },
        {
            "context_tier": ">512k",
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
    )
    assert minimax._LEGACY_MODEL_SPECS["MiniMax-M2.7"] == {
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


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
async def test_serialized_text_request_preserves_model(monkeypatch, protocol, region, model):
    requests = []

    def handler(request):
        requests.append(request)
        body = (
            {"choices": [{"message": {"content": '{"answer":"ok"}'}}]}
            if protocol == "openai"
            else {"content": [{"type": "text", "text": '{"answer":"ok"}'}]}
        )
        return httpx.Response(200, json=body)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        minimax.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )
    provider = MiniMaxProvider("test-key", model=model, protocol=protocol, region=region)
    result = await provider.generate_with_images(
        images=[], response_schema=Answer, user_prompt="Return an answer."
    )
    assert result == Answer(answer="ok")
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == provider.request_url
    assert payload["model"] == (model or "MiniMax-M3.1-Flash")
    for field in (
        "thinking",
        "reasoning_effort",
        "output_config",
        "service_tier",
        "reasoning_split",
    ):
        assert field not in payload


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_serialized_flash_effort_and_custom_endpoint(monkeypatch, protocol, effort):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "content": [{"type": "text", "text": '{"answer":"ok"}'}],
            },
        )

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        minimax.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )
    root = (
        "https://gateway.example/anthropic"
        if protocol == "anthropic"
        else "https://gateway.example/v1"
    )
    provider = MiniMaxProvider("test-key", protocol=protocol, base_url=root)
    options = (
        {"reasoning_effort": effort}
        if protocol == "openai"
        else {"output_config": {"effort": effort}}
    )
    await provider.generate_with_images(images=[], response_schema=Answer, **options)
    payload = json.loads(requests[0].content)
    assert str(requests[0].url) == provider.request_url
    assert payload["model"] == "MiniMax-M3.1-Flash"
    for key, value in options.items():
        assert payload[key] == value


@pytest.mark.parametrize(
    ("protocol", "options", "error"),
    [
        ("openai", {"thinking": {"type": "disabled"}}, "adaptive thinking"),
        ("anthropic", {"thinking": {"type": "disabled"}}, "adaptive thinking"),
        ("openai", {"reasoning_effort": "none"}, "reasoning effort"),
        ("anthropic", {"output_config": {"effort": "none"}}, "reasoning effort"),
        ("openai", {"service_tier": "priority"}, "service_tier"),
        ("anthropic", {"service_tier": "standard"}, "service_tier"),
        ("openai", {"reasoning_split": False}, "separate reasoning"),
    ],
)
async def test_invalid_flash_options_fail_before_transport(monkeypatch, protocol, options, error):
    def forbidden_client(**kwargs):
        pytest.fail("Invalid Flash options must not reach the transport")

    monkeypatch.setattr(minimax.httpx, "AsyncClient", forbidden_client)
    with pytest.raises(ValueError, match=error):
        await MiniMaxProvider("test-key", protocol=protocol).generate_with_images(
            images=[], response_schema=Answer, **options
        )


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("thinking", ["adaptive", "disabled"])
async def test_saved_m3_thinking_is_preserved(monkeypatch, protocol, thinking):
    client = install_stub_client(
        monkeypatch,
        {
            "choices": [{"message": {"content": '{"answer":"ok"}'}}],
            "content": [{"type": "text", "text": '{"answer":"ok"}'}],
        },
    )
    provider = MiniMaxProvider("test-key", model="MiniMax-M3", protocol=protocol)
    await provider.generate_with_images(
        images=[], response_schema=Answer, thinking={"type": thinking}
    )
    assert client.request["json"]["thinking"] == {"type": thinking}
    assert client.request["json"]["model"] == "MiniMax-M3"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_explicit_messages_and_raw_reasoning_are_preserved(monkeypatch, protocol):
    response = (
        {"choices": [{"message": {"content": '{"answer":"ok"}', "reasoning_content": "reasoning"}}]}
        if protocol == "openai"
        else {
            "content": [
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": '{"answer":"ok"}'},
            ]
        }
    )
    client = install_stub_client(monkeypatch, response)
    messages = [{"role": "user", "content": "Return an answer."}]
    provider = MiniMaxProvider("test-key", protocol=protocol)
    await provider.generate_with_images(images=[], response_schema=Answer, messages=messages)
    assert client.request["json"]["messages"] == messages
    assert provider.last_response == response


def test_anthropic_override_requires_an_api_root():
    with pytest.raises(ValueError, match="must end with /anthropic"):
        MiniMaxProvider(
            "test-key",
            protocol="anthropic",
            base_url="https://gateway.example/v1",
        )
