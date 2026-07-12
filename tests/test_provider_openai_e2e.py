# -*- coding: utf-8 -*-
"""End-to-end tests for OpenAICompatibleProvider against a fake OpenAI server.

These exercise the REAL openai SDK client, real base64 image encoding, real
message building, and both the strict-parse and json_object fallback paths --
the SDK contract that unit-level mocks cannot cover. No network access: a
loopback http.server on an ephemeral port stands in for the endpoint.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import BaseModel

from hcaptcha_challenger.tools.internal.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


class Answer(BaseModel):
    value: int


class _FakeOpenAI:
    """A loopback OpenAI-compatible /v1/chat/completions server."""

    def __init__(self):
        self.requests: list[dict] = []
        self.reject_json_schema = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                rf_type = body.get("response_format", {}).get("type")

                if outer.reject_json_schema and rf_type == "json_schema":
                    self._json(
                        400,
                        {
                            "error": {
                                "message": "response_format json_schema is not supported",
                                "type": "invalid_request_error",
                            }
                        },
                    )
                    return

                self._json(
                    200,
                    {
                        "id": "chatcmpl-x",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body.get("model", "fake"),
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": '{"value": 3}',
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                )

            def _json(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()


@pytest.fixture
def server():
    with _FakeOpenAI() as s:
        yield s


@pytest.fixture
def png() -> Path:
    p = Path(__file__).parent / "challenge_view/image_label_binary/1.png"
    assert p.exists(), p
    return p


async def test_strict_path_end_to_end(server, png):
    provider = OpenAICompatibleProvider(model="fake", base_url=server.base_url, api_key="k")
    result = await provider.generate_with_images(
        images=[png], response_schema=Answer, user_prompt="go", description="sys"
    )
    assert result.value == 3
    assert provider._supports_json_schema is True

    # the image was really inlined as a base64 data URL through the SDK
    content = server.requests[0]["messages"][-1]["content"]
    image_part = next(c for c in content if c["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert server.requests[0]["response_format"]["type"] == "json_schema"


async def test_fallback_path_end_to_end(server, png):
    server.reject_json_schema = True
    provider = OpenAICompatibleProvider(model="fake", base_url=server.base_url, api_key="k")
    result = await provider.generate_with_images(
        images=[png], response_schema=Answer, user_prompt="go", description="sys"
    )
    assert result.value == 3
    assert provider._supports_json_schema is False
    # probe (json_schema, rejected) then fallback (json_object)
    assert [r["response_format"]["type"] for r in server.requests] == [
        "json_schema",
        "json_object",
    ]


async def test_memoization_skips_reprobe(server, png):
    server.reject_json_schema = True
    provider = OpenAICompatibleProvider(model="fake", base_url=server.base_url, api_key="k")
    await provider.generate_with_images(
        images=[png], response_schema=Answer, user_prompt="go", description="sys"
    )
    server.requests.clear()
    await provider.generate_with_images(
        images=[png], response_schema=Answer, user_prompt="go", description="sys"
    )
    # second call goes straight to json_object, no json_schema re-probe
    assert [r["response_format"]["type"] for r in server.requests] == ["json_object"]
