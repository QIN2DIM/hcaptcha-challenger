# -*- coding: utf-8 -*-
"""MiniMax chat provider with OpenAI- and Anthropic-compatible transports."""

import base64
import json
import re
from pathlib import Path
from typing import Any, List, Literal, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)
MiniMaxProtocol = Literal["openai", "anthropic"]
MiniMaxRegion = Literal["global_en", "cn_zh"]

MINIMAX_ENDPOINTS: dict[MiniMaxRegion, dict[MiniMaxProtocol, str]] = {
    "global_en": {
        "openai": "https://api.minimax.io/v1",
        "anthropic": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai": "https://api.minimaxi.com/v1",
        "anthropic": "https://api.minimaxi.com/anthropic",
    },
}

MINIMAX_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "context_window_semantics": "input_and_output",
        "input_modalities": ("text", "image", "video"),
        "thinking": ("adaptive", "disabled"),
        "thinking_defaults": {"openai": "adaptive", "anthropic": "disabled"},
        "pricing_usd_per_million_tokens": (
            {
                "service_tier": "standard",
                "input_tokens_lte": 512_000,
                "input": 0.3,
                "output": 1.2,
                "cache_read": 0.06,
                "cache_write": None,
            },
            {
                "service_tier": "standard",
                "input_tokens_gt": 512_000,
                "input": 0.6,
                "output": 2.4,
                "cache_read": 0.12,
                "cache_write": None,
            },
            {
                "service_tier": "priority",
                "input_tokens_lte": 512_000,
                "input": 0.45,
                "output": 1.8,
                "cache_read": 0.09,
                "cache_write": None,
            },
            {
                "service_tier": "priority",
                "input_tokens_gt": 512_000,
                "input": 0.9,
                "output": 3.6,
                "cache_read": 0.18,
                "cache_write": None,
            },
        ),
    },
    "MiniMax-M2.7": {
        "context_window": 204_800,
        "context_window_semantics": "input_and_output",
        "input_modalities": ("text",),
        "thinking": ("always_on",),
        "pricing_usd_per_million_tokens": (
            {
                "service_tier": "standard",
                "input": 0.3,
                "output": 1.2,
                "cache_read": 0.06,
                "cache_write": 0.375,
            },
        ),
    },
}

_IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class MiniMaxProvider:
    """Call supported MiniMax models through either compatible API protocol."""

    default_model = "MiniMax-M3"
    default_region: MiniMaxRegion = "global_en"
    default_protocol: MiniMaxProtocol = "openai"
    default_base_url = MINIMAX_ENDPOINTS[default_region][default_protocol]
    default_anthropic_base_url = MINIMAX_ENDPOINTS[default_region]["anthropic"]
    endpoints = MINIMAX_ENDPOINTS
    model_specs = MINIMAX_MODEL_SPECS

    def __init__(
        self,
        api_key: str,
        model: str = default_model,
        *,
        region: MiniMaxRegion = default_region,
        protocol: MiniMaxProtocol = default_protocol,
        base_url: str | None = None,
    ):
        """Initialize a provider for a model, region, and API protocol.

        The Anthropic ``base_url`` is an API root ending in ``/anthropic``;
        this adapter appends ``/v1/messages`` for direct HTTP requests.
        """
        if model not in self.model_specs:
            raise ValueError(f"Unsupported MiniMax model: {model}")
        if region not in self.endpoints:
            raise ValueError(f"Unsupported MiniMax region: {region}")
        if protocol not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported MiniMax protocol: {protocol}")

        resolved_base_url = (base_url or self.endpoints[region][protocol]).rstrip("/")
        if protocol == "anthropic" and not resolved_base_url.endswith("/anthropic"):
            raise ValueError("Anthropic base_url must end with /anthropic")

        self._api_key = api_key
        self._model = model
        self._region = region
        self._protocol = protocol
        self._base_url = resolved_base_url
        self._response: dict | None = None

    @property
    def last_response(self) -> dict | None:
        """Get the last raw response for debugging or caching."""
        return self._response

    @property
    def request_url(self) -> str:
        """Get the fully derived request URL for the configured protocol."""
        path = "chat/completions" if self._protocol == "openai" else "v1/messages"
        return f"{self._base_url}/{path}"

    @staticmethod
    def _encode_image(path: Path) -> tuple[str, str]:
        """Encode a supported image and return its media type and base64 data."""
        try:
            media_type = _IMAGE_MEDIA_TYPES[path.suffix.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported image format: {path.suffix or '<none>'}") from exc
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return media_type, data

    def _build_user_content(self, images: List[Path], user_prompt: str | None) -> list[dict]:
        """Build protocol-specific user content blocks."""
        if images and "image" not in self.model_specs[self._model]["input_modalities"]:
            raise ValueError(f"{self._model} does not support image input")

        content: list[dict] = []
        if user_prompt and isinstance(user_prompt, str):
            content.append({"type": "text", "text": user_prompt})

        for image in images:
            image_path = Path(image)
            if not image_path.exists():
                continue
            media_type, data = self._encode_image(image_path)
            if self._protocol == "openai":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
            else:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )

        if not content:
            content.append({"type": "text", "text": "Generate the requested JSON response."})
        return content

    @staticmethod
    def _system_prompt(description: str | None, response_schema: Type[BaseModel]) -> str:
        schema = json.dumps(
            response_schema.model_json_schema(), ensure_ascii=True, separators=(",", ":")
        )
        instruction = f"Return only a JSON object matching this JSON Schema:\n{schema}"
        return f"{description.strip()}\n\n{instruction}" if description else instruction

    def _build_payload(
        self,
        *,
        images: List[Path],
        response_schema: Type[BaseModel],
        user_prompt: str | None,
        description: str | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        user_content = self._build_user_content(images, user_prompt)
        system_prompt = self._system_prompt(description, response_schema)

        if self._protocol == "openai":
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
        else:
            payload = {
                "model": self._model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
                "max_tokens": 4096,
            }

        payload.update(options)
        return payload

    @staticmethod
    def _extract_json_object(content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError(f"Expected response text, got {type(content).__name__}")

        text = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        decoder = json.JSONDecoder()
        for candidate in [*fenced, text]:
            candidate = candidate.strip()
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed

            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

        raise ValueError("Failed to parse a JSON object from the MiniMax response")

    def _response_content(self) -> Any:
        if not self._response:
            raise ValueError("MiniMax returned an empty response")
        if self._protocol == "openai":
            return self._response["choices"][0]["message"]["content"]

        blocks = self._response.get("content", [])
        if isinstance(blocks, list):
            text_blocks = [
                block["text"]
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text" and "text" in block
            ]
            return "\n".join(text_blocks)
        return blocks

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry MiniMax request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """Generate and validate a structured response from text and image input."""
        payload = self._build_payload(
            images=images,
            response_schema=response_schema,
            user_prompt=user_prompt,
            description=description,
            options=kwargs,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.request_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
            self._response = response.json()

        parsed = self._extract_json_object(self._response_content())
        return response_schema.model_validate(parsed)

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to cache MiniMax response: {e}")
