# -*- coding: utf-8 -*-
"""
OpenAICompatibleProvider - provider for any OpenAI-compatible chat endpoint.

Works with the official OpenAI API, OpenRouter, and self-hosted engines
(Ollama, vLLM, SGLang, LM Studio, llama.cpp, TGI, LocalAI). Requires a
vision-capable model (e.g. Qwen2-VL, LLaVA, Pixtral).

The `openai` SDK is an optional dependency and is imported lazily.
"""
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, List, Type, TypeVar

from loguru import logger
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from ._utils import parse_json_response

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class _CapabilityError(Exception):
    """Raised when the backend does not support strict json_schema output."""


def _encode_image_data_url(path: Path) -> str:
    """Read an image file and return a base64 data URL."""
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _is_capability_error(exc: BaseException) -> bool:
    """Heuristic: does this exception mean 'json_schema unsupported'?"""
    if isinstance(exc, _CapabilityError):
        return True
    msg = str(exc).lower()
    markers = (
        "response_format",
        "json_schema",
        "not supported",
        "unsupported",
        "invalid",
    )
    return any(m in msg for m in markers)


def _is_transient(exc: BaseException) -> bool:
    """Retry only transient errors; never retry capability errors."""
    if isinstance(exc, _CapabilityError):
        return False
    name = type(exc).__name__
    transient = (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    )
    if name in transient:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return False


class OpenAICompatibleProvider:
    """Chat provider backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._extra = kwargs
        self._client_instance = None
        self._response: Any = None
        # Capability flag, memoized after first probe.
        self._supports_json_schema: bool = True

    @property
    def _client(self):
        """Lazy-initialize the AsyncOpenAI client; import the SDK on demand."""
        if self._client_instance is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for OpenAICompatibleProvider. "
                    "Install it with: pip install hcaptcha-challenger[openai]"
                ) from e
            self._client_instance = AsyncOpenAI(
                api_key=self._api_key or "sk-no-auth",
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client_instance

    @property
    def last_response(self):
        return self._response

    def _build_messages(
        self, images: List[Path], user_prompt: str | None, description: str | None
    ) -> list[dict]:
        content: list[dict] = []
        for img in images:
            if img and Path(img).exists():
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _encode_image_data_url(Path(img)),
                            "detail": "high",
                        },
                    }
                )
        if user_prompt:
            content.append({"type": "text", "text": user_prompt})

        messages: list[dict] = []
        if description:
            messages.append({"role": "system", "content": description})
        messages.append({"role": "user", "content": content})
        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        retry=retry_if_exception(_is_transient),
        before_sleep=lambda rs: logger.warning(
            f"Retry request ({rs.attempt_number}/3) - Wait 3s - "
            f"Exception: {rs.outcome.exception()}"
        ),
        reraise=True,
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> ResponseT:
        messages = self._build_messages(images, user_prompt, description)

        # Strict json_schema path (memoized capability).
        if self._supports_json_schema:
            try:
                completion = await self._client.chat.completions.parse(
                    model=self._model,
                    messages=messages,
                    response_format=response_schema,
                    **kwargs,
                )
                self._response = completion
                parsed = completion.choices[0].message.parsed
                if isinstance(parsed, BaseModel):
                    return response_schema(**parsed.model_dump())
                if isinstance(parsed, dict):
                    return response_schema(**parsed)
                # No parsed payload -> treat as capability miss, fall through.
                raise _CapabilityError("empty parsed payload")
            except Exception as e:  # noqa: BLE001
                if _is_transient(e):
                    raise  # let tenacity retry
                if not _is_capability_error(e):
                    raise
                logger.warning(
                    f"json_schema unsupported by backend, falling back to "
                    f"json_object: {e}"
                )
                self._supports_json_schema = False

        # Fallback: json_object + schema-in-prompt + tolerant parse.
        schema_hint = {
            "role": "system",
            "content": (
                "Respond with a single JSON object matching this JSON schema "
                "(no prose, no code fences):\n"
                f"{json.dumps(response_schema.model_json_schema())}"
            ),
        }
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[schema_hint, *messages],
            response_format={"type": "json_object"},
            **kwargs,
        )
        self._response = completion
        text = completion.choices[0].message.content
        data = parse_json_response(text or "")
        return response_schema(**data)

    def cache_response(self, path: Path) -> None:
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            dump = self._response.model_dump(mode="json")
            path.write_text(
                json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to cache response: {e}")
