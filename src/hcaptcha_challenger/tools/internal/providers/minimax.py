# -*- coding: utf-8 -*-
"""
MiniMaxProvider - MiniMax OpenAI-compatible API implementation.

This provider uses MiniMax-M3 through MiniMax's OpenAI-compatible chat
completions endpoint to provide image-based structured content generation.
"""
import base64
import json
from pathlib import Path
from typing import List, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class MiniMaxProvider:
    """
    MiniMax chat provider implementation.

    The provider targets MiniMax-M3 via MiniMax's OpenAI-compatible endpoint.
    """

    default_model = "MiniMax-M3"
    default_base_url = "https://api.minimax.io/v1"
    default_anthropic_base_url = "https://api.minimax.io/anthropic"
    context_window = 1_000_000
    input_price = 0.6
    output_price = 2.4
    cache_read_price = 0.12
    cache_write_price = None

    def __init__(
        self,
        api_key: str,
        model: str = default_model,
        *,
        base_url: str = default_base_url,
    ):
        """
        Initialize the MiniMax provider.

        Args:
            api_key: MiniMax API key.
            model: Model name to use. Defaults to MiniMax-M3.
            base_url: OpenAI-compatible API base URL.
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._response: dict | None = None

    @property
    def last_response(self) -> dict | None:
        """Get the last raw response for debugging/caching purposes."""
        return self._response

    @staticmethod
    def _encode_image(path: Path) -> str:
        """Encode an image as a data URL for chat completions."""
        suffix = path.suffix.lower().lstrip(".") or "png"
        media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/{media_type};base64,{data}"

    def _build_user_content(self, images: List[Path], user_prompt: str | None) -> list[dict]:
        """Build OpenAI-compatible multimodal user content."""
        content: list[dict] = []
        if user_prompt and isinstance(user_prompt, str):
            content.append({"type": "text", "text": user_prompt})

        for image in images:
            image_path = Path(image)
            if image_path.exists():
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(image_path)},
                    }
                )
        return content

    @retry(
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
        """
        Generate content with image inputs.

        Args:
            images: List of image file paths to include in the request.
            response_schema: Pydantic model class for structured output.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            **kwargs: Additional chat completion options.

        Returns:
            Parsed response matching the response_schema type.
        """
        messages: list[dict] = []
        if description:
            messages.append({"role": "system", "content": description})
        messages.append({"role": "user", "content": self._build_user_content(images, user_prompt)})

        payload = {
            "model": self._model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                },
            },
        }
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
            self._response = response.json()

        content = self._response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return response_schema(**json.loads(content))
        if isinstance(content, dict):
            return response_schema(**content)

        raise ValueError(f"Failed to parse MiniMax response: {content}")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._response, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to cache MiniMax response: {e}")
