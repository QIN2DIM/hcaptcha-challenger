# -*- coding: utf-8 -*-
"""
AnthropicProvider - Anthropic Claude API implementation.

This provider wraps the anthropic SDK to provide image-based content generation.
"""
import base64
import json
from pathlib import Path
from typing import List, Type, TypeVar

import anthropic
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    import re

    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


def _get_media_type(file_path: Path) -> str:
    """Get the media type for a file based on its extension."""
    suffix = file_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/png")


class AnthropicProvider:
    """
    Anthropic Claude-based chat provider implementation.

    This class encapsulates all Anthropic-specific logic, making it easy to
    swap out for other providers.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model name to use (e.g., "claude-sonnet-4-20250514").
        """
        self._api_key = api_key
        self._model = model
        self._client: anthropic.AsyncAnthropic | None = None
        self._response: anthropic.types.Message | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    @property
    def last_response(self) -> anthropic.types.Message | None:
        """Get the last response for debugging/caching purposes."""
        return self._response

    def _encode_image(self, file_path: Path) -> tuple[str, str]:
        """
        Encode an image file to base64.

        Args:
            file_path: Path to the image file.

        Returns:
            Tuple of (base64_data, media_type)
        """
        with open(file_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        media_type = _get_media_type(file_path)
        return data, media_type

    def _build_content(
        self, images: List[Path], user_prompt: str | None = None
    ) -> list[dict]:
        """Build the content array for the message."""
        content = []

        # Add images
        for img_path in images:
            if img_path and Path(img_path).exists():
                data, media_type = self._encode_image(img_path)
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

        # Add user prompt if provided
        if user_prompt and isinstance(user_prompt, str):
            content.append({"type": "text", "text": user_prompt})

        return content

    def _build_json_schema_prompt(self, response_schema: Type[ResponseT]) -> str:
        """Build a prompt that instructs Claude to return JSON matching the schema."""
        schema = response_schema.model_json_schema()
        return (
            f"\n\nYou MUST respond with a valid JSON object that matches this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Do not include any other text, only the JSON object."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
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
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            response_schema: Pydantic model class for structured output.
            **kwargs: Additional options passed to the API.

        Returns:
            Parsed response matching the response_schema type.
        """
        # Build content with images and prompt
        content = self._build_content(images, user_prompt)

        # Add JSON schema instructions to content
        json_schema_prompt = self._build_json_schema_prompt(response_schema)
        content.append({"type": "text", "text": json_schema_prompt})

        # Build system prompt
        system_prompt = description or ""

        # Generate response
        self._response = await self.client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        # Parse response
        response_text = ""
        for block in self._response.content:
            if hasattr(block, "text"):
                response_text += block.text

        # Try to parse JSON directly
        try:
            json_data = json.loads(response_text)
            return response_schema(**json_data)
        except json.JSONDecodeError:
            pass

        # Fallback to JSON extraction from code blocks
        json_data = extract_first_json_block(response_text)
        if json_data:
            return response_schema(**json_data)

        raise ValueError(f"Failed to parse response: {response_text}")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Convert the response to a serializable format
            response_data = {
                "id": self._response.id,
                "type": self._response.type,
                "role": self._response.role,
                "model": self._response.model,
                "content": [
                    {"type": block.type, "text": getattr(block, "text", "")}
                    for block in self._response.content
                ],
                "stop_reason": self._response.stop_reason,
                "usage": {
                    "input_tokens": self._response.usage.input_tokens,
                    "output_tokens": self._response.usage.output_tokens,
                },
            }
            path.write_text(
                json.dumps(response_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
