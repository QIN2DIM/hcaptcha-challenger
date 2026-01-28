# -*- coding: utf-8 -*-
"""
ChatProvider Protocol - Abstract interface for LLM providers.

This module defines the Protocol that all chat providers must implement.
Using Protocol instead of ABC allows for duck typing - external implementations
don't need to inherit from our classes.

Future implementations:
- GeminiProvider (current)
- OpenAIProvider (planned)
- AnthropicProvider (planned)
"""
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable, List

from pydantic import BaseModel

ResponseT = TypeVar("ResponseT", bound=BaseModel)


@runtime_checkable
class ChatProvider(Protocol[ResponseT]):
    """
    Protocol for chat providers that support image-based content generation.

    This is an extensibility point - users can implement their own providers
    """

    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: type[ResponseT],
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
            **kwargs: Provider-specific options (e.g., thinking_level, temperature).

        Returns:
            Parsed response matching the response_schema type.
        """
        ...
