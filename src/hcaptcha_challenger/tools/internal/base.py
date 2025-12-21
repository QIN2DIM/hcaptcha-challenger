# -*- coding: utf-8 -*-
"""
Reasoner - Abstract base class for all reasoning tools.

This module provides the base class that all tool classes inherit from.
Using ABC allows us to share common implementation code while enforcing
that subclasses implement the required methods.

Design principles:
1. Provider-agnostic: Uses ChatProvider protocol for flexibility
2. Description-driven: Loads prompts from .md files
3. Standalone-friendly: Can be used without agent context
"""
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar, Union

from loguru import logger
from pydantic import BaseModel

from .providers.gemini import GeminiProvider
from .providers.protocol import ChatProvider

ModelT = TypeVar("ModelT", bound=str)
ResponseT = TypeVar("ResponseT", bound=Union[BaseModel, Enum])


class Reasoner(ABC, Generic[ModelT, ResponseT]):
    """
    Abstract base class for all reasoning tools.

    Subclasses must:
    1. Define a `description` class attribute with the system prompt
    2. Implement __call__() with their specific async logic

    Attributes:
        description: The system prompt for the tool.
            Subclasses should define this using `load_desc(Path(__file__).parent / 'xxx.md')`.
    """

    description: str = ""
    """The description of the tool."""

    def __init__(
        self,
        gemini_api_key: str,
        model: ModelT | None = None,
        *,
        provider: ChatProvider | None = None,
        **kwargs,
    ):
        """
        Initialize the reasoner.

        Args:
            gemini_api_key: Gemini API key (used if no custom provider is set).
            model: Model name to use.
            provider: Optional custom provider (for extensibility).
            **kwargs: Additional options for subclasses.
        """
        self._api_key = gemini_api_key
        self._model = model
        self._provider: ChatProvider = provider or self._create_default_provider()
        self._response = None

    def _create_default_provider(self) -> GeminiProvider:
        """Create the default Gemini provider."""
        return GeminiProvider(api_key=self._api_key, model=self._model)

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> ResponseT:
        """
        Invoke the reasoning tool asynchronously.

        Subclasses must implement this method with their specific logic.

        Usage:
            result = await tool(challenge_screenshot=path)

        Returns:
            The parsed response from the provider.
        """
        raise NotImplementedError

    def cache_response(self, path: Path) -> None:
        """
        Cache the last response to a file.

        Args:
            path: Path to save the response JSON.
        """
        cache_fn = getattr(self._provider, "cache_response", None)
        if cache_fn is not None and callable(cache_fn):
            cache_fn(path)
        elif self._response:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        self._response.model_dump(mode="json"), indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
