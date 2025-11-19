import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Generic

from google.genai import types
from loguru import logger

from hcaptcha_challenger.models import THINKING_BUDGET_MODELS, THINKING_LEVEL_MODELS
from hcaptcha_challenger.tools.common import run_sync

M = TypeVar("M")


class _Reasoner(ABC, Generic[M]):

    def __init__(self, gemini_api_key: str, model: M | None = None, **kwargs):
        self._api_key: str = gemini_api_key
        self._model: M | None = model
        self._response = None

    def cache_response(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(e)

    @abstractmethod
    async def invoke_async(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _set_thinking_config(
        config: types.GenerateContentConfig,
        model_to_use: str,
        thinking_level: types.ThinkingLevel | str | None = None,
    ):
        config.thinking_config = types.ThinkingConfig(include_thoughts=True)

        if model_to_use in THINKING_LEVEL_MODELS:
            if isinstance(thinking_level, str):
                if thinking_level.lower() in ["low"]:
                    thinking_level = types.ThinkingLevel.LOW
                elif thinking_level.lower() in ["high"]:
                    thinking_level = types.ThinkingLevel.HIGH
            elif isinstance(thinking_level, types.ThinkingLevel):
                thinking_level = thinking_level

            if not thinking_level or not isinstance(thinking_level, types.ThinkingLevel):
                thinking_level = types.ThinkingLevel.LOW

            config.thinking_config = types.ThinkingConfig(
                include_thoughts=False, thinking_level=thinking_level
            )

    @staticmethod
    @logger.catch
    def _set_temperature(config: types.GenerateContentConfig, model_to_use: str):
        if model_to_use in THINKING_BUDGET_MODELS:
            config.temperature = 0
        elif model_to_use in THINKING_LEVEL_MODELS:
            config.temperature = 1.0

    # for backward compatibility
    def invoke(self, *args, **kwargs):
        return run_sync(self.invoke_async(*args, **kwargs))
