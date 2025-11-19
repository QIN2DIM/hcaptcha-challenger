import asyncio
import json
import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Generic, Union, List, Type

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import THINKING_BUDGET_MODELS, THINKING_LEVEL_MODELS
from hcaptcha_challenger.tools.common import run_sync, extract_first_json_block

M = TypeVar("M")
R = TypeVar("R")


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

    @staticmethod
    async def _upload_files(
        client: genai.Client, files: List[Union[str, Path, os.PathLike]]
    ) -> List[types.File]:
        """Upload multiple files concurrently."""
        valid_files = [f for f in files if f]
        if not valid_files:
            return []
        upload_tasks = [client.aio.files.upload(file=f) for f in valid_files]
        return await asyncio.gather(*upload_tasks)

    @staticmethod
    def _files_to_parts(files: List[types.File]) -> List[types.Part]:
        """Convert uploaded files to parts."""
        return [types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type) for f in files]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/2) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def _generate_content(
        self,
        client: genai.Client,
        model: str,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
        response_schema: Type[R],
    ) -> R:
        """Generate content with retry logic and response parsing."""
        self._response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )
        if self._response.parsed:
            return response_schema(**self._response.parsed.model_dump())
        return response_schema(**extract_first_json_block(self._response.text))

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
