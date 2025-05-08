import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Generic

from loguru import logger

from hcaptcha_challenger.tools.common import run_sync

M = TypeVar("M")


class _Reasoner(ABC, Generic[M]):

    def __init__(
        self, gemini_api_key: str, model: M | None = None, constraint_response_schema: bool = False
    ):
        self._api_key: str = gemini_api_key
        self._model: M | None = model
        self._constraint_response_schema = constraint_response_schema
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

    # for backward compatibility
    def invoke(self, *args, **kwargs):
        return run_sync(self.invoke_async(*args, **kwargs))
