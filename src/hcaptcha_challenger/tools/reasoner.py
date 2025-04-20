import json
from abc import abstractmethod, ABC
from pathlib import Path

from loguru import logger


class _Reasoner(ABC):

    def __init__(self, gemini_api_key: str):
        self._api_key: str = gemini_api_key
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
    def invoke(self, **kwargs):
        raise NotImplementedError
