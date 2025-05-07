import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union

from loguru import logger

from hcaptcha_challenger.models import SCoTModelType, FastShotModelType
from hcaptcha_challenger.tools.common import run_sync


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
    async def invoke_async(
        self,
        model: Union[SCoTModelType, FastShotModelType] = "gemini-2.5-pro-exp-03-25",
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    # for backward compatibility
    def invoke(
        self,
        model: Union[SCoTModelType, FastShotModelType] = "gemini-2.5-pro-exp-03-25",
        *args,
        **kwargs,
    ):
        return run_sync(self.invoke_async(model=model, *args, **kwargs))
