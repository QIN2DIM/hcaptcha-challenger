# -*- coding: utf-8 -*-
# Time       : 2023/11/17 18:35
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger
from pydantic import BaseModel

from hcaptcha_challenger import handle, ModelHub, ResNetControl
from hcaptcha_challenger.components.image_classifier import rank_models
from hcaptcha_challenger.components.middleware import QuestionResp, Status, RequestType


class RanniTheWitch(BaseModel):
    qr: QuestionResp
    modelhub: ModelHub
    self_supervised: bool = False

    prompt: str = ""
    label: str = ""

    @classmethod
    def from_question_data(
        cls,
        data: Dict[str, Any],
        modelhub: ModelHub | None = None,
        tmp_dir: Path | None = None,
        **kwargs,
    ):
        """
        Hijacking task from `/getcaptcha/` API

        Args:
            data:
                Examples
                ---

                ```python
                import requests

                response = requests.post("https://api.hcaptcha.com/getcaptcha/UUID", **params)
                data = response.json()
                agent = RanniTheWitch.from_question_data(data)
                ```
            modelhub:
            tmp_dir:
            **kwargs:

        Returns:

        """
        self_supervised = kwargs.get("self_supervised", False)

        modelhub = modelhub or ModelHub
        if not modelhub.label_alias:
            modelhub.parse_objects()

        return cls(qr=QuestionResp(**data), modelhub=modelhub, self_supervised=self_supervised)

    @property
    def status(self):
        return Status

    @property
    def ash(self):
        answer_keys = list(self.qr.requester_restricted_answer_set.keys())
        ak = answer_keys[0] if len(answer_keys) > 0 else ""
        ash = f"{self._label} {ak}"
        return ash

    def _parse_label(self):
        self.prompt = self.qr.requester_question.get("en")
        self.label = handle(self.prompt)

    def _rank_models(self, nested_models: List[str]) -> ResNetControl | None:
        result = rank_models(nested_models, self._example_paths, self.modelhub)
        if result and isinstance(result, tuple):
            best_model, model_name = result
            logger.debug("rank model", resnet=model_name, prompt=self._prompt)
            return best_model

    def _binary_challenge(self, model=None):
        pass

    def _binary_challenge_clip(self):
        pass

    def _keypoint_default_challenge(self):
        pass

    def _bounding_challenge(self):
        pass

    def _multiple_choice_challenge(self):
        pass


class AgentR(RanniTheWitch):
    async def __call__(self, *args, **kwargs):
        return await self.execute(**kwargs)

    async def execute(self, **kwargs) -> Status | None:
        if not self.qr.requester_question.keys():
            return self.status.CHALLENGE_SUCCESS

        self._parse_label()

        # Match: image_label_binary
        if self.qr.request_type == RequestType.ImageLabelBinary:
            if nested_models := self.modelhub.nested_categories.get(self.label, []):
                if model := self._rank_models(nested_models):
                    self._binary_challenge(model)
                else:
                    return self.status.CHALLENGE_BACKCALL
            elif self.modelhub.label_alias.get(self.label):
                self._binary_challenge()
            elif self.self_supervised:
                self._binary_challenge_clip()
            else:
                return self.status.CHALLENGE_BACKCALL
        # Match: image_label_area_select
        elif self.qr.request_type == RequestType.ImageLabelAreaSelect:
            pass
        # Match: image_label_multiple_choice
        elif self.qr.request_type == RequestType.ImageLabelMultipleChoice:
            pass
        # Match: Unknown case
        else:
            logger.warning("task interrupt", reason="Unknown type of challenge")
            return self.status.CHALLENGE_BACKCALL
