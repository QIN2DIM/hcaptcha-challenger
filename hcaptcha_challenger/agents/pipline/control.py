# -*- coding: utf-8 -*-
# Time       : 2023/11/17 18:35
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from pathlib import Path
from typing import List, Literal

from loguru import logger
from pydantic import BaseModel, Field

from hcaptcha_challenger.components.common import match_solution
from hcaptcha_challenger.components.image_classifier import rank_models
from hcaptcha_challenger.components.middleware import Status, QuestionResp, RequestType
from hcaptcha_challenger.components.prompt_handler import handle
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8, is_matched_ash_of_war


class RanniTheWitch(BaseModel):
    modelhub: ModelHub
    tmp_dir: Path
    self_supervised: bool

    prompt: str = ""
    label: str = ""
    ash: str = ""

    img_paths: List[Path] = Field(default_factory=list)

    @classmethod
    def from_question_data(
        cls,
        modelhub: ModelHub | None = None,
        tmp_dir: Path | None = None,
        self_supervised: bool = True,
        **kwargs,
    ):
        modelhub = modelhub or ModelHub.from_github_repo()
        if not modelhub.label_alias:
            modelhub.parse_objects()

        tmp_dir = tmp_dir or Path(__file__).parent.parent.joinpath("tmp_dir")

        self_supervised = kwargs.get("clip", self_supervised)

        return cls(modelhub=modelhub, tmp_dir=tmp_dir, self_supervised=self_supervised)

    @property
    def status(self):
        return Status

    def _match_solution(self, select: Literal["yolo", "resnet"] = None) -> ResNetControl | YOLOv8:
        """match solution after `tactical_retreat`"""
        model = match_solution(self.label, self.ash, self.modelhub, select=select)
        logger.debug("match model", select=select, prompt=self.prompt)

        return model

    def _rank_models(self, nested_models: List[str]) -> ResNetControl | None:
        result = rank_models(nested_models, self._example_paths, self.modelhub)
        if result and isinstance(result, tuple):
            best_model, model_name = result
            logger.debug("rank model", resnet=model_name, prompt=self._prompt)
            return best_model

    def binary_challenge(self, model=None):
        classifier = model or self._match_solution(select="resnet")

        for img_path in self.img_paths:
            result = classifier.execute(img_stream=img_path.read_bytes())

    def binary_challenge_clip(self):
        pass

    def keypoint_default_challenge(self):
        pass

    def bounding_challenge(self):
        pass

    def multiple_choice_challenge(self):
        pass


class AgentR(RanniTheWitch):
    async def __call__(self, *args, **kwargs):
        return await self.execute(*args, **kwargs)

    def _parse_label(self, qr: QuestionResp):
        self.prompt = qr.requester_question.get("en")
        self.label = handle(self.prompt)

        answer_keys = list(qr.requester_restricted_answer_set.keys())
        ak = answer_keys[0] if len(answer_keys) > 0 else ""
        self.ash = f"{self.label} {ak}"

        return self.prompt, self.label, self.ash

    async def execute(self, qr: QuestionResp, **kwargs) -> Status | None:
        if not qr.requester_question.keys():
            logger.success("task done", reasone="The challenge has been completed")
            return self.status.CHALLENGE_SUCCESS

        prompt, label, ash = self._parse_label(qr)

        # Match: image_label_binary
        if qr.request_type == RequestType.ImageLabelBinary:
            if nested_models := self.modelhub.nested_categories.get(label, []):
                if model := self._rank_models(nested_models):
                    self.binary_challenge(model)
                else:
                    return self.status.CHALLENGE_BACKCALL
            elif self.modelhub.label_alias.get(label):
                self.binary_challenge()
            elif self.self_supervised:
                self.binary_challenge_clip()
            else:
                return self.status.CHALLENGE_BACKCALL
        # Match: image_label_area_select
        elif qr.request_type == RequestType.ImageLabelAreaSelect:
            shape_type = qr.request_config.get("shape_type", "")

            if "default" in ash:
                if shape_type == "point":
                    self.keypoint_default_challenge()
                else:
                    return self.status.CHALLENGE_BACKCALL
            else:
                if not any(is_matched_ash_of_war(ash, c) for c in self.modelhub.yolo_names):
                    return self.status.CHALLENGE_BACKCALL
                if shape_type == "point":
                    self._keypoint_challenge()
                elif shape_type == "bounding_box":
                    self.bounding_challenge()
        # Match: image_label_multiple_choice
        elif qr.request_type == RequestType.ImageLabelMultipleChoice:
            if not self.self_supervised:
                logger.warning(
                    "task interrupt",
                    reasone="ImageLabelMultipleChoice mode must enable the self-supervised option",
                )
                return self.status.CHALLENGE_BACKCALL
            self.multiple_choice_challenge()
        # Match: Unknown case
        else:
            logger.warning("task interrupt", reason="Unknown type of challenge")
            return self.status.CHALLENGE_BACKCALL

        self.modelhub.unplug()
