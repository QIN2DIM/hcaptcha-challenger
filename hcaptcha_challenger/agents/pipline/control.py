# -*- coding: utf-8 -*-
# Time       : 2023/11/17 18:35
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Iterable

import cv2
from PIL import Image
from loguru import logger

from hcaptcha_challenger.components.common import (
    match_model,
    download_challenge_images,
    rank_models,
    match_datalake,
)
from hcaptcha_challenger.components.cv_toolkit import (
    annotate_objects,
    find_unique_object,
    find_unique_color,
)
from hcaptcha_challenger.components.middleware import Status, QuestionResp, RequestType, Answers
from hcaptcha_challenger.components.prompt_handler import handle
from hcaptcha_challenger.components.zero_shot_image_classifier import (
    ZeroShotImageClassifier,
    register_pipline,
)
from hcaptcha_challenger.onnx.modelhub import ModelHub, DataLake
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8, is_matched_ash_of_war, YOLOv8Seg


@dataclass
class RanniTheWitch:
    modelhub: ModelHub
    tmp_dir: Path
    self_supervised: bool = True

    prompt: str = ""
    label: str = ""
    ash: str = ""
    entity_type: str = ""

    img_paths: List[Path] = field(default_factory=list)
    example_paths: List[Path] = field(default_factory=list)

    qr: QuestionResp | None = None
    response: Answers | None = None

    @classmethod
    def summon_ranni_the_witch(
        cls,
        tmp_dir: Path | None = None,
        modelhub: ModelHub | None = None,
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

    def _match_model(self, select: Literal["yolo", "resnet"] = None) -> ResNetControl | YOLOv8:
        """match solution after `tactical_retreat`"""
        model = match_model(self.label, self.ash, self.modelhub, select=select)
        logger.debug("match model", select=select, prompt=self.prompt)

        return model

    def _match_solution(self, qr: QuestionResp):
        # Match: image_label_binary
        if qr.request_type == RequestType.ImageLabelBinary:
            if nested_models := self.modelhub.nested_categories.get(self.label, []):
                if model := self._rank_models(nested_models):
                    self.binary_challenge(model)
                elif self.self_supervised:
                    self.catch_all_binary_challenge()
                else:
                    return self.status.CHALLENGE_BACKCALL
            elif self.modelhub.label_alias.get(self.label):
                self.binary_challenge()
            elif self.self_supervised:
                self.catch_all_binary_challenge()
            else:
                return self.status.CHALLENGE_BACKCALL
        # Match: image_label_area_select
        elif qr.request_type == RequestType.ImageLabelAreaSelect:
            shape_type = qr.request_config.get("shape_type", "")

            if "default" in self.ash:
                if shape_type == "point":
                    self.keypoint_default_challenge()
                else:
                    logger.warning("unknown shape type", shape_type=shape_type, qr=qr)
                    return self.status.CHALLENGE_BACKCALL
            else:
                if not any(is_matched_ash_of_war(self.ash, c) for c in self.modelhub.yolo_names):
                    return self.status.CHALLENGE_BACKCALL
                if shape_type == "point":
                    self.keypoint_challenge()
                elif shape_type == "bounding_box":
                    self.bounding_challenge()
                else:
                    logger.warning("unknown shape type", shape_type=shape_type, qr=qr)
                    return self.status.CHALLENGE_BACKCALL
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

    def _rank_models(self, nested_models: List[str]) -> ResNetControl | None:
        result = rank_models(nested_models, self.example_paths, self.modelhub)
        if result and isinstance(result, tuple):
            best_model, model_name = result
            return best_model

    def _reset_executor(self, qr: QuestionResp):
        self.qr = qr

        self.response = Answers(c=json.dumps(qr.c), job_mode=qr.request_type)

        # parse label
        self.prompt = qr.requester_question.get("en")
        self.label = handle(self.prompt)

        # parse ash
        answer_keys = list(qr.requester_restricted_answer_set.keys())
        if len(answer_keys) > 0:
            self.entity_type = answer_keys[-1]
        else:
            self.entity_type = ""
        self.ash = f"{self.label} {self.entity_type}"

    async def _download_images(self, *, ignore_examples: bool = False):
        self.img_paths, self.example_paths = await download_challenge_images(
            self.qr, self.label, self.tmp_dir, ignore_examples
        )

    def binary_challenge(self, model: ResNetControl | None = None):
        classifier = model or self._match_model(select="resnet")

        answers = {}
        for i, img_path in enumerate(self.img_paths):
            result = None
            if img_path.stat().st_size:
                result = classifier.execute(img_stream=img_path.read_bytes())

            uid = self.qr.tasklist[i].task_key
            answers[uid] = "true" if result else "false"

        self.response.answers = answers

    def catch_all_binary_challenge(self):
        dl = match_datalake(self.modelhub, self.label)
        tool = ZeroShotImageClassifier.from_datalake(dl)

        model = register_pipline(self.modelhub)

        # {{< CATCH EXAMPLES >}}
        target = {}
        if self.example_paths:
            example_path = self.example_paths[-1]
            results = tool(model, image=Image.open(example_path))
            target = results[0]

        # {{< IMAGE CLASSIFICATION >}}
        answers = {}
        for i, img_path in enumerate(self.img_paths):
            results = tool(model, image=Image.open(img_path))

            if (
                results[0]["label"] in target.get("label", "")
                or results[0]["label"] in tool.positive_labels
            ):
                result = "true"
            else:
                result = "false"

            uid = self.qr.tasklist[i].task_key
            answers[uid] = result

        self.response.answers = answers

    def keypoint_default_challenge(self):
        def lookup_objects(_iter_launcher: Iterable) -> List[int] | None:
            count: int = 0
            deep: int = 4
            for focus_name, classes in _iter_launcher:
                count += 1
                session = self.modelhub.match_net(focus_name=focus_name)
                if "-seg" in focus_name:
                    detector = YOLOv8Seg.from_pluggable_model(session, classes)
                    res = detector(img_path, shape_type="point")
                else:
                    detector = YOLOv8.from_pluggable_model(session, classes)
                    res = detector(image, shape_type="point")
                self.modelhub.unplug()
                for name, (center_x, center_y), score in res:
                    point = [int(center_x), int(center_y)]
                    logger.debug("handle task", point=point, catch_model=focus_name, ash=self.ash)
                    return point
                if count > deep:
                    return

        def lookup_unique_object(trident) -> List[int] | None:
            model_name = self.modelhub.circle_segment_model
            classes = self.modelhub.ashes_of_war.get(model_name)
            session = self.modelhub.match_net(model_name)
            detector = YOLOv8Seg.from_pluggable_model(session, classes)
            results = detector(img_path, shape_type="point")
            self.modelhub.unplug()
            img, circles = annotate_objects(str(img_path))
            # Extract point coordinates
            if results:
                circles = [[int(result[1][0]), int(result[1][1]), 48] for result in results]
                logger.debug(
                    "handle task", select_model=model_name, trident=trident.__name__, ash=self.ash
                )
            if result := trident(img, circles):
                x, y, _ = result
                return [int(x), int(y)]

        answers = {}
        for i, img_path in enumerate(self.img_paths):
            image = img_path.read_bytes()

            coords: List[int] | None = []
            launcher = []

            if nested_models := self.modelhub.nested_categories.get(self.label, []):
                for model_name in nested_models:
                    element = model_name, self.modelhub.ashes_of_war.get(model_name, [])
                    launcher.append(element)
                if launcher:
                    coords = lookup_objects(launcher)
            elif "appears only once" in self.ash or "never repeated" in self.ash:
                coords = lookup_unique_object(trident=find_unique_object)
            elif "shapes are of the same color" in self.ash:
                coords = lookup_unique_object(trident=find_unique_color)
            else:
                launcher = self.modelhub.lookup_ash_of_war(self.ash)
                coords = lookup_objects(launcher)

            if not coords:
                z1 = cv2.imread(str(img_path)).shape[0]
                coords = [int(z1 / 2), int(z1 / 3)]

            uid = self.qr.tasklist[i].task_key
            answers[uid] = [
                {"entity_name": 0, "entity_type": self.entity_type, "entity_coords": coords}
            ]

        self.response.answers = answers

    def keypoint_challenge(self):
        detector: YOLOv8 = self._match_model(select="yolo")

        answers = {}
        for i, img_path in enumerate(self.img_paths):
            coords: List[int] | None = []
            res = detector(img_path, shape_type="point")

            # Subject: Select the object with the highest confidence
            entities = []
            for name, (center_x, center_y), score in res:
                if is_matched_ash_of_war(ash=self.ash, class_name=name):
                    entity = (score, [int(center_x), int(center_y)])
                    entities.append(entity)
            if len(entities) > 1:
                entities = sorted(entities, key=lambda x: x[0])
            if len(entities) > 0:
                coords = entities[-1][-1]

            uid = self.qr.tasklist[i].task_key
            answers[uid] = [
                {"entity_name": 0, "entity_type": self.entity_type, "entity_coords": coords}
            ]

    def bounding_challenge(self):
        detector: YOLOv8 = self._match_model(select="yolo")

        answers = {}
        for i, img_path in enumerate(self.img_paths):
            coords: List[int] | None = []

            res = detector(img_path, shape_type="bounding_box")

            # Subject: the entity that occupies the largest area of the image
            entities = []
            for name, (x1, y1), (x2, y2), score in res:
                if is_matched_ash_of_war(ash=self.ash, class_name=name):
                    scoop = (x2 - x1) * (y2 - y1)
                    entity = [scoop, int(x1), int(y1), int(x2), int(y2)]
                    entities.append(entity)
            if len(entities) > 1:
                entities = sorted(entities, key=lambda xf: xf[0])
            if len(entities) > 0:
                x1, y1, x2, y2 = entities[-1][1:]
                coords = [x1, y1, x2, y1, x2, y2, x1, y2]

            uid = self.qr.tasklist[i].task_key
            answers[uid] = [
                {"entity_name": 0, "entity_type": self.entity_type, "entity_coords": coords}
            ]

        self.response.answers = answers

    def multiple_choice_challenge(self):
        candidates = list(self.qr.requester_restricted_answer_set.keys())
        dl = DataLake.from_binary_labels(candidates[:1], candidates[1:])
        tool = ZeroShotImageClassifier.from_datalake(dl)

        model = register_pipline(self.modelhub)

        answers = {}
        for i, img_path in enumerate(self.img_paths):
            results = tool(model, image=Image.open(img_path))
            trusted_label = results[0]["label"]

            uid = self.qr.tasklist[i].task_key
            answers[uid] = [trusted_label]

        self.response.answers = answers


class AgentR(RanniTheWitch):
    async def execute(self, qr: QuestionResp, **kwargs) -> Status | Answers | None:
        # Match: ChallengePassed
        if not qr.requester_question.keys():
            logger.success("task done", reasone="The challenge has been completed")
            return self.status.CHALLENGE_SUCCESS

        self._reset_executor(qr)

        await self._download_images()

        tnt = self._match_solution(qr)

        if isinstance(tnt, self.status):
            return tnt
        return self.response
