# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
from __future__ import annotations

import os
import typing
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from hcaptcha_challenger._solutions.kernel import ChallengeStyle
from hcaptcha_challenger._solutions.kernel import ModelHub

warnings.filterwarnings("ignore", category=UserWarning)


class ResNetFactory(ModelHub):
    def __init__(self, _onnx_prefix, _name, _dir_model: Path):
        super().__init__(_onnx_prefix, _name, _dir_model)
        self.register_model()

    def classifier(
        self,
        img_stream,
        feature_filters: typing.Union[typing.Callable, typing.List[typing.Callable]] = None,
    ):
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        if img.shape[0] == ChallengeStyle.WATERMARK:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        if feature_filters is not None:
            if not isinstance(feature_filters, list):
                feature_filters = [feature_filters]
            for tnt in feature_filters:
                if not tnt(img):
                    return False

        img = cv2.resize(img, (64, 64))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)

        # 使用延迟反射机制确保分布式网络的两端一致性
        net = self.match_net()
        if net is None:
            _err_prompt = f"""
            The remote network does not exist or the local cache has expired.
            1. Check objects.yaml for typos | model={self.fn};
            2. Restart the program after deleting the local cache | dir={self.assets.assets_dir};
            """
            logger.warning(_err_prompt)
            self.assets.sync()
            return False
        net.setInput(blob)
        out = net.forward()
        if not np.argmax(out, axis=1)[0]:
            return True
        return False

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        return self.classifier(img_stream, feature_filters=None)


class PluggableONNXModels:
    """
    Manage pluggable models. Provides high-level interfaces
    such as model download, model cache, and model scheduling.
    """

    def __init__(self, path_objects_yaml: Path, dir_model: Path, lang: str | None = "en"):
        self.dir_model = dir_model
        self.lang = lang
        self._fingers = []
        self._label_alias = {i: {} for i in ["zh", "en"]}
        self._register(path_objects_yaml)

    @property
    def label_alias(self) -> typing.Dict[str, str]:
        return self._label_alias.get(self.lang)

    def get_label_alias(self, lang):
        return self._label_alias.get(lang)

    @property
    def fingers(self) -> typing.List[str]:
        return self._fingers

    def _register(self, objects_path: Path):
        """
        Register pluggable ONNX models from `objects.yaml`.

        :rtype: List[str]
        :rtype: None
        """
        if not objects_path or not objects_path.exists():
            return

        with open(objects_path, "r", encoding="utf8") as file:
            data: typing.Dict[str, dict] = yaml.safe_load(file.read())

        if not data:
            os.remove(objects_path)
            return

        label_to_i18ndict = data.get("label_alias", {})
        if not label_to_i18ndict:
            return

        for model_label, i18n_to_raw_labels in label_to_i18ndict.items():
            self._fingers.append(model_label)
            for lang, prompt_labels in i18n_to_raw_labels.items():
                for prompt_label in prompt_labels:
                    self._label_alias[lang].update({prompt_label.strip(): model_label})

    def summon(self):
        """
        Download ONNX models from upstream repositories,
        skipping installed model files by default.

        :rtype: None
        """
        for finger in self._fingers:
            new_tarnished(finger, self.dir_model).pull_model()

    def overload(self) -> typing.Dict[str, ModelHub]:
        """
        Load the ONNX model into memory.
        Executed before the task starts.

        :rtype: Dict[str, ModelHub]
        """
        return {finger: new_tarnished(finger, self.dir_model) for finger in self._fingers}

    def lazy_loading(self, model_label: str) -> typing.Optional[ModelHub]:
        return new_tarnished(onnx_prefix=model_label, dir_model=self.dir_model)


def new_tarnished(onnx_prefix: str, dir_model: Path) -> ModelHub:
    """ResNet model factory, used to produce abstract model call interface."""
    return ResNetFactory(
        _onnx_prefix=onnx_prefix, _name=f"{onnx_prefix}(ResNet)_model", _dir_model=dir_model
    )
