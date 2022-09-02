# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import os
import warnings
from typing import List, Callable, Union, Dict

import cv2
import numpy as np
import yaml

from .kernel import ChallengeStyle
from .kernel import ModelHub

warnings.filterwarnings("ignore", category=UserWarning)


class ResNetFactory(ModelHub):
    def __init__(self, _onnx_prefix, _name, _dir_model: str, on_rainbow: bool = None):
        super().__init__(_onnx_prefix, _name, _dir_model, on_rainbow)
        self.register_model()

    def classifier(
        self, img_stream, rainbow_key, feature_filters: Union[Callable, List[Callable]] = None
    ):
        if hasattr(self, "rainbow"):
            matched = self.rainbow.match(img_stream, rainbow_key)
            if matched is not None:
                return matched

        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        # fixme: dup-code
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
            _err_prompt = """
            The remote network does not exist or the local cache has expired.
            1. Check objects.yaml for typos | model={};
            2. Restart the program after deleting the local cache | dir={};
            """.format(
                self.fn, self.assets.dir_assets
            )
            raise ResourceWarning(_err_prompt)
        net.setInput(blob)
        out = net.forward()
        if not np.argmax(out, axis=1)[0]:
            return True
        return False

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""


class FingersOfTheGoldenOrder(ResNetFactory):
    """ResNet model factory, used to produce abstract model call interface."""

    def __init__(self, onnx_prefix: str, dir_model: str, on_rainbow=None):
        self.rainbow_key = onnx_prefix
        super().__init__(onnx_prefix, f"{onnx_prefix}(ResNet)_model", dir_model, on_rainbow)

    def solution(self, img_stream, **kwargs) -> bool:
        return self.classifier(img_stream, self.rainbow_key, feature_filters=None)


class PluggableONNXModels:
    """
    Manage pluggable models. Provides high-level interfaces
    such as model download, model cache, and model scheduling.
    """

    def __init__(self, path_objects_yaml: str):
        self.fingers = []
        self.label_alias = {i: {} for i in ["zh", "en"]}
        self._register(path_objects_yaml)

    def _register(self, path_objects_yaml):
        """
        Register pluggable ONNX models from `objects.yaml`.

        :type path_objects_yaml: str
        :rtype: List[str]
        :rtype: None
        """
        if not path_objects_yaml or not os.path.exists(path_objects_yaml):
            return

        with open(path_objects_yaml, "r", encoding="utf8") as file:
            data: Dict[str, dict] = yaml.safe_load(file.read())

        label_to_i18ndict = data.get("label_alias", {})
        if not label_to_i18ndict:
            return

        for model_label, i18n_to_raw_labels in label_to_i18ndict.items():
            self.fingers.append(model_label)
            for lang, prompt_labels in i18n_to_raw_labels.items():
                for prompt_label in prompt_labels:
                    self.label_alias[lang].update({prompt_label.strip(): model_label})

    def summon(self, dir_model):
        """
        Download ONNX models from upstream repositories,
        skipping installed model files by default.

        :type dir_model: str
        :rtype: None
        """
        for finger in self.fingers:
            FingersOfTheGoldenOrder(finger, dir_model, on_rainbow=None).pull_model()

    def overload(self, dir_model, on_rainbow=None):
        """
        Load the ONNX model into memory.
        Executed before the task starts.

        :type dir_model: str
        :type on_rainbow: bool | None
        :rtype: Dict[str, FingersOfTheGoldenOrder]
        """
        return {
            finger: FingersOfTheGoldenOrder(finger, dir_model, on_rainbow)
            for finger in self.fingers
        }
