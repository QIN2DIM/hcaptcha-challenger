# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import os
import warnings
from typing import List, Callable, Union, Optional

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2

from .kernel import ChallengeStyle
from .kernel import Solutions

warnings.filterwarnings("ignore", category=UserWarning)


class _Fingers:
    ELEPHANTS_DRAWN_WITH_LEAVES = "elephants drawn with leaves"
    HORSES_DRAWN_WITH_FLOWERS = "horses drawn with flowers"
    SEAPLANE = "seaplane"
    DOMESTIC_CAT = "domestic cat"
    BEDROOM = "bedroom"
    BRIDGE = "bridge"
    LION = "lion"
    HORSE_WITH_WHITE_LEGS = "horse with white legs"
    LION_YAWNING_WITH_OPEN_MOUTH = "lion yawning with open mouth"
    LION_WITH_CLOSED_EYES = "lion with closed eyes"
    ELEPHANT_WITH_LONG_TUSK = "elephant with long tusk"
    PARROT_BIRD_WITH_EYES_OPEN = "parrot bird with eyes open"
    HORSE = "horse"
    LIVING_ROOM = "living room"
    SMILING_DOG = "smiling dog"
    GIRAFFE = "giraffe"


class ResNetFactory(Solutions):
    def __init__(self, _onnx_prefix, _name, _dir_model: str, path_rainbow=None):
        """

        :param _name: 日志打印显示的标记
        :param _dir_model: 模型所在的本地目录
        :param _onnx_prefix: 模型文件名，远程仓库文件和本地的一致。也用于拼接下载链接，因此该参数不允许用户自定义，
          仅支持在范围内选择。
        :param path_rainbow: 彩虹表本地路径，可选。
        """
        super().__init__(_name, path_rainbow=path_rainbow)
        self.dir_model = _dir_model
        self.onnx_model = {
            "name": _name,
            "path": os.path.join(_dir_model, f"{_onnx_prefix}.onnx"),
            "src": f"https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{_onnx_prefix}.onnx",
        }

        self.download_model()
        self.net = cv2.dnn.readNetFromONNX(self.onnx_model["path"])

    def download_model(self, upgrade: Optional[bool] = None):
        """Download the ResNet ONNX classification model"""
        Solutions.download_model_(
            dir_model=self.dir_model,
            path_model=self.onnx_model["path"],
            model_src=self.onnx_model["src"],
            model_name=self.onnx_model["name"],
            upgrade=upgrade,
        )

    def classifier(
        self, img_stream, rainbow_key, feature_filters: Union[Callable, List[Callable]] = None
    ):
        match_output = self.match_rainbow(img_stream, rainbow_key)
        if match_output is not None:
            return match_output

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

        self.net.setInput(blob)
        out = self.net.forward()

        if not np.argmax(out, axis=1)[0]:
            return True
        return False

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""


class FingersOfTheGolderOrder(ResNetFactory):
    def __init__(self, rainbow_key: str, _onnx_prefix: str, dir_model: str, path_rainbow=None):
        self.rainbow_key = rainbow_key
        super().__init__(_onnx_prefix, f"{_onnx_prefix}(ResNet)_model", dir_model, path_rainbow)

    def solution(self, img_stream, **kwargs) -> bool:
        return self.classifier(img_stream, self.rainbow_key, feature_filters=None)


class ElephantsDrawnWithLeaves(ResNetFactory):
    """Handle challenge 「Please select all the elephants drawn with leaves」"""

    def __init__(self, dir_model, path_rainbow=None):
        _onnx_prefix = "elephants_drawn_with_leaves"
        self.rainbow_key = _onnx_prefix.replace("_", " ")
        super().__init__(
            _onnx_prefix, f"{_onnx_prefix}(de-stylized)_model", dir_model, path_rainbow
        )

    @staticmethod
    def is_drawn_with_leaves(img) -> bool:
        img = np.array(img)

        img = img.reshape((img.shape[0] * img.shape[1], img.shape[2])).astype(np.float64)
        centroid, label = kmeans2(img, k=3)

        green_centroid = np.array([0.0, 255.0, 0.0])

        min_dis = np.inf
        for i, _ in enumerate(centroid):
            min_dis = min(min_dis, np.linalg.norm(centroid[i] - green_centroid))

        if min_dis < 200:
            return True
        return False

    def solution(self, img_stream, **kwargs) -> bool:
        return self.classifier(
            img_stream, self.rainbow_key, feature_filters=self.is_drawn_with_leaves
        )


class HorsesDrawnWithFlowers(ResNetFactory):
    """Handle challenge「Please select all the horses drawn with flowers」"""

    def __init__(self, dir_model, path_rainbow=None):
        _onnx_prefix = "horses_drawn_with_flowers"
        self.rainbow_key = _onnx_prefix.replace("_", " ")
        super().__init__(
            _onnx_prefix, f"{_onnx_prefix}(de-stylized)_model", dir_model, path_rainbow
        )

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""


class PluggableONNXModel:
    def __init__(self):
        # registered service
        self.fingers = [
            _Fingers.SEAPLANE,
            _Fingers.DOMESTIC_CAT,
            _Fingers.BEDROOM,
            _Fingers.BRIDGE,
            _Fingers.LION,
            _Fingers.LIVING_ROOM,
            _Fingers.HORSE,
        ]
        self.talismans = {
            # _Fingers.elephants_drawn_with_leaves: ElephantsDrawnWithLeaves,
            # _Fingers.horses_drawn_with_flowers: HorsesDrawnWithFlowers,
        }

    def summon(self, dir_model, path_rainbow=None, upgrade: Optional[bool] = None):
        for finger in self.fingers:
            model = FingersOfTheGolderOrder(
                finger, finger.replace(" ", "_"), dir_model, path_rainbow
            )
            model.download_model(upgrade)
        for model in self.talismans.values():
            model(dir_model=dir_model).download_model(upgrade)

    def overload(self, dir_model, path_rainbow=None) -> Optional[dict]:
        pluggable_onnx_model = {}
        for i, finger in enumerate(self.fingers):
            onnx_prefix = finger.replace(" ", "_")
            print(
                f"OVERLOAD [PluggableONNXModel] - progress=[{i + 1}/{len(self.fingers)}] finger={onnx_prefix} "
            )
            model = FingersOfTheGolderOrder(finger, onnx_prefix, dir_model, path_rainbow)
            pluggable_onnx_model[finger] = model
        for i, model in enumerate(self.talismans):
            print(
                f"OVERLOAD [PluggableONNXModel] - progress=[{i + 1}/{len(self.talismans)}] finger={model}"
            )
            pluggable_onnx_model[model] = self.talismans[model]
        return pluggable_onnx_model

    def black_knife(self, label_alias: str, dir_model, path_rainbow: Optional[str] = None):
        """
        Use to summon the spirit of Black Knife Tiche.

        :param label_alias:
        :param dir_model:
        :param path_rainbow:
        :return:
        """

        label_alias = label_alias.strip()

        # 1.Return to standard paradigm model
        if label_alias in self.fingers:
            return FingersOfTheGolderOrder(
                rainbow_key=label_alias,
                _onnx_prefix=label_alias.replace(" ", "_"),
                dir_model=dir_model,
                path_rainbow=path_rainbow,
            )

        # 2.Returns a heterogeneous model that needs to be preprocessed
        if self.talismans.get(label_alias):
            return self.talismans[label_alias](dir_model=dir_model, path_rainbow=path_rainbow)

    def mimic_tear(self):
        """
        This spirit takes the form of the summoner to fight alongside them,
        but its mimicry does not extend to imitating the summoner's will.
        :return:
        """
