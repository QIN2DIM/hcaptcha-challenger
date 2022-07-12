# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import os
import time
import warnings
from typing import List, Callable, Union, Optional

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2

from .kernel import ChallengeStyle
from .kernel import Solutions

warnings.filterwarnings("ignore", category=UserWarning)


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
            time.sleep(0.3)
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

        self.download_model()

        img = cv2.resize(img, (64, 64))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)

        net = cv2.dnn.readNetFromONNX(self.onnx_model["path"])

        net.setInput(blob)

        out = net.forward()

        if not np.argmax(out, axis=1)[0]:
            return True
        return False

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""


class ResNetDomesticCat(ResNetFactory):
    """Handle challenge 「domestic cat」"""

    def __init__(self, dir_model: str, path_rainbow=None):
        _onnx_prefix = "domestic_cat"
        self.rainbow_key = _onnx_prefix
        super().__init__(_onnx_prefix, f"{_onnx_prefix}(ResNet)_model", dir_model, path_rainbow)

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        return self.classifier(img_stream, self.rainbow_key, feature_filters=None)


class ResNetSeaplane(ResNetFactory):
    """Handle challenge 「seaplane」"""

    def __init__(self, dir_model: str, path_rainbow=None):
        _onnx_prefix = "seaplane"
        self.rainbow_key = _onnx_prefix
        super().__init__(_onnx_prefix, f"{_onnx_prefix}(ResNet)_model", dir_model, path_rainbow)

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        return self.classifier(img_stream, self.rainbow_key, feature_filters=None)


class ElephantsDrawnWithLeaves(ResNetFactory):
    """Handle challenge 「Please select all the elephants drawn with leaves」"""

    def __init__(self, dir_model, path_rainbow=None):
        _onnx_prefix = "elephants_drawn_with_leaves"
        self.rainbow_key = _onnx_prefix
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
        """Implementation process of solution"""
        return self.classifier(
            img_stream, self.rainbow_key, feature_filters=self.is_drawn_with_leaves
        )


class HorsesDrawnWithFlowers(ResNetFactory):
    """Handle challenge「Please select all the horses drawn with flowers」"""

    def __init__(self, dir_model, path_rainbow=None):
        _onnx_prefix = "horses_drawn_with_flowers"
        self.rainbow_key = _onnx_prefix
        super().__init__(
            _onnx_prefix, f"{_onnx_prefix}(de-stylized)_model", dir_model, path_rainbow
        )

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
