# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import os
import time
import warnings
from typing import Optional

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2

from .kernel import Solutions

warnings.filterwarnings("ignore", category=UserWarning)


class DeStylized(Solutions):
    def __init__(self, dir_model: str, onnx_prefix: str, path_rainbow: Optional[str] = None):
        super(DeStylized, self).__init__(flag="de-stylized", path_rainbow=path_rainbow)

        self.dir_model = "./model" if dir_model is None else dir_model

        self.onnx_prefix = onnx_prefix

        self.onnx_model = {
            "name": f"{self.onnx_prefix}(de-stylized)_model",
            "path": os.path.join(self.dir_model, f"{self.onnx_prefix}.onnx"),
            "src": f"https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{self.onnx_prefix}.onnx",
        }

        self.flag = self.onnx_model["name"]

    def download_model(self):
        """Download the de-stylized binary classification model"""
        Solutions.download_model_(
            dir_model=self.dir_model,
            path_model=self.onnx_model["path"],
            model_src=self.onnx_model["src"],
            model_name=self.onnx_model["name"],
        )

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError


class HorsesDrawnWithFlowers(DeStylized):
    """Handle challenge「Please select all the horses drawn with flowers」"""

    def __init__(self, dir_model, path_rainbow=None):
        super().__init__(
            dir_model=dir_model, path_rainbow=path_rainbow, onnx_prefix="horses_drawn_with_flowers"
        )
        self.rainbow_key = "horses drawn with flowers"

    @staticmethod
    def is_drawn_with_flowers(img) -> bool:
        """Work in progress"""

    def classifier(self, img_stream) -> bool:
        """Work in progress"""

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""


class ElephantsDrawnWithLeaves(DeStylized):
    """Handle challenge 「Please select all the elephants drawn with leaves」"""

    def __init__(self, dir_model, path_rainbow=None):
        super().__init__(
            dir_model=dir_model,
            path_rainbow=path_rainbow,
            onnx_prefix="elephants_drawn_with_leaves",
        )
        self.rainbow_key = "elephants drawn with leaves"

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

    def classifier(self, img_stream) -> bool:
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        # de-stylized
        if not self.is_drawn_with_leaves(img):
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
        match_output = self.match_rainbow(img_stream, rainbow_key=self.rainbow_key)
        if match_output is not None:
            time.sleep(0.3)
            return match_output
        return self.classifier(img_stream)
