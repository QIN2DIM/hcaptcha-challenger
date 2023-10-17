# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np
from cv2.dnn import Net
from loguru import logger
from scipy.special import expit


class ChallengeStyle:
    WATERMARK = 144  # onTrigger 128x128
    GENERAL = 128
    GAN = 144


@dataclass
class ResNetControl:
    net: Net

    @classmethod
    def from_pluggable_model(cls, net: Net):
        return cls(net=net)

    def binary_classify(self, img_stream: Any) -> bool | Tuple[bool, np.ndarray]:
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        if img.shape[0] == ChallengeStyle.WATERMARK:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        img = cv2.resize(img, (64, 64))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)

        # Use the delayed reflection mechanism
        # to ensure the consistency of both ends of the distributed network
        if self.net is None:
            logger.debug("The remote network does not exist or the local cache has expired.")
            return False
        self.net.setInput(blob)
        out = self.net.forward()
        logit = out[0]
        proba = expit(logit)
        if not np.argmax(out, axis=1)[0]:
            return True, proba
        return False, proba

    def execute(self, img_stream: bytes, **kwargs) -> bool | Tuple[bool, np.ndarray]:
        """Implementation process of solution"""
        get_proba = kwargs.get("proba", False)
        result = self.binary_classify(img_stream)
        if result is None or isinstance(result, bool):
            return result
        if isinstance(result, tuple):
            prediction, confidence = result
            if get_proba is True:
                return prediction, confidence
            return prediction
