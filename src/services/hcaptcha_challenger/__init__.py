# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from .core import ArmorCaptcha, ArmorUtils
from .solutions.resnet import ElephantsDrawnWithLeaves
from .solutions.resnet import ResNetBedroom
from .solutions.resnet import ResNetDomesticCat
from .solutions.resnet import ResNetSeaplane
from .solutions.resnet import ResNetBridge
from .solutions.resnet import ResNetLion
from .solutions.sk_recognition import SKRecognition
from .solutions.yolo import YOLO

__all__ = [
    "SKRecognition",
    "YOLO",
    "ArmorCaptcha",
    "ArmorUtils",
    "ElephantsDrawnWithLeaves",
    "ResNetSeaplane",
    "ResNetDomesticCat",
    "ResNetBedroom",
    "ResNetBridge",
    "ResNetLion",
]
