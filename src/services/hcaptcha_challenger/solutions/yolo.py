# -*- coding: utf-8 -*-
# Time       : 2022/3/2 0:52
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os

import cv2
import numpy as np

from .kernel import ChallengeStyle
from .kernel import Solutions


class YOLO:
    """YOLO model for image classification"""

    classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self, dir_model: str = None, onnx_prefix: str = None):
        self.dir_model = "./model" if dir_model is None else dir_model

        # Select default onnx model.
        self.onnx_prefix = (
            "yolov5s6"
            if onnx_prefix
            not in [
                # Reference - Ultralytics YOLOv5 https://github.com/ultralytics/yolov5
                "yolov5m6",
                "yolov5s6",
                "yolov5n6",
                # Reference - MT-YOLOv6 https://github.com/meituan/YOLOv6
                "yolov6n",
                "yolov6s",
                "yolov6t",
                # "yolov7"  # Vision Transformer
            ]
            else onnx_prefix
        )

        self.name = f"YOLOv5{self.onnx_prefix[-2:]}"
        if self.onnx_prefix.startswith("yolov6"):
            self.name = f"MT-YOLOv6{self.onnx_prefix[-1]}"

        self.onnx_model = {
            "name": f"{self.name}(ONNX)_model",
            "path": os.path.join(self.dir_model, f"{self.onnx_prefix}.onnx"),
            "src": f"https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{self.onnx_prefix}.onnx",
        }

        self.flag = self.onnx_model["name"]

        self.download_model()
        self.net = cv2.dnn.readNetFromONNX(self.onnx_model["path"])

    def download_model(self):
        """Download YOLOv5(ONNX) model"""
        Solutions.download_model_(
            dir_model=self.dir_model,
            path_model=self.onnx_model["path"],
            model_src=self.onnx_model["src"],
            model_name=self.onnx_model["name"],
            upgrade=False,
        )

    def detect_common_objects(self, img: np.ndarray, confidence=0.4, nms_thresh=0.4):
        """
        Object Detection

        Get multiple labels identified in a given image

        :param img:
        :param confidence:
        :param nms_thresh:
        :return: bbox, label, conf
        """
        height, width = img.shape[:2]

        class_ids = []
        confidences = []
        boxes = []

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                max_conf = scores[class_id]
                if max_conf > confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - (w / 2)
                    y = center_y - (h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(max_conf))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

        return [str(self.classes[class_ids[i]]) for i in indices]

    def solution(self, img_stream: bytes, label: str, **kwargs) -> bool:
        """
        Implementation process of solution.

         with open(img_filepath, "rb") as file:
            data = file.read()
         solution(img_stream=data, label="truck")

        :param img_stream: image file binary stream
        :param label:
        :param kwargs:
        :return:
        """
        confidence = kwargs.get("confidence", 0.4)
        nms_thresh = kwargs.get("nms_thresh", 0.4)

        np_array = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(np_array, flags=1)
        img = (
            cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            if img.shape[0] == ChallengeStyle.WATERMARK
            else img
        )
        try:
            labels = self.detect_common_objects(img, confidence, nms_thresh)
            return bool(label in labels)
        # patch for `ValueError: attempt to get argmax of an empty sequence.`
        # at code `class_id=np.argmax(scores)`
        except ValueError:
            return False
