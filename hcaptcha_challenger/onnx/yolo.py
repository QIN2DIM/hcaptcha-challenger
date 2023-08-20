# -*- coding: utf-8 -*-
# Time       : 2022/3/2 0:52
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from cv2.dnn import Net

from hcaptcha_challenger.onnx.modelhub import ChallengeStyle


class Prefix:
    YOLOv5s6 = "yolov5s6"
    YOLOv5m6 = "yolov5m6"
    YOLOv5n6 = "yolov5n6"
    YOLOv6n = "yolov6n"
    YOLOv6s = "yolov6s"
    YOLOv6t = "yolov6t"


@dataclass
class YOLO:
    """YOLO model for image classification"""

    net: Net
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

    @classmethod
    def from_pluggable_model(cls, net):
        return cls(net=net)

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

    def execute(self, img_stream: bytes, label: str, **kwargs) -> bool:
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
