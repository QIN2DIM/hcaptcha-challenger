# -*- coding: utf-8 -*-
# Time       : 2022/3/2 0:52
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os

import cv2
import numpy as np

from .kernel import Solutions


class YOLO:
    """YOLO model for image classification"""

    def __init__(self, dir_model: str = None, onnx_prefix: str = "yolov5s6"):
        self.dir_model = "./model" if dir_model is None else dir_model
        self.onnx_prefix = "yolov5s6" if onnx_prefix not in ["yolov5m6", "yolov5s6", "yolov5n6"] else onnx_prefix

        self.onnx_model = {
            "name": f"{self.onnx_prefix}(onnx)_model",
            "path": os.path.join(self.dir_model, f"{self.onnx_prefix}.onnx"),
            "src": f"https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/{self.onnx_prefix}.onnx",
        }

        self.flag = self.onnx_model["name"]

        # COCO namespace
        self.classes = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
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
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
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

    def download_model(self):
        """Download YOLOv5(ONNX) model"""
        Solutions.download_model_(
            dir_model=self.dir_model,
            path_model=self.onnx_model["path"],
            model_src=self.onnx_model["src"],
            model_name=self.onnx_model["name"],
        )

    def detect_common_objects(self, img, confidence=0.4, nms_thresh=0.4):
        """
        Object Detection

        Get multiple labels identified in a given image

        :param img_stream: image file binary stream
             with open(img_filepath, "rb") as file:
                data = file.read()
             detect_common_objects(img_stream=data)
        :param confidence:
        :param nms_thresh:
        :return: bbox, label, conf
        """
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True, crop=False)
        self.download_model()

        net = cv2.dnn.readNetFromONNX(self.onnx_model["path"])

        net.setInput(blob)

        class_ids = []
        confidences = []
        boxes = []

        outs = net.forward()

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
        """Implementation process of solution"""
        confidence = kwargs.get("confidence", 0.4)
        nms_thresh = kwargs.get("nms_thresh", 0.4)
        img = self.stream_to_nparray(img_stream)
        labels = self.detect_common_objects(img, confidence, nms_thresh)
        return bool(label in labels)

    def solution_dev(self, src_dir: str, **kwargs):
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"{src_dir} not found")
            return
        _suffix = ".png"
        for _prefix, _, files in os.walk(src_dir):
            for filename in files:
                if not filename.endswith(_suffix):
                    continue
                path_img = os.path.join(_prefix, filename)
                with open(path_img, "rb") as file:
                    yield path_img, self.solution(file.read(), **kwargs)

    @staticmethod
    def stream_to_nparray(img_stream: bytes) -> np.ndarray:
        """Convert a stream to numpy array"""
        np_array = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(np_array, flags=1)
        return img


class YOLOWithAugmentation(YOLO):
    """YOLO model with data augmentation"""

    def __init__(self, dir_model: str = None, onnx_prefix: str = "yolov5s6"):
        super().__init__(dir_model, onnx_prefix)

    def solution(self, img_stream: bytes, label: str, **kwargs) -> bool:
        """Implementation process of solution"""
        confidence = kwargs.get("confidence", 0.4)
        nms_thresh = kwargs.get("nms_thresh", 0.4)
        img = self.stream_to_nparray(img_stream)
        # cv2.imshow("img", img)
        # denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        # cv2.imshow("img_denoise", img)
        # cv2.waitKey(0)

        labels = self.detect_common_objects(img, confidence, nms_thresh)
        return bool(label in labels)


if __name__ == "__main__":
    sol = YOLOWithAugmentation()

    yes_path = os.path.join(".", "database", "airplane", "yes")
    for path_, resp in sol.solution_dev(yes_path, label="aeroplane"):
        print(path_, resp)

    print("-" * 20)

    no_path = os.path.join(".", "database", "airplane", "bad")
    for path_, resp in sol.solution_dev(no_path, label="aeroplane"):
        print(path_, resp)
