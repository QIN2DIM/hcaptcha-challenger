# -*- coding: utf-8 -*-
# Time       : 2022/3/2 0:52
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from typing import Literal

import cv2
import numpy as np
from onnxruntime import InferenceSession

from .utils import nms, xywh2xyxy, draw_detections, sigmoid, multiclass_nms


@dataclass
class YOLOv8:
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    classes: List[str] = field(default_factory=list)

    session: InferenceSession = None

    input_names = None
    input_shape = None
    input_height = None
    input_width = None
    output_names = None

    img_height = None
    img_width = None

    def __post_init__(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @classmethod
    def from_pluggable_model(cls, session: InferenceSession, classes: List[str]):
        return cls(session=session, classes=classes)

    def __call__(self, image: Path | bytes, shape_type: Literal["point", "bounding_box"] = "point"):
        if isinstance(image, Path):
            image = image.read_bytes()

        np_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_array, flags=1)

        boxes, scores, class_ids = self.detect_objects(image)

        response = []
        if shape_type == "point":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                center_x, center_y = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                response.append((self.classes[class_id], (center_x, center_y), scores[i]))
        elif shape_type == "bounding_box":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                point_start, point_end = (x1, y1), (x2, y2)
                response.append((self.classes[class_id], point_start, point_end, scores[i]))
        return response

    def detect_objects(self, image: np.ndarray):
        self.img_height, self.img_width = image.shape[:2]

        input_tensor = self._prepare_input(image)

        # Perform inference on the image
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        boxes, scores, class_ids = self._process_output(outputs)

        return boxes, scores, class_ids

    def _prepare_input(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def _process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]


def is_matched_ash_of_war(ash: str, class_name: str):
    if "head of " in ash:
        if "head" not in class_name:
            return False
        keyword = class_name.replace("-head", "").strip()
        if keyword not in ash:
            return False
        return True

    # catch-all rules
    if class_name not in ash:
        return False
    return True


def finetune_keypoint(name: str, point: List[int, int]) -> List[int, int]:
    point = point.copy()
    if name in ["nine", "9"]:
        point[-1] = point[-1] + 8
        point[0] = point[0] + 2
    if name in ["two", "2"]:
        point[-1] = point[-1] + 7
        point[0] = point[0] + 4
    return point


@dataclass
class YOLOv8Seg:
    conf_threshold: float = 0.71
    iou_threshold: float = 0.5
    num_masks: int = 32
    classes: List[str] = field(default_factory=list)
    session: InferenceSession = None

    input_names = None
    input_shape = None
    input_height = None
    input_width = None
    output_names = None

    img_height = None
    img_width = None
    boxes = None
    scores = None
    class_ids = None
    mask_maps = None

    colors = None

    def __post_init__(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        if not self.colors:
            rng = np.random.default_rng(3)
            self.colors = rng.uniform(0, 255, size=(len(self.classes), 3))

    @classmethod
    def from_pluggable_model(cls, session: InferenceSession, classes: List[str]):
        return cls(session=session, classes=classes)

    def __call__(
        self, image: Path | bytes, shape_type: Literal["point", "bounding_box"] = "point", **kwargs
    ):
        if isinstance(image, Path):
            image = image.read_bytes()

        np_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_array, flags=1)

        boxes, scores, class_ids, mask_maps = self.segment_objects(image)

        response = []
        if shape_type == "point":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                center_x, center_y = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                response.append((self.classes[class_id], (center_x, center_y), scores[i]))
        elif shape_type == "bounding_box":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                point_start, point_end = (x1, y1), (x2, y2)
                response.append((self.classes[class_id], point_start, point_end, scores[i]))
        return response

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(
            self.boxes, (self.img_height, self.img_width), (mask_height, mask_width)
        )

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
            )

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes, (self.input_height, self.input_width), (self.img_height, self.img_width)
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, mask_alpha=0.4):
        return draw_detections(
            image, self.boxes, self.scores, self.class_ids, self.colors, self.classes, mask_alpha
        )

    def draw_masks(self, image, mask_alpha=0.5):
        return draw_detections(
            image,
            self.boxes,
            self.scores,
            self.class_ids,
            self.colors,
            self.classes,
            mask_alpha,
            mask_maps=self.mask_maps,
        )

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes
