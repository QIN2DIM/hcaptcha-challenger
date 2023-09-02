# -*- coding: utf-8 -*-
# Time       : 2022/3/2 0:52
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np
from onnxruntime import InferenceSession


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
        # get_input_details
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # get_output_details
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


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
