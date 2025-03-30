# -*- coding: utf-8 -*-
# Time       : 2023/10/15 21:37
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os

import cv2
import importlib_metadata
import numpy as np

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_package_available(pkg_name: str) -> bool:
    try:
        package_version = importlib_metadata.metadata(pkg_name) is not None
        return bool(package_version)
    except importlib_metadata.PackageNotFoundError:
        return False


_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = _is_package_available("torch")
else:
    _torch_available = False

_transformers_available = _is_package_available("transformers")


def is_torch_available():
    return _torch_available


def is_transformers_available():
    return _transformers_available


is_cuda_pipline_available = is_torch_available() and is_transformers_available()


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_detections(
    image, boxes, scores, class_ids, colors, classes, mask_alpha=0.3, mask_maps=None
):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, colors, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = classes[class_id]
        caption = f"{label} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(
            text=caption,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness,
        )
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(
            mask_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return mask_img


def draw_masks(image, boxes, class_ids, colors, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(
        text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontsize, thickness=text_thickness
    )
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img1,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (0, 115, 255),
        -1,
    )
    cv2.putText(
        img1, name1, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), text_thickness
    )

    (tw, th), _ = cv2.getTextSize(
        text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontsize, thickness=text_thickness
    )
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img2,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (94, 23, 235),
        -1,
    )

    cv2.putText(
        img2, name2, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), text_thickness
    )

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img
