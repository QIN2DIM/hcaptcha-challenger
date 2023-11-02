# -*- coding: utf-8 -*-
# Time       : 2023/10/20 17:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: zero-shot image classification
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List, Literal, Iterable, Tuple

import onnxruntime
from PIL.Image import Image

from hcaptcha_challenger.components.prompt_handler import handle
from hcaptcha_challenger.onnx.clip import MossCLIP
from hcaptcha_challenger.onnx.modelhub import ModelHub, DataLake
from hcaptcha_challenger.onnx.utils import is_cuda_pipline_available


def register_pipline(
    modelhub: ModelHub,
    *,
    fmt: Literal["onnx", "transformers"] = None,
    install_only: bool = False,
    **kwargs,
):
    """
    Ace Model:
        - laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K --> ONNX 1.7GB
        - QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_224to336 --> ONNX

    :param install_only:
    :param modelhub:
    :param fmt:
    :param kwargs:
    :return:
    """
    if fmt in ["transformers", None]:
        fmt = "transformers" if is_cuda_pipline_available else "onnx"

    if fmt in ["onnx"]:
        v_net, t_net = None, None

        if not modelhub.label_alias:
            modelhub.parse_objects()

        if visual_path := kwargs.get("visual_path"):
            if not isinstance(visual_path, Path):
                raise ValueError("visual_path should be a pathlib.Path")
            if not visual_path.is_file():
                raise FileNotFoundError(
                    "Select to use visual ONNX model, but the specified model does not exist -"
                    f" {visual_path=}"
                )
            v_net = onnxruntime.InferenceSession(
                visual_path, providers=onnxruntime.get_available_providers()
            )
        if textual_path := kwargs.get("textual_path"):
            if not isinstance(textual_path, Path):
                raise ValueError("textual_path should be a pathlib.Path")
            if not textual_path.is_file():
                raise FileNotFoundError(
                    "Select to use textual ONNX model, but the specified model does not exist -"
                    f" {textual_path=}"
                )
            t_net = onnxruntime.InferenceSession(
                textual_path, providers=onnxruntime.get_available_providers()
            )

        if not v_net:
            visual_model = kwargs.get("visual_model", modelhub.DEFAULT_CLIP_VISUAL_MODEL)
            v_net = modelhub.match_net(visual_model, install_only=install_only)
        if not t_net:
            textual_model = kwargs.get("textual_model", modelhub.DEFAULT_CLIP_TEXTUAL_MODEL)
            t_net = modelhub.match_net(textual_model, install_only=install_only)

        if not install_only:
            _pipeline = MossCLIP.from_pluggable_model(v_net, t_net)
            return _pipeline

    if fmt in ["transformers"]:
        from transformers import pipeline  # type:ignore

        import torch  # type:ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = kwargs.get("checkpoint", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
        task = kwargs.get("task", "zero-shot-image-classification")
        batch_size = kwargs.get("batch_size", 16)
        _pipeline = pipeline(task=task, device=device, model=checkpoint, batch_size=batch_size)
        return _pipeline


def format_datalake(dl: DataLake) -> Tuple[List[str], List[str]]:
    positive_labels = dl.positive_labels.copy()
    negative_labels = dl.negative_labels.copy()

    # When the input is a challenge prompt, cut it into phrases
    if dl.raw_prompt:
        prompt = dl.raw_prompt
        prompt = prompt.replace("_", " ")
        true_label = handle(prompt)
        if true_label not in positive_labels:
            positive_labels.append(true_label)
        if not negative_labels:
            false_label = dl.PREMISED_BAD.format(true_label)
            negative_labels.append(false_label)

    # Insert hypothesis_template
    for labels in [positive_labels, negative_labels]:
        for i, label in enumerate(labels):
            if "This is a" in label:
                continue
            labels[i] = dl.PREMISED_YES.format(label)

    # Formatting model input
    candidate_labels = positive_labels.copy()
    if isinstance(negative_labels, list) and len(negative_labels) != 0:
        candidate_labels.extend(negative_labels)

    return positive_labels, candidate_labels


@dataclass
class ZeroShotImageClassifier:
    positive_labels: List[str] = field(default_factory=list)
    candidate_labels: List[str] = field(default_factory=list)

    @classmethod
    def from_datalake(cls, dl: DataLake):
        positive_labels, candidate_labels = format_datalake(dl)
        return cls(positive_labels=positive_labels, candidate_labels=candidate_labels)

    def __call__(self, detector: MossCLIP, image: Image, *args, **kwargs):
        if isinstance(detector, MossCLIP) and not isinstance(image, Iterable):
            image = [image]
        predictions = detector(image, candidate_labels=self.candidate_labels)
        return predictions
