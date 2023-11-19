# -*- coding: utf-8 -*-
# Time       : 2023/11/18 22:38
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import hashlib
import shutil
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

from hcaptcha_challenger.components.image_downloader import Cirilla
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8
from hcaptcha_challenger.components.middleware import QuestionResp


def match_solution(
    label: str, ash: str, modelhub: ModelHub, select: Literal["yolo", "resnet"] = None
) -> ResNetControl | YOLOv8:
    """match solution after `tactical_retreat`"""
    focus_label = modelhub.label_alias.get(label, "")

    # Match YOLOv8 model
    if not focus_label or select == "yolo":
        focus_name, classes = modelhub.apply_ash_of_war(ash=ash)
        session = modelhub.match_net(focus_name=focus_name)
        detector = YOLOv8.from_pluggable_model(session, classes)
        return detector

    # Match ResNet model
    focus_name = focus_label
    if not focus_name.endswith(".onnx"):
        focus_name = f"{focus_name}.onnx"
    net = modelhub.match_net(focus_name=focus_name)
    control = ResNetControl.from_pluggable_model(net)
    return control


async def download_challenge_images(
    qr: QuestionResp, label: str, tmp_dir: Path, ignore_examples: bool = False
):
    request_type = qr.request_type
    ks = list(qr.requester_restricted_answer_set.keys())

    inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
    for c in inv:
        label = label.replace(c, "")
    label = label.strip()

    if len(ks) > 0:
        typed_dir = tmp_dir.joinpath(request_type, label, ks[0])
    else:
        typed_dir = tmp_dir.joinpath(request_type, label)
    typed_dir.mkdir(parents=True, exist_ok=True)

    ciri = Cirilla()
    container = []
    tasks = []
    for i, tk in enumerate(qr.tasklist):
        challenge_img_path = typed_dir.joinpath(f"{time.time()}.{i}.png")
        context = (challenge_img_path, tk.datapoint_uri)
        container.append(context)
        tasks.append(asyncio.create_task(ciri.elder_blood(context)))

    examples = []
    if not ignore_examples:
        with suppress(Exception):
            for i, uri in enumerate(qr.requester_question_example):
                example_img_path = typed_dir.joinpath(f"{time.time()}.exp.{i}.png")
                context = (example_img_path, uri)
                examples.append(context)
                tasks.append(asyncio.create_task(ciri.elder_blood(context)))

    await asyncio.gather(*tasks)

    # Optional deduplication
    _img_paths = []
    for src, _ in container:
        cache = src.read_bytes()
        dst = typed_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
        shutil.move(src, dst)
        _img_paths.append(dst)

    # Optional deduplication
    _example_paths = []
    if examples:
        for src, _ in examples:
            cache = src.read_bytes()
            dst = typed_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
            shutil.move(src, dst)
            _example_paths.append(dst)

    return _img_paths, _example_paths
