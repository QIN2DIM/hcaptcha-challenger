# -*- coding: utf-8 -*-
# Time       : 2023/9/12 14:45
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import re
from pathlib import Path

import pytest

from hcaptcha_challenger import split_prompt_message, label_cleaning

pattern = re.compile(r"[^\x00-\x7F]")

prompts = []

qa_data_path = Path(__file__).parent.joinpath("qa_data.txt")
if qa_data_path.exists():
    prompts = qa_data_path.read_text(encoding="utf8").split("\n")


@pytest.mark.parametrize("prompt", prompts)
def test_split_prompt_message(prompt: str):
    result = split_prompt_message(prompt, lang="en")
    assert result != prompt


@pytest.mark.parametrize("prompt", prompts)
def test_is_illegal_chars(prompt: str):
    result = label_cleaning(prompt)
    assert not pattern.search(result)


def test_split_area_select_prompt():
    prompt = "Please click on the thumbnail of something that can be eaten"
    model_name = "can_be_eaten_2309_yolov8s.onnx"
    binder = " ".join(model_name.split("_")[:-2])
    assert binder in prompt


@pytest.mark.parametrize(
    "model_name",
    [
        "can_be_eaten_2309_yolov8s.onnx",
        "notanimal2310_yolov8s.onnx",
        "animalhead_zebra_yolov8s.onnx",
        "fantasia_starfish_yolov8n.onnx",
        "COCO2020_yolov8m.onnx",
    ],
)
def test_prefix_binder(model_name: str):
    binder = model_name.split("_")
    if len(binder) > 2 and binder[-2].isdigit():
        binder = " ".join(model_name.split("_")[:-2])
        assert " " in binder
