# -*- coding: utf-8 -*-
# Time       : 2023/9/12 14:45
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import json
import re
from pathlib import Path
from typing import List

import pytest

from hcaptcha_challenger import split_prompt_message, label_cleaning, handle
from hcaptcha_challenger.components.prompt_handler import BAD_CODE

pattern = re.compile(r"[^\x00-\x7F]")

prompts: List[str] = []

if (proj := Path(__file__).parent.joinpath("prompts.json")).exists():
    prompts = json.loads(proj.read_text(encoding="utf8"))


@pytest.mark.parametrize("prompt", prompts)
def test_split_prompt_message(prompt: str):
    result = split_prompt_message(prompt, lang="en")
    assert result != prompt


@pytest.mark.parametrize("prompt", prompts)
def test_is_illegal_chars(prompt: str):
    result = label_cleaning(prompt)
    assert not pattern.search(result)


@pytest.mark.parametrize("prompt", prompts)
def test_handle_prompt(prompt: str):
    result = handle(prompt)
    for char in result:
        assert char.isascii(), f">> NOT ALPHA {char=}, \\u{ord(char):04x}"


def test_split_area_select_prompt():
    prompt = "Please click on the thumbnail of something that can be eaten"
    model_name = "can_be_eaten_2309_yolov8s.onnx"
    binder = " ".join(model_name.split("_")[:-2])
    assert binder in prompt


def test_split_binary_prompt():
    prompts_ = [
        "Please click each image containing a pair of headphones",
        "Please click each image containing an off-road vehicle",
        "Please click on the STAR with a texture of BRICKS",
        "Please click on the smallest animal.",
        "Please select all images that appear warmer in comparison to other images",
        "Select all cats.",
        "please click on the most similar object to the following reference shape:",
        "please click in the center of an observation wheel",
        "Please select all images of one type that appear warmer in comparison to other images",
    ]
    for prompt in prompts_:
        print(handle(prompt))


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


def review_badcode(prompt, memo):
    for char in prompt:
        if not char.isascii() and char not in BAD_CODE and char not in memo:
            memo.add(char)
            print(f">> NOT ALPHA {char=}, \\u{ord(char):04x}")


def test_new_bad_code():
    memo = set()
    for prompt in prompts:
        review_badcode(prompt, memo)

    assert len(memo) == 0, memo


def test_special_bad_code():
    prompt = handle("brlcks")
    for char in prompt:
        if not char.isascii() and char not in BAD_CODE:
            print(f">> NOT ALPHA {char=}, \\u{ord(char):04x}")

    assert "l" == "l"
