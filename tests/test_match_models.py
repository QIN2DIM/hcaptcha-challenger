# -*- coding: utf-8 -*-
# Time       : 2023/10/21 17:56
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import json
from pathlib import Path
from typing import Tuple

import pytest
from loguru import logger

from hcaptcha_challenger import install
from hcaptcha_challenger import split_prompt_message, label_cleaning, ModelHub

install(upgrade=True)

modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()


def get_prompts():
    jp = Path(__file__).parent.joinpath("prompts.json")
    if not jp.exists():
        return []
    return json.loads(jp.read_bytes())


@pytest.mark.parametrize(
    "prompt2m", [("Please click on the head of the animal", "head_of_the_animal_2310_yolov8s.onnx")]
)
def test_lookup_model_by_ash(prompt2m: Tuple[str, str]):
    prompt, target = prompt2m
    _label = split_prompt_message(label_cleaning(prompt), "en")
    ash = f"{_label} default"

    pending = []
    for focus_name, classes in modelhub.lookup_ash_of_war(ash):
        pending.append(focus_name)

    assert target in pending


@pytest.mark.parametrize("prompt", get_prompts())
def test_nested(prompt: str):
    _label = split_prompt_message(label_cleaning(prompt), "en")

    if nested_models := modelhub.nested_categories.get(_label, []):
        for model_name in nested_models:
            assert model_name in modelhub.ashes_of_war
            element = model_name, modelhub.ashes_of_war.get(model_name, [])
            assert element
