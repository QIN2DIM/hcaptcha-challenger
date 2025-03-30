# -*- coding: utf-8 -*-
# Time       : 2023/9/28 14:04
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path
from typing import Dict

import pytest

import hcaptcha_challenger as solver
from hcaptcha_challenger import LocalBinaryClassifier

this_dir = Path(__file__).parent


def test_lbc():
    model_path = this_dir.joinpath("goose2309.onnx")
    image_dir = this_dir.joinpath("goose")
    lbc = LocalBinaryClassifier(model_path)
    for i, image_name in enumerate(os.listdir(image_dir)):
        image = image_dir.joinpath(image_name).read_bytes()
        result = lbc.parse_once(image)
        assert isinstance(result, bool)


@pytest.mark.parametrize(
    "model_path",
    ["goose2309.onnx", "goose.onnx", Path("goose239.onnx"), None, True, False, -1, 0.1, 0],
)
def test_lbc_model_path(model_path):
    with pytest.raises(FileNotFoundError):
        assert LocalBinaryClassifier(model_path)


@pytest.mark.parametrize("image", [b"goose"])
def test_lbc_image_type(image):
    model_path = this_dir.joinpath("goose2309.onnx")
    lbc = LocalBinaryClassifier(model_path)
    result = lbc.parse_once(image)
    assert result is None


@pytest.mark.parametrize(
    "prompt2images",
    [{"desert": this_dir.joinpath("desert")}, {"goose": this_dir.joinpath("goose")}],
)
def test_rbc(prompt2images: Dict[str, Path]):
    solver.install(upgrade=True)
    classifier = solver.BinaryClassifier()

    for prompt, image_dir in prompt2images.items():
        images = [image_dir.joinpath(image_name) for image_name in os.listdir(image_dir)]
        results = classifier.execute(prompt, images)
        assert results

        for i, result in enumerate(results):
            assert isinstance(result, bool)
            print(f"{images[i]} - {result}")
