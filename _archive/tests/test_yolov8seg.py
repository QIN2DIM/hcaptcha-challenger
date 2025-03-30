from typing import Tuple

import pytest

from hcaptcha_challenger import handle, ModelHub

modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()


def test_yolov8seg_ash():
    ash = "please click on the object that appears only once"
    for focus_name, classes in modelhub.lookup_ash_of_war(ash):
        assert "appears_only_once" in focus_name
        assert "_yolov8" in focus_name
        assert "-seg" in focus_name
        assert ".onnx" in focus_name


@pytest.mark.parametrize(
    "prompt2model",
    [
        (
            "Please click on the STAR with a texture of BRICKS",
            "star_with_a_texture_of_bricks_2309_yolov8s-seg.onnx",
        ),
        (
            "please click on the object that appears only once",
            "appears_only_once_2309_yolov8s-seg.onnx",
        ),
    ],
)
def test_yolov8seg_match(prompt2model: Tuple[str, str]):
    prompt, model_name = prompt2model
    prompt = handle(prompt)
    p = [f for f, _ in modelhub.lookup_ash_of_war(prompt)]
    assert model_name in p
