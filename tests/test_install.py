# -*- coding: utf-8 -*-
# Time       : 2023/10/5 17:36
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import typing

import httpx
import pytest

import hcaptcha_challenger as solver
from hcaptcha_challenger import ModelHub


@pytest.mark.parametrize(
    ["upgrade", "flush_yolo"],
    [
        [False, "e"],
        [False, "0"],
        [False, "1"],
        [False, 0],
        [False, 1],
        [False, -0.1],
        [False, False],
        [False, True],
        [False, "head_of_the_animal_bear_2309_yolov8n"],
        [False, "head_of_the_animal_bear_2309_yolov8n.onnx"],
        [False, ("head_of_the_animal_bear_2309_yolov8n.onnx",)],
        [False, ["head_of_the_animal_bear_2309_yolov8n.onnx"]],
        # then create objects.yaml
        [True, True],
        [True, False],
        [True, "head_of_the_animal_bear_2309_yolov8n"],
        [True, "head_of_the_animal_bear_2309_yv8n"],
        [True, "head_onimal_bear_2309_yv8n"],
        [True, "head_onimal_b9_yv8n"],
        [True, "head_of_the_animal_bear_2309_yolov8n.onnx"],
        [True, ["head_of_the_animal_bear_2309_yolov8n.onnx"]],
        [True, ["can_be_eaten_2312_yolov8s.onnx"]],
    ],
)
def test_install(upgrade, flush_yolo):
    try:
        pending_models = solver.install(upgrade=upgrade, flush_yolo=flush_yolo)
    # Too many requests for a single IP
    except httpx.ReadTimeout:
        return

    modelhub = ModelHub()
    modelhub.parse_objects()
    files = os.listdir(modelhub.models_dir)

    if isinstance(flush_yolo, bool):
        if flush_yolo is False:
            assert pending_models is None
        elif flush_yolo is True:
            assert isinstance(pending_models, list)
            assert len(pending_models) == 1
            assert pending_models[0] == modelhub.circle_segment_model
            if upgrade:
                assert modelhub.models_dir.joinpath(pending_models[0]).exists()
    elif isinstance(flush_yolo, typing.Iterable):
        for fy in flush_yolo:
            if fy in modelhub.ashes_of_war:
                assert fy in pending_models
                if upgrade:
                    assert modelhub.models_dir.joinpath(fy).exists()
            else:
                if isinstance(pending_models, typing.Iterable):
                    assert fy not in pending_models
                    assert fy not in files
                else:
                    assert pending_models is None
    else:
        assert pending_models is None

    assert modelhub.objects_path.exists()
