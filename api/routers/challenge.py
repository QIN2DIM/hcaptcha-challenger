# -*- coding: utf-8 -*-
# Time       : 2024/4/14 12:25
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:

from fastapi import APIRouter
from loguru import logger

import hcaptcha_challenger as solver
from hcaptcha_challenger import ModelHub, register_pipline
from hcaptcha_challenger.models import SelfSupervisedResponse, SelfSupervisedPayload
from hcaptcha_challenger.tools.zero_shot_image_classifier import invoke_clip_tool

router = APIRouter()

# Init local-side of the ModelHub
solver.install(upgrade=True, clip=True)

modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()

clip_model = register_pipline(modelhub, fmt="onnx")
logger.success(
    "register clip_model", tool=clip_model.__class__.__name__, modelhub=modelhub.__class__.__name__
)


@router.post("/image_label_binary", response_model=SelfSupervisedResponse)
async def challenge_image_label_binary(payload: SelfSupervisedPayload):
    results = invoke_clip_tool(modelhub, payload, clip_model)
    return SelfSupervisedResponse(results=results)
