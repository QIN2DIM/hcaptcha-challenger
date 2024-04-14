# -*- coding: utf-8 -*-
# Time       : 2024/4/14 12:25
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from io import BytesIO
from typing import List

from PIL import Image
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field, Base64Bytes

import hcaptcha_challenger as solver
from hcaptcha_challenger import (
    handle,
    ModelHub,
    DataLake,
    register_pipline,
    ZeroShotImageClassifier,
)

router = APIRouter()

# Init local-side of the ModelHub
solver.install(upgrade=True, clip=True)

modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()

clip_model = register_pipline(modelhub, fmt="onnx")
logger.success(
    "register clip_model", tool=clip_model.__class__.__name__, modelhub=modelhub.__class__.__name__
)


class SelfSupervisedPayload(BaseModel):
    """hCaptcha payload of the image_label_binary challenge"""

    prompt: str = Field(..., description="challenge prompt")
    challenge_images: List[Base64Bytes] = Field(default_factory=list)
    positive_labels: List[str] | None = Field(default_factory=list)
    negative_labels: List[str] | None = Field(default_factory=list)


class SelfSupervisedResponse(BaseModel):
    """The binary classification result of the image, in the same order as the challenge_images."""

    results: List[bool] = Field(default_factory=list)


def invoke_clip_tool(payload: SelfSupervisedPayload) -> List[bool]:
    label = handle(payload.prompt)

    if any(payload.positive_labels) and any(payload.negative_labels):
        serialized = {
            "positive_labels": payload.positive_labels,
            "negative_labels": payload.negative_labels,
        }
        modelhub.datalake[label] = DataLake.from_serialized(serialized)

    if not (dl := modelhub.datalake.get(label)):
        dl = DataLake.from_challenge_prompt(label)
    tool = ZeroShotImageClassifier.from_datalake(dl)

    # Default to `RESNET.OPENAI` perf_counter 1.794s
    model = clip_model or register_pipline(modelhub)

    response: List[bool] = []
    for image_data in payload.challenge_images:
        results = tool(model, image=Image.open(BytesIO(image_data)))
        trusted = results[0]["label"] in tool.positive_labels
        response.append(trusted)

    return response


@router.post("/image_label_binary", response_model=SelfSupervisedResponse)
async def challenge_image_label_binary(payload: SelfSupervisedPayload):
    results = invoke_clip_tool(payload)
    return SelfSupervisedResponse(results=results)
