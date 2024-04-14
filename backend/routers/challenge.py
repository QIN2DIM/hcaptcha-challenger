# -*- coding: utf-8 -*-
# Time       : 2024/4/14 12:25
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from fastapi import APIRouter


from typing import List

from pydantic import BaseModel, Field, Base64Str

router = APIRouter()


class SelfSupervisedPayload(BaseModel):
    """hCaptcha payload of the image_label_binary challenge"""

    prompt: str = Field(..., description="challenge prompt")
    challenge_images: List[Base64Str] = Field(default_factory=list)
    positive_labels: List[str] | None = Field(default_factory=list)
    negative_labels: List[str] | None = Field(default_factory=list)


class SelfSupervisedResponse(BaseModel):
    """The binary classification result of the image, in the same order as the challenge_images."""

    results: List[bool] = Field(default_factory=list)


import os
from pathlib import Path
from typing import List

import hcaptcha_challenger as solver
from hcaptcha_challenger import handle, ModelHub, DataLake, register_pipline
import pandas as pd

# Init local-side of the ModelHub
solver.install(upgrade=True, clip=True)

images_dir = Path("tmp_dir/image_label_binary/streetlamp")

prompt = "streetlamp"

# Patch datalake maps not updated in time
datalake_post = {
    # => prompt: sedan car
    handle(prompt): {"positive_labels": ["streetlamp"], "negative_labels": ["duck", "shark"]}
}


def prelude_self_supervised_config():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()
    for prompt_, serialized_binary in datalake_post.items():
        modelhub.datalake[prompt_] = DataLake.from_serialized(serialized_binary)
    clip_model = register_pipline(modelhub, fmt="onnx")

    return modelhub, clip_model


def demo(image_paths: List[Path]):
    modelhub, clip_model = prelude_self_supervised_config()

    classifier = solver.BinaryClassifier(modelhub=modelhub, clip_model=clip_model)
    if results := classifier.execute(prompt, image_paths, self_supervised=True):
        output = [
            {"image": f"![]({image_path})", "result": result}
            for image_path, result in zip(image_paths, results)
        ]
        mdk = pd.DataFrame.from_records(output).to_markdown()
        Path(f"result_{prompt}.md").write_text(mdk, encoding="utf8")


@router.post("/image_label_binary/clip", response_model=SelfSupervisedResponse)
async def read_item(payload: SelfSupervisedPayload):
    return SelfSupervisedResponse()
