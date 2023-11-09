# -*- coding: utf-8 -*-
# Time       : 2023/11/9 22:10
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path
from typing import List

from PIL import Image

import hcaptcha_challenger as solver
from hcaptcha_challenger import ModelHub, register_pipline, DataLake, ZeroShotImageClassifier

# Init local-side of the ModelHub
solver.install(upgrade=True, clip=True)

assets_dir = Path(__file__).parent.parent.joinpath("assets")
images_dir = assets_dir.joinpath(
    "image_label_multiple_choice", "select the most accurate description of the image"
)


def prelude_self_supervised_config():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()
    clip_model = register_pipline(modelhub)

    return modelhub, clip_model


def load_samples() -> List[Path]:
    image_paths = []
    for image_name in os.listdir(images_dir):
        image_path = images_dir.joinpath(image_name)
        if image_path.is_file():
            image_paths.append(image_path)
    return image_paths


def demo():
    modelhub, clip_model = prelude_self_supervised_config()
    image_paths = load_samples()

    # Parse following fields from `requester_restricted_answer_set`
    # from: list(requester_restricted_answer_set.keys())
    candidates = ["Kitchen", "Living Room", "Bedroom"]

    dl = DataLake.from_binary_labels(positive_labels=candidates[:1], negative_labels=candidates[1:])
    tool = ZeroShotImageClassifier.from_datalake(dl)

    # `image_paths` are the already downloaded images of the challenge
    # That is, the picture downloaded from the endpoint link `tasklist.datapoint_uri`
    for image_path in image_paths:
        results = tool(clip_model, image=Image.open(image_path))
        trusted_prompt = results[0]["label"]
        for label in candidates:
            if DataLake.PREMISED_YES.format(label) == trusted_prompt:
                # That is, the most consistent description of the picture is the `trusted_label`
                # You can return this variable as needed, and then submit the form
                # or click on the btn page element to which it is bound.
                trusted_label = label
                print(f"{image_path.name=} {trusted_label=}")


if __name__ == "__main__":
    demo()
