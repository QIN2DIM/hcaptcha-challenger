# -*- coding: utf-8 -*-
# Time       : 2023/10/24 5:39
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from hcaptcha_challenger import (
    DataLake,
    install,
    ModelHub,
    ZeroShotImageClassifier,
    register_pipline,
)

install(upgrade=True)

assets_dir = Path(__file__).parent.parent.joinpath("assets")
images_dir = assets_dir.joinpath("image_label_binary/off_road_vehicle")


def auto_labeling():
    """
    Example:
    ---

    1. Roughly observe the distribution of the dataset and design a DataLake for the challenge prompt.
        - ChallengePrompt: "Please click each image containing an off-road vehicle"
        - positive_labels --> ["off-road vehicle"]
        - negative_labels --> ["bicycle", "car"]

    2. You can design them in batches and save them as YAML files,
    which the classifier can read and automatically DataLake

    3. Note that positive_labels is a list, and you can specify multiple labels for this variable
    if the label pointed to by the prompt contains ambiguity。

    :return:
    """
    # Refresh experiment environment
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    yes_dir = images_dir.joinpath("yes")
    bad_dir = images_dir.joinpath("bad")
    for cd in [yes_dir, bad_dir]:
        shutil.rmtree(cd, ignore_errors=True)
        cd.mkdir(parents=True, exist_ok=True)

    # !! IMPORT !!
    # Prompt: "Please click each image containing an off-road vehicle"
    data_lake = DataLake.from_binary_labels(
        positive_labels=["off-road vehicle"], negative_labels=["bicycle", "car"]
    )

    # Parse DataLake and build the model pipline
    tool = ZeroShotImageClassifier.from_datalake(data_lake)
    model = register_pipline(modelhub)

    total = len(os.listdir(images_dir))
    with tqdm(total=total, desc=f"Labeling | {images_dir.name}") as progress:
        for image_name in os.listdir(images_dir):
            image_path = images_dir.joinpath(image_name)
            if not image_path.is_file():
                progress.total -= 1
                continue

            # The label at position 0 is the highest scoring target
            image = Image.open(image_path)
            results = tool(model, image)

            # we're only dealing with binary classification tasks here
            if results[0]["label"] in tool.positive_labels:
                output_path = yes_dir.joinpath(image_name)
            else:
                output_path = bad_dir.joinpath(image_name)
            shutil.copyfile(image_path, output_path)

            progress.update(1)

    if "win32" in sys.platform:
        os.startfile(images_dir)


if __name__ == "__main__":
    auto_labeling()
