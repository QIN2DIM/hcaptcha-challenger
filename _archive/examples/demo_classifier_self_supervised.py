# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path
from typing import List

import hcaptcha_challenger as solver
from hcaptcha_challenger import handle, ModelHub, DataLake, register_pipline

# Init local-side of the ModelHub
solver.install(upgrade=True, clip=True)

assets_dir = Path(__file__).parent.parent.joinpath("assets")
images_dir = assets_dir.joinpath("image_label_binary/off_road_vehicle")

prompt = "Please click each image containing a sedan car"

# Patch datalake maps not updated in time
datalake_post = {
    # => prompt: sedan car
    handle(prompt): {
        "positive_labels": ["sedan car"],
        "negative_labels": ["bicycle", "off-road vehicle"],
    }
}


def prelude_self_supervised_config():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()
    for prompt_, serialized_binary in datalake_post.items():
        modelhub.datalake[prompt_] = DataLake.from_serialized(serialized_binary)
    clip_model = register_pipline(modelhub, fmt="onnx")

    return modelhub, clip_model


def get_test_images() -> List[Path]:
    images = []
    for image_name in os.listdir(images_dir):
        image_path = images_dir.joinpath(image_name)
        if image_path.is_file():
            images.append(image_path)

    return images


def demo():
    def output_markdown_preview():
        """# pip install pandas tabulate"""
        try:
            import pandas as pd
            import tabulate
        except ImportError:
            for image_path, result in zip(image_paths, results):
                print(image_path, f"{result=}")
        else:
            output = [
                {"image": f"![]({image_path})", "result": result}
                for image_path, result in zip(image_paths, results)
            ]
            mdk = pd.DataFrame.from_records(output).to_markdown()
            mdk = f"- prompt: `{prompt}`\n\n{mdk}"
            Path("result.md").write_text(mdk, encoding="utf8")
            print(mdk)

    modelhub, clip_model = prelude_self_supervised_config()
    image_paths = get_test_images()

    classifier = solver.BinaryClassifier(modelhub=modelhub, clip_model=clip_model)
    if results := classifier.execute(prompt, image_paths, self_supervised=True):
        output_markdown_preview()


if __name__ == "__main__":
    demo()
