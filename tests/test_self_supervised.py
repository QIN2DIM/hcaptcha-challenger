import os
from pathlib import Path

import pytest

from hcaptcha_challenger import handle, BinaryClassifier, ModelHub, DataLake

images_dir = Path(__file__).parent.joinpath("largest_animal")
prompt = handle("Please click on the largest animal.")


def get_prelude_images(to_bytes: bool):
    unexpected_challenge_images = []
    for image_name in os.listdir(images_dir):
        image_path = images_dir.joinpath(image_name)
        if not image_path.is_file():
            continue
        if not to_bytes:
            unexpected_challenge_images.append(image_path)
        else:
            unexpected_challenge_images.append(image_path.read_bytes())

    return unexpected_challenge_images


def get_patched_modelhub():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    datalake_post = {
        prompt: {
            "positive_labels": ["dog"],
            "negative_labels": ["frog", "hedgehog", "squirrel", "hummingbird"],
        }
    }
    for prompt_, serialized_binary in datalake_post.items():
        modelhub.datalake[prompt_] = DataLake.from_serialized(serialized_binary)

    # Remove this prepared label and
    # simulate a new challenge that has never been encountered before.
    del modelhub.nested_categories[prompt]

    return modelhub


@pytest.mark.parametrize("modelhub", [get_patched_modelhub()])
@pytest.mark.parametrize(
    "image_paths", [get_prelude_images(to_bytes=True), get_prelude_images(to_bytes=False)]
)
@pytest.mark.parametrize("self_supervised", [True, False])
def test_self_supervised_image_classification(modelhub, image_paths, self_supervised):
    classifier = BinaryClassifier(modelhub=modelhub)
    if results := classifier.execute(prompt, image_paths, self_supervised=self_supervised):
        for image_path, result in zip(image_paths, results):
            # print(image_path.name, result, classifier.model_name)
            pass

    assert isinstance(results, list)

    if self_supervised is True:
        assert classifier.model_name in [
            modelhub.DEFAULT_CLIP_VISUAL_MODEL,
            modelhub.DEFAULT_CLIP_TEXTUAL_MODEL,
        ]
        if isinstance(image_paths[0], Path):
            assert len(results) == len(image_paths)
            for ri in results:
                assert ri is not None
        elif isinstance(image_paths[0], bytes):
            assert len(results) == len(image_paths)
            for ri in results:
                assert ri is None
    elif self_supervised is False:
        assert classifier.model_name == ""
        assert len(results) == 0
