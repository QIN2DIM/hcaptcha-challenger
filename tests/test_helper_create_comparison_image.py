from pathlib import Path

from hcaptcha_challenger.helper.create_comparison_image import create_comparison_image

dataset_dir = Path(__file__).parent.joinpath("funcaptcha/compare_entity_orientation")


def test_create_comparison_image():
    for input_path in dataset_dir.glob("*"):
        if "cmp" in input_path.name:
            continue

        array_filename = f"{input_path.stem}_cmp.png"
        array_path = input_path.parent.joinpath(array_filename)
        array_image, _ = create_comparison_image(input_path, 135)
        print(f"Array plot saved to: {array_path}")
