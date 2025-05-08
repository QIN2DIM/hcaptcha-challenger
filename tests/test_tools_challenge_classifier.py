import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import dotenv
import pytest

from hcaptcha_challenger import FastShotModelType, ChallengeClassifier, ChallengeTypeEnum

# Load environment variables
dotenv.load_dotenv()

# Test configuration
TEST_MODEL: FastShotModelType = "gemini-2.0-flash"

CHALLENGE_CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "dir_name": "image_drag_drop",
        "expected_map": {
            "single": ChallengeTypeEnum.IMAGE_DRAG_SINGLE,
            "multi": ChallengeTypeEnum.IMAGE_DRAG_MULTI,
        },
    },
    {
        "dir_name": "image_label_area_select",
        "expected_map": {
            "single": ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT,
            "multi": ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT,
        },
    },
]

MAX_IMAGES_PER_TYPE = 2


def generate_individual_test_cases():
    """Generates test cases for each image file based on CHALLENGE_CONFIGURATIONS,
    limiting to MAX_IMAGES_PER_TYPE images per ChallengeTypeEnum per directory."""
    test_cases = []
    base_data_path = Path(__file__).parent / "challenge_view"

    for config in CHALLENGE_CONFIGURATIONS:
        dir_name = config["dir_name"]
        expected_map = config["expected_map"]
        dataset_dir = base_data_path / dir_name

        # Counter for images per type within the current directory/config
        type_counts = defaultdict(int)

        if not dataset_dir.is_dir():
            print(f"Warning: Test data directory does not exist: {dataset_dir}, skipping.")
            continue

        images = [
            img for img in dataset_dir.rglob("*.png") if not img.name.startswith("coordinate_grid")
        ]

        if not images:
            print(f"Warning: No images found in {dataset_dir}, skipping for this directory.")
            continue

        for image_file in images:
            determined_expected_type = None
            image_name_lower = image_file.name.lower()
            for keyword, enum_val in expected_map.items():
                if keyword.lower() in image_name_lower:
                    determined_expected_type = enum_val
                    break

            if determined_expected_type:
                if type_counts[determined_expected_type] < MAX_IMAGES_PER_TYPE:
                    test_cases.append(
                        pytest.param(
                            image_file, determined_expected_type, id=f"{dir_name}-{image_file.name}"
                        )
                    )
                    type_counts[determined_expected_type] += 1
                # else:
                #     print(f"Skipping {image_file.name} for {dir_name} as limit for {determined_expected_type.name} reached.")
            else:
                print(
                    f"Warning: Could not determine expected type for {image_file.name} in {dir_name}. "
                    f"Skipping this file. Ensure filename contains one of: {list(expected_map.keys())}"
                )
    if not test_cases:
        print("Warning: No test cases were generated. Check configurations and data paths.")
    return test_cases


class TestChallengeClassifier:
    """Challenge classifier test class"""

    @pytest.fixture(scope="class")
    def classifier(self):
        """Create challenge classifier instance"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY environment variable not set")
        return ChallengeClassifier(gemini_api_key=api_key)

    @pytest.mark.parametrize("image_file, expected_type_enum", generate_individual_test_cases())
    def test_challenge_classifier(
        self,
        classifier: ChallengeClassifier,
        image_file: Path,
        expected_type_enum: ChallengeTypeEnum,
    ):
        """Test challenge classification for a single image file."""

        actual_challenge_type = classifier.invoke(image_file, model=TEST_MODEL)

        assert isinstance(actual_challenge_type, ChallengeTypeEnum), (
            f"Classifier for '{image_file.name}' returned type "
            f"{type(actual_challenge_type).__name__} instead of ChallengeTypeEnum. "
            f"Value: {actual_challenge_type}"
        )

        assert actual_challenge_type == expected_type_enum, (
            f"Failed classification for '{image_file.name}': "
            f"Expected {expected_type_enum.name}, "
            f"got {actual_challenge_type.name if isinstance(actual_challenge_type, ChallengeTypeEnum) else actual_challenge_type}"
        )
