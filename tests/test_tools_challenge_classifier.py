import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import dotenv
import pytest

from hcaptcha_challenger import FastShotModelType, ChallengeClassifier, ChallengeTypeEnum

# Load environment variables
dotenv.load_dotenv()

# Test configuration
TEST_MODELS: List[FastShotModelType] = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]


class TestChallengeClassifier:
    """Challenge classifier test class"""

    @pytest.fixture(scope="class")
    def classifier(self):
        """Create challenge classifier instance"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY environment variable not set")
        return ChallengeClassifier(gemini_api_key=api_key)

    @staticmethod
    def _run_classification_test(
        classifier: ChallengeClassifier,
        dataset_dir: Path,
        expected_types: Dict[str, ChallengeTypeEnum],
        model: FastShotModelType,
    ) -> Tuple[int, int, float]:
        """
        Run classification test and return statistics

        Args:
            classifier: Challenge classifier instance
            dataset_dir: Test dataset directory
            expected_types: Expected classification types {keyword: enum_type}
            model: Model to use

        Returns:
            total_samples, correct_classifications, average_processing_time
        """
        if not dataset_dir.is_dir():
            return 0, 0, 0.0

        # Collect all valid images
        images = [
            img for img in dataset_dir.rglob("*.png") if not img.name.startswith("coordinate_grid")
        ]

        if not images:
            return 0, 0, 0.0

        total_time = 0
        correct_cases = 0

        # Process images sequentially
        for image in images:
            start_time = time.perf_counter()
            challenge_type = classifier.invoke(image, model=model)
            end_time = time.perf_counter()
            process_time = end_time - start_time

            # Check if classification is correct
            if isinstance(challenge_type, ChallengeTypeEnum):
                for keyword, expected_type in expected_types.items():
                    if keyword in image.name and challenge_type == expected_type:
                        correct_cases += 1
                        break

            total_time += process_time

        return len(images), correct_cases, total_time / len(images) if images else 0

    @pytest.mark.parametrize(
        "challenge_type,expected_types",
        [
            (
                "image_drag_drop",
                {
                    "single": ChallengeTypeEnum.IMAGE_DRAG_SINGLE,
                    "multi": ChallengeTypeEnum.IMAGE_DRAG_MULTI,
                },
            ),
            (
                "image_label_area_select",
                {
                    "single": ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT,
                    "multi": ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT,
                },
            ),
        ],
    )
    def test_challenge_classifier(self, classifier, challenge_type, expected_types):
        """Test different types of challenge classification"""
        # Get test data directory
        dataset_dir = Path(__file__).parent.joinpath(f"challenge_view/{challenge_type}")
        if not dataset_dir.is_dir():
            pytest.skip(f"Test data directory does not exist: {dataset_dir}")

        # Record overall results
        overall_results = []

        # Test each model sequentially
        for model in TEST_MODELS:
            print(f"\n=== Model: {model} ===")

            # Run sequential test
            total, correct, avg_time = self._run_classification_test(
                classifier, dataset_dir, expected_types, model
            )

            print(f"\n--- {challenge_type} [{model}] ---")
            print(f"Total samples: {total}")
            print(f"Correct classifications: {correct}")
            print(f"Accuracy: {correct/total * 100:.1f}%" if total > 0 else "Accuracy: N/A")
            print(
                f"Average processing time: {avg_time:.2f}s/image"
                if total > 0
                else "Average processing time: N/A"
            )

            # Record model results
            overall_results.append(
                {
                    "model": model,
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total * 100 if total > 0 else 0,
                    "avg_time": avg_time,
                }
            )

        # Print model comparison results
        if overall_results:
            print(f"\n=== {challenge_type} Model Comparison ===")
            for result in overall_results:
                print(
                    f"Model: {result['model']}, Accuracy: {result['accuracy']:.1f}%, Average time: {result['avg_time']:.2f}s"
                )

            # Sort to find the best model
            best_model = max(overall_results, key=lambda x: (x["accuracy"], -x["avg_time"]))
            print(
                f"\nRecommended model: {best_model['model']} (Accuracy: {best_model['accuracy']:.1f}%, Average time: {best_model['avg_time']:.2f}s)"
            )

        # Assert only when there are test samples
        if overall_results and overall_results[0]["total"] > 0:
            assert any(
                result["correct"] > 0 for result in overall_results
            ), f"No samples passed for {challenge_type} test"
