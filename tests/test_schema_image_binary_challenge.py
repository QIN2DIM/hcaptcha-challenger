# -*- coding: utf-8 -*-
import pytest

from hcaptcha_challenger.models import BoundingBoxCoordinate, ImageBinaryChallenge


class TestBoundingBoxCoordinate:
    @pytest.mark.parametrize(
        "input_coords, expected_coords",
        [([0, 0], [0, 0]), ([1, 1], [1, 1]), ([2, 2], [2, 2]), ([0, 2], [0, 2]), ([2, 0], [2, 0])],
    )
    def test_direct_mapping(self, input_coords, expected_coords):
        """Test coordinates already in the [0, 2] range."""
        bbc = BoundingBoxCoordinate(box_2d=input_coords)
        assert bbc.box_2d == expected_coords

    @pytest.mark.parametrize(
        "input_coords, expected_coords",
        [
            ([-1, 0], [0, 0]),
            ([0, -5], [0, 0]),
            ([-10, -20], [0, 0]),
            ([3, 1], [0, 1]),  # 3 is < 333, so maps to 0
            ([1, 4], [1, 0]),  # 4 is < 333, so maps to 0
            ([3, 3], [0, 0]),
            ([10, 20], [0, 0]),  # 10 < 333, 20 < 333
        ],
    )
    def test_small_out_of_range_mapping(self, input_coords, expected_coords):
        """Test coordinates slightly out of [0, 2] range or small positive integers."""
        bbc = BoundingBoxCoordinate(box_2d=input_coords)
        assert bbc.box_2d == expected_coords

    @pytest.mark.parametrize(
        "input_coords, expected_coords",
        [
            ([100, 200], [0, 0]),  # 100 < 333, 200 < 333
            ([332, 332], [0, 0]),  # 332 < 333
            ([333, 333], [1, 1]),  # 333 >= 333 and < 667
            ([400, 500], [1, 1]),  # 400 < 667, 500 < 667
            ([666, 666], [1, 1]),  # 666 < 667
            ([667, 667], [2, 2]),  # 667 >= 667
            ([700, 800], [2, 2]),  # 700 >= 667, 800 >= 667
            ([999, 1000], [2, 2]),  # 999 >= 667, 1000 >= 667
        ],
    )
    def test_percentage_like_mapping(self, input_coords, expected_coords):
        """Test coordinates in the larger "percentage-like" range."""
        bbc = BoundingBoxCoordinate(box_2d=input_coords)
        assert bbc.box_2d == expected_coords

    @pytest.mark.parametrize(
        "input_coords, expected_coords",
        [
            ([0, 400], [0, 1]),  # 0 is direct, 400 maps to 1
            ([700, 1], [2, 1]),  # 700 maps to 2, 1 is direct
            ([-5, 500], [0, 1]),  # -5 maps to 0, 500 maps to 1
            ([600, 3], [1, 0]),  # 600 maps to 1, 3 maps to 0
            ([300, 667], [0, 2]),  # 300 maps to 0, 667 maps to 2
            ([667, 300], [2, 0]),  # 667 maps to 2, 300 maps to 0
        ],
    )
    def test_mixed_mapping(self, input_coords, expected_coords):
        """Test coordinates with mixed direct and percentage-like values."""
        bbc = BoundingBoxCoordinate(box_2d=input_coords)
        assert bbc.box_2d == expected_coords


class TestImageBinaryChallenge:
    def test_instantiation_and_log_message(self):
        """Test basic instantiation and log message format."""
        coords_data = [[0, 0], [100, 400], [700, 2]]
        expected_normalized_coords_for_log = [[0, 0], [0, 1], [2, 2]]

        coordinates = [BoundingBoxCoordinate(box_2d=data) for data in coords_data]
        challenge = ImageBinaryChallenge(
            challenge_prompt="Select all images with cats", coordinates=coordinates
        )
        assert challenge.challenge_prompt == "Select all images with cats"
        assert len(challenge.coordinates) == 3

        # Verify normalized coordinates in the log message
        # Note: log_message itself stringifies the coordinates list directly
        # So we check the internal state of the BoundingBoxCoordinate objects
        actual_normalized_coords = [coord.box_2d for coord in challenge.coordinates]
        assert actual_normalized_coords == expected_normalized_coords_for_log

        log_msg = challenge.log_message
        assert '"Challenge Prompt": "Select all images with cats"' in log_msg
        # The log message will show the stringified list of lists
        # assert f'"Coordinates": "{expected_normalized_coords_for_log}"' in log_msg.replace(" ", "")

    @pytest.mark.parametrize(
        "coords_data, expected_matrix",
        [
            ([[0, 0]], [True, False, False, False, False, False, False, False, False]),
            (
                [[0, 0], [1, 1], [2, 2]],
                [True, False, False, False, True, False, False, False, True],
            ),
            (
                [[10, 20], [350, 450], [700, 800]],  # Raw: [[0,0], [1,1], [2,2]]
                [True, False, False, False, True, False, False, False, True],
            ),
            (
                [[-5, 1], [1, 600], [800, -10]],  # Raw: [[0,1], [1,1], [2,0]]
                [False, True, False, False, True, False, True, False, False],
            ),
            ([], [False, False, False, False, False, False, False, False, False]),
            (
                [[0, 0], [0, 0], [0, 1]],  # Test duplicates and multiple entries
                [True, True, False, False, False, False, False, False, False],
            ),
        ],
    )
    def test_convert_box_to_boolean_matrix(self, coords_data, expected_matrix):
        """Test the conversion to a boolean matrix with various coordinate inputs."""
        coordinates = [BoundingBoxCoordinate(box_2d=data) for data in coords_data]
        challenge = ImageBinaryChallenge(challenge_prompt="Test", coordinates=coordinates)
        assert challenge.convert_box_to_boolean_matrix() == expected_matrix

    def test_convert_box_to_boolean_matrix_empty_coordinates(self):
        """Test conversion with an empty list of coordinates."""
        challenge = ImageBinaryChallenge(challenge_prompt="Test", coordinates=[])
        expected_matrix = [False] * 9
        assert challenge.convert_box_to_boolean_matrix() == expected_matrix

    def test_log_message_empty_coordinates(self):
        """Test log_message with empty coordinates list."""
        challenge = ImageBinaryChallenge(challenge_prompt="Empty Test", coordinates=[])
        log_msg = challenge.log_message
        assert '"Challenge Prompt": "Empty Test"' in log_msg
        assert '"Coordinates": "[]"' in log_msg
