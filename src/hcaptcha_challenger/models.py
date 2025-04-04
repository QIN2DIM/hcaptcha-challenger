# -*- coding: utf-8 -*-
# Time       : 2023/11/16 0:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import json
from enum import Enum
from typing import Literal, List, Dict, Any

from pydantic import BaseModel, Field


class ChallengeSignal(str, Enum):
    """
    Represents the possible statuses of a challenge.

    Enum Members:
      SUCCESS: The challenge was completed successfully.
      FAILURE: The challenge failed or encountered an error.
      START: The challenge has been initiated or started.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    START = "start"
    RETRY = "retry"
    QR_DATA_NOT_FOUND = "qr_data_not_found"
    EXECUTION_TIMEOUT = "challenge_execution_timeout"
    RESPONSE_TIMEOUT = "challenge_response_timeout"


class Token(BaseModel):
    req: str
    type: str = "hsw"


class CaptchaRequestConfig(BaseModel):
    version: int | None
    shape_type: str | None = None
    min_points: int | None = None
    max_points: int | None = None
    min_shapes_per_image: int | None = None
    max_shapes_per_image: int | None = None
    restrict_to_coords: Any | None = None
    minimum_selection_area_per_shape: int | None = None
    multiple_choice_max_choices: int | None = 1
    multiple_choice_min_choices: int | None = 1
    overlap_threshold: Any | None = None
    answer_type: str | None = None
    max_value: Any | None = None
    min_value: Any | None = None
    max_length: Any | None = None
    min_length: Any | None = None
    sig_figs: Any | None = None
    keep_answers_order: Any | None = None
    ignore_case: bool | None = None
    new_translation: bool | None = None


class CaptchaPayload(BaseModel):
    key: str = Field(default="")
    request_config: CaptchaRequestConfig | dict = Field(default_factory=dict)
    request_type: RequestType = Field(default="")
    requester_question: Dict[str, str] | None = Field(default_factory=dict)
    requester_restricted_answer_set: Dict[str, Any] | None = Field(default_factory=dict)
    requester_question_example: List[str] | str | None = Field(default=None)
    tasklist: List[Dict[str, Any]] = Field(default_factory=list)
    oby: str | None = Field(default=None)
    normalized: bool | None = Field(default=None)
    c: Token = Field(default_factory=dict)


class CaptchaResponse(BaseModel):

    c: Token
    """
    type: hsw
    req: eyj0 ...
    """

    is_pass: bool | None = Field(default=False, alias="pass")
    """
    true or false
    """

    expiration: int | None = None
    """
    Return only when the challenge passes. (Optional)
    """

    generated_pass_UUID: str | None = ""
    """
    Return only when the challenge passes. (Optional)
    P1_eyj0 ...
    """

    error: str | None = ""
    """
    Return only when the challenge failure. (Optional)
    """


class RequestType(str, Enum):
    """
    https://github.com/hCaptcha/hmt-basemodels/blob/71ee970ba38691139e484928999daa85920d4b0c/basemodels/constants.py
    """

    # General Intelligence
    HCI = "HCI"

    # -- Focus --
    IMAGE_LABEL_BINARY = "image_label_binary"
    IMAGE_LABEL_AREA_SELECT = "image_label_area_select"
    IMAGE_DRAG_DROP = "image_drag_drop"

    # -- Unknown --
    IMAGE_LABEL_MULTIPLE_CHOICE = "image_label_multiple_choice"
    TEXT_FREE_ENTRY = "text_free_entry"
    TEXT_LABEL_MULTIPLE_SPAN_SELECT = "text_label_multiple_span_select"
    TEXT_MULTIPLE_CHOICE_ONE_OPTION = "text_multiple_choice_one_option"
    TEXT_MULTIPLE_CHOICE_MULTIPLE_OPTIONS = "text_multiple_choice_multiple_options"
    IMAGE_LABEL_AREA_ADJUST = "image_label_area_adjust"
    IMAGE_LABEL_SINGLE_POLYGON = "image_label_single_polygon"
    IMAGE_LABEL_MULTIPLE_POLYGONS = "image_label_multiple_polygons"
    IMAGE_LABEL_SEMANTIC_SEGMENTATION_ONE_OPTION = "image_label_semantic_segmentation_one_option"
    IMAGE_LABEL_SEMANTIC_SEGMENTATION_MULTIPLE_OPTIONS = (
        "image_label_semantic_segmentation_multiple_options"
    )
    IMAGE_LABEL_TEXT = "image_label_text"
    MULTI_CHALLENGE = "multi_challenge"


# https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits
SCoTModelType = Literal[
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash-thinking-exp-01-21",
    # This model is not available in the free plan
    # Recommended for production environments for more tolerant rate limits
    "gemini-2.5-pro-preview-03-25",
]

FastShotModelType = Literal[
    "gemini-2.0-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-preview-03-25"
]


class BoundingBoxCoordinate(BaseModel):
    box_2d: List[int] = Field(
        description="It can only be in planar coordinate format, e.g. [0,2] for the 3rd element in the first row"
    )


class ImageBinaryChallenge(BaseModel):
    challenge_prompt: str
    coordinates: List[BoundingBoxCoordinate]

    def convert_box_to_boolean_matrix(self) -> List[bool]:
        """
        Converts the coordinate list to a one-dimensional Boolean matrix.

        Convert coordinates in a 3x3 matrix to a one-dimensional boolean list where:
        - [0,0] Corresponding index 0
        - [0,1] Corresponding index 1
        - ...
        - [2,2] Corresponding index 8

        Returns:
            List[bool]: Boolean list with length 9, coordinate position is True, other positions are False
        """
        # Initialize a boolean list of length 9, all False
        result = [False] * 9

        for coord in self.coordinates:
            row, col = coord.box_2d

            if 0 <= row < 3 and 0 <= col < 3:
                index = row * 3 + col
                result[index] = True

        return result

    @property
    def log_message(self) -> str:
        _coordinates = [i.box_2d for i in self.coordinates]
        bundle = {"Challenge Prompt": self.challenge_prompt, "Coordinates": str(_coordinates)}
        return json.dumps(bundle, indent=2, ensure_ascii=False)


class PointCoordinate(BaseModel):
    x: int
    y: int


class ImageAreaSelectChallenge(BaseModel):
    challenge_prompt: str
    points: List[PointCoordinate]

    @property
    def log_message(self) -> str:
        _coordinates = [{"x": i.x, "y": i.y} for i in self.points]
        bundle = {"Challenge Prompt": self.challenge_prompt, "Coordinates": str(_coordinates)}
        return json.dumps(bundle, indent=2, ensure_ascii=False)


class SpatialPath(BaseModel):
    start_point: PointCoordinate
    end_point: PointCoordinate


class ImageDragDropChallenge(BaseModel):
    challenge_prompt: str
    paths: List[SpatialPath]

    @property
    def log_message(self) -> str:
        _coordinates = [
            {
                "from": i.start_point.model_dump(mode='json'),
                "to": i.end_point.model_dump(mode='json'),
            }
            for i in self.paths
        ]
        bundle = {"Challenge Prompt": self.challenge_prompt, "Coordinates": str(_coordinates)}
        return json.dumps(bundle, indent=2, ensure_ascii=False)

    def get_approximate_paths(self, bbox) -> List[SpatialPath]:
        if len(self.paths) > 1:
            return self.paths

        path = self.paths[0]
        start_x, start_y = path.start_point.x, path.start_point.y
        if start_x > bbox["x"] + (bbox["width"] / 2) and start_y < bbox["y"] + (bbox["height"] / 2):
            path.start_point.x = int(bbox["x"] + (bbox["width"] * 0.875))
            path.start_point.y = int(bbox["y"] + (bbox["height"] * 0.393))
        return [path]
