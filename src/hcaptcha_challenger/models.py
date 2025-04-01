# -*- coding: utf-8 -*-
# Time       : 2023/11/16 0:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import json
from enum import Enum
from typing import Literal, List

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


class CaptchaResponse(BaseModel):
    class Token(BaseModel):
        req: str
        type: str = "hsw"

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


SCOTModelType = Literal["gemini-2.5-pro-exp-03-25", "gemini-2.0-flash-thinking-exp-01-21"]

FastShotModelType = Literal["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp-01-21"]


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
