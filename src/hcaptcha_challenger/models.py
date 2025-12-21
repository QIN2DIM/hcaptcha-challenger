# -*- coding: utf-8 -*-
# Time       : 2023/11/16 0:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import json
import unicodedata
from enum import Enum
from typing import Literal, List, Dict, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# Known Unicode homoglyphs mapping (legacy, kept for reference)
BAD_CODE = {
    "а": "a",
    "е": "e",
    "e": "e",
    "i": "i",
    "і": "i",
    "ο": "o",
    "с": "c",
    "ԁ": "d",
    "ѕ": "s",
    "һ": "h",
    "у": "y",
    "р": "p",
    "ϳ": "j",
    "х": "x",
    "\u0405": "S",
    "\u0042": "B",
    "\u0052": "R",
    "\u0049": "I",
    "\u0043": "C",
    "\u004b": "K",
    "\u039a": "K",
    "\u0053": "S",
    "\u0421": "C",
    "\u006c": "l",
    "\u0399": "I",
    "\u0392": "B",
    "\u03a1": "P",  # Greek Rho -> Latin P
    "ー": "一",
    "土": "士",
}

INV = {"\\", "/", ":", "*", "?", "<", ">", "|", "\n"}


def normalize_unicode_text(text: str) -> str:
    """
    Normalize Unicode text to ASCII-safe string for file paths.

    This function applies a three-layer defense against Unicode homoglyphs:
    1. NFKC normalization - converts compatibility characters to canonical forms
    2. BAD_CODE mapping - replaces known homoglyphs with ASCII equivalents
    3. ASCII fallback - removes any remaining non-ASCII characters

    Args:
        text: The input text that may contain Unicode homoglyphs

    Returns:
        A normalized ASCII-safe string suitable for file paths
    """
    # Layer 1: NFKC normalization (handles many compatibility characters)
    # e.g., fullwidth letters, circled letters, superscripts, etc.
    result = unicodedata.normalize("NFKC", text)

    # Layer 2: Apply known homoglyph mappings
    for bad_char, good_char in BAD_CODE.items():
        result = result.replace(bad_char, good_char)

    # Layer 3: ASCII-only fallback for any remaining non-ASCII characters
    # This ensures the path will always be valid on all file systems
    result = "".join(c if ord(c) < 128 else "_" for c in result)

    return result


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


class CaptchaTaskEntity(BaseModel):
    entity_id: str | None = Field(default="")
    entity_uri: str | None = Field(default="")
    coords: List[int] | None = Field(default_factory=list)
    size: List[int] | None = Field(default_factory=list)
    metadata: dict | None = Field(default_factory=dict)


class CaptchaTask(BaseModel):
    datapoint_uri: str | None = Field(default="")
    task_key: str | None = Field(default="")
    entities: List[CaptchaTaskEntity] | None = Field(default_factory=list)


class CaptchaPayload(BaseModel):
    key: str = Field(default="")
    request_config: CaptchaRequestConfig | dict = Field(default_factory=dict)
    request_type: RequestType | None = Field(default=None)
    requester_question: Dict[str, str] | None = Field(default_factory=dict)
    requester_restricted_answer_set: Dict[str, Any] | None = Field(default_factory=dict)
    requester_question_example: List[str] | str | None = Field(default=None)
    tasklist: List[CaptchaTask] = Field(default_factory=list)
    oby: str | None = Field(default=None)
    normalized: bool | None = Field(default=None)
    c: Token = Field(default_factory=dict)

    def get_requester_question(self, language: str = "en") -> str:
        rq = self.requester_question.get(language, "unknown")
        return normalize_unicode_text(rq)


class CaptchaResponse(BaseModel):

    c: Token | None = Field(default_factory=dict)
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


class ChallengeTypeEnum(str, Enum):
    IMAGE_LABEL_SINGLE_SELECT = "image_label_single_select"
    IMAGE_LABEL_MULTI_SELECT = "image_label_multi_select"
    IMAGE_DRAG_SINGLE = "image_drag_single"
    IMAGE_DRAG_MULTI = "image_drag_multi"


# Type alias for skill rule job_type field - mirrors ChallengeTypeEnum values
JobTypeLiteral = Literal[
    "image_label_single_select", "image_label_multi_select", "image_drag_single", "image_drag_multi"
]


IGNORE_REQUEST_TYPE_LITERAL = Literal[
    "image_label_binary",
    "image_label_area_select",
    "image_drag_drop",
    "image_label_single_select",
    "image_label_multi_select",
    "image_drag_single",
    "image_drag_multi",
]

# https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits
SCoTModelType = Union[
    str,
    Literal[
        # This model is not available in the free plan.
        # Recommended for production environments for more tolerant rate limits.
        # [✨] https://ai.google.dev/gemini-api/docs/models?hl=zh-cn#gemini-3-pro
        "gemini-3-pro-preview",
        # https://ai.google.dev/gemini-api/docs/models?hl=zh-cn#gemini-3-flash
        "gemini-3-flash-preview",
        # [✨] https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro
        "gemini-2.5-pro",
        # [🤷‍♂️] https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash
        "gemini-2.5-flash",
    ],
]

DEFAULT_SCOT_MODEL: SCoTModelType = "gemini-2.5-pro"

FastShotModelType = Union[
    str,
    Literal[
        # [✨] https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash
        "gemini-2.5-flash",
        # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-lite
        "gemini-2.5-flash-lite",
    ],
]

DEFAULT_FAST_SHOT_MODEL: FastShotModelType = "gemini-2.5-flash"

THINKING_BUDGET_MODELS: List[Union[SCoTModelType, FastShotModelType]] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

THINKING_LEVEL_MODELS: List[str] = [
    "gemini-3-pro-preview",
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-3-flash-preview",
]


class ChallengeRouterResult(BaseModel):
    challenge_prompt: str
    challenge_type: ChallengeTypeEnum


class BoundingBoxCoordinate(BaseModel):
    box_2d: List[int] = Field(
        description="It can only be in planar coordinate format, e.g. [0,2] for the 3rd element in the first row",
        min_length=2,
        max_length=2,
    )

    def model_post_init(self, context: Any, /) -> None:
        val_for_x = self.box_2d[0]
        val_for_y = self.box_2d[1]

        # Determine the new x-coordinate
        if not (0 <= val_for_x <= 2):
            if val_for_x < 0:
                new_x = 0
            elif val_for_x < 333:
                new_x = 0
            elif val_for_x < 667:
                new_x = 1
            else:
                new_x = 2
        else:
            new_x = val_for_x

        # Determine the new y-coordinate
        if not (0 <= val_for_y <= 2):
            if val_for_y < 0:
                new_y = 0
            elif val_for_y < 333:
                new_y = 0
            elif val_for_y < 667:
                new_y = 1
            else:
                new_y = 2
        else:
            new_y = val_for_y

        self.box_2d = [new_x, new_y]


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


class SpatialBbox(BaseModel):
    top_left_x: int = Field(description="No more than 65% of width")
    top_left_y: int
    bottom_right_x: int = Field(description="No more than 65% of width")
    bottom_right_y: int


class ImageBboxChallenge(BaseModel):
    challenge_prompt: str
    bounding_boxes: SpatialBbox

    @property
    def log_message(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)


SPATIAL_PATH_STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "challenge_prompt": {"type": "string"},
        "paths": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_point": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                    "end_point": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                },
                "required": ["start_point", "end_point"],
            },
        },
    },
    "required": ["challenge_prompt", "paths"],
}

GameRuleMathType = Union[
    ChallengeTypeEnum,
    Literal[
        "image_label_single_select",
        "image_label_multi_select",
        "image_drag_single",
        "image_drag_multi",
    ],
]


class GameRule(BaseModel):
    rule: str
    name: str = Field(default="game-rule-default", description="Name of the rule")
    match_keys: List[str] | None = Field(
        default_factory=list,
        description="""
        Call the challenge by keyword matching, can also be set to full challenge_prompt.
        Only effective when `insert_mode=router`
        """,
    )
    challenge_type: GameRuleMathType | None = None
    insert_mode: Literal["router", "always"] = Field(
        default="router",
        description="""
    - router: Decide whether to insert the rule through the routing of `challenge_prompt + challenge_type`
    - always: Always as part of user_prompt
    """,
    )

    def model_post_init(self, context: Any, /) -> None:
        self.rule = f"\n{self.rule.strip()}"

        self.name = self.name or f"game-rule-{uuid4()}"

        if self.insert_mode == "router":
            if not self.challenge_type:
                raise ValueError("challenge_type is required when insert_mode is router")
            if not self.match_keys:
                raise ValueError("match_keys is required when insert_mode is router")


class GameRuleGroup(BaseModel):
    name: str = Field(default="custom")
    type: str = Field(default="select", description="Reserved fields, not used yet")
    rules: List[GameRule] = Field(default_factory=list)


class PluggableUserPrompt(BaseModel):
    rules: List[GameRule] = Field(default_factory=list)
    rule_groups: List[GameRule]


class CoordinateGrid(BaseModel):
    x_line_space_num: int | None = Field(
        default=15, description="Number of horizontal lines", ge=3, le=30
    )
    y_line_space_num: int | None = Field(
        default=20, description="Number of vertical lines", ge=3, le=30
    )
    color: str | None = Field(
        default="gray", description="The color of the auxiliary line, supports RGB and hexadecimal"
    )
    adaptive_contrast: bool | None = Field(
        default=False, description="Visual assist effects for enhancing high-contrast scenes"
    )
