# -*- coding: utf-8 -*-
# Time       : 2023/11/16 0:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field, field_validator

from hcaptcha_challenger.components.prompt_handler import label_cleaning


class Status(str, Enum):
    # <success> Challenge Passed
    CHALLENGE_SUCCESS = "success"
    # <retry> Your proxy IP may have been flagged
    CHALLENGE_RETRY = "retry"
    # <backcall> (New Challenge) Types of challenges not yet scheduled
    CHALLENGE_BACKCALL = "backcall"


class ImageTask(BaseModel):
    datapoint_uri: str
    task_key: str


class QuestionResp(BaseModel):
    c: Dict[str, str] = Field(default_factory=dict)
    """
    type: hsw
    req: eyj0 ...
    """

    challenge_uri: str = ""
    """
    https://hcaptcha.com/challenge/grid/challenge.js
    """

    key: str = ""
    """
    E0_eyj0 ...
    """

    request_config: Dict[str, Any] = Field(default_factory=dict)

    request_type: str = ""
    """
    1. image_label_binary
    2. image_label_area_select
    """

    requester_question: Dict[str, str] = Field(default_factory=dict)
    """
    image_label_binary      | { en: Please click on all images containing an animal }
    image_label_area_select | { en: Please click on the rac\u0441oon }
    """

    requester_question_example: List[str] = Field(default_factory=list)
    """
    [
        "https://imgs.hcaptcha.com/ + base64"
    ]
    """

    requester_restricted_answer_set: Dict[str, Any] = Field(default_factory=dict)
    """
    Not available on the binary challenge
    """

    tasklist: List[ImageTask] = Field(default_factory=list)
    """
    [
        {datapoint_uri: "https://imgs.hcaptcha.com + base64", task_key: "" },
        {datapoint_uri: "https://imgs.hcaptcha.com + base64", task_key: "" },
    ]
    """

    @field_validator("requester_question_example")
    def check_requester_question_example(cls, v: str | List[str]):
        # In the case of multichoice challenge
        if isinstance(v, str):
            v = [v]
        return v

    def cache(self, tmp_dir: Path):
        shape_type = self.request_config.get("shape_type", "")

        requester_question = label_cleaning(self.requester_question.get("en", ""))
        answer_keys = list(self.requester_restricted_answer_set.keys())
        ak = f".{answer_keys[0]}" if len(answer_keys) > 0 else ""
        fn = f"{self.request_type}.{shape_type}.{requester_question}{ak}.json"

        inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
        for c in inv:
            fn = fn.replace(c, "")

        if tmp_dir and tmp_dir.exists():
            fn = tmp_dir.joinpath(fn)

        Path(fn).write_text(self.model_dump_json(indent=2), encoding="utf8")


class ChallengeResp(BaseModel):
    c: Dict[str, str] = Field(default_factory=dict)
    """
    type: hsw
    req: eyj0 ...
    """

    is_pass: bool = Field(default=False, alias="pass")
    """
    true or false
    """

    generated_pass_UUID: str = ""
    """
    P1_eyj0 ...
    """

    error: str = ""
    """
    Only available if `pass is False`
    """


class RequestType(str, Enum):
    ImageLabelBinary = "image_label_binary"
    ImageLabelAreaSelect = "image_label_area_select"
    ImageLabelMultipleChoice = "image_label_multiple_choice"


class Answers(BaseModel):
    v: str = "fc6ae83"
    job_mode: str = ""
    answers: Dict[str, Any] = Field(default_factory=dict)
    serverdomain: str = ""
    sitekey: str = ""
    motionData: str = ""
    n: str = ""
    c: str = ""
