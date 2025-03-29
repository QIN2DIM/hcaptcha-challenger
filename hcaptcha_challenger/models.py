# -*- coding: utf-8 -*-
# Time       : 2023/11/16 0:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import base64
import shutil
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Mapping
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, UUID4, AnyHttpUrl, Base64Bytes

from hcaptcha_challenger.constant import BAD_CODE, INV




class Status(str, Enum):
    # <success> Challenge Passed
    CHALLENGE_SUCCESS = "success"
    # <retry> Your proxy IP may have been flagged
    CHALLENGE_RETRY = "retry"
    # <backcall> (New Challenge) Types of challenges not yet scheduled
    CHALLENGE_BACKCALL = "backcall"
    # <timeout> Failed to pass the challenge within the specified time frame
    CHALLENGE_EXECUTION_TIMEOUT = "challenge_execution_timeout"
    CHALLENGE_RESPONSE_TIMEOUT = "challenge_response_timeout"


class Collectible(BaseModel):
    point: UUID4 | AnyHttpUrl = Field(..., description="sitelink or sitekey")

    @field_validator("point")
    def validate_point(cls, v: str):
        def is_valid_uuid4(string):
            try:
                uuid_obj = UUID(string)
            except ValueError:
                return False
            return uuid_obj.version == 4

        _sitekey = "c86d730b-300a-444c-a8c5-5312e7a93628"
        _sitelink = "https://accounts.hcaptcha.com/demo"

        if not isinstance(v, str):
            v = f"{_sitelink}?sitekey={_sitekey}"
        elif is_valid_uuid4(v):
            v = f"{_sitelink}?sitekey={v}"
        elif not v.startswith(_sitelink):
            v = f"{_sitelink}?sitekey={_sitekey}"

        return v

    @property
    def fixed_sitelink(self) -> str:
        return self.point


CollectibleType = Union[UUID4, AnyHttpUrl, str]


class ToolExecution(str, Enum):
    CHALLENGE = "challenge"
    COLLECT = "collect"


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

        # label cleaning
        requester_question = self.requester_question.get("en", "")
        for c in BAD_CODE:
            requester_question = requester_question.replace(c, BAD_CODE[c])

        answer_keys = list(self.requester_restricted_answer_set.keys())
        ak = f".{answer_keys[0]}" if len(answer_keys) > 0 else ""
        fn = f"{self.request_type}.{shape_type}.{requester_question}{ak}.json"

        for c in INV:
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


class ChallengeImage(BaseModel):
    datapoint_uri: str = Field(default="", description="图片的临时访问链接")

    filename: str = Field(default="challenge-image.jpeg", description="HASH 后的文件名，带有后缀")

    body: bytes = Field(default=b"", description="图片缓存字节")

    runtime_fp: Path = Field(
        default_factory=Path, description="图片的临时缓存路径，fp = typed_dir / filename"
    )

    def save(self, typed_dir: Path) -> Path:
        fp = typed_dir / self.filename
        fp.write_bytes(self.body)
        return fp

    def into_base64bytes(self) -> str:
        return base64.b64encode(self.body).decode()

    def move_to(self, dst: Path):
        if dst.is_dir():
            dst = dst / self.filename
        return shutil.move(self.runtime_fp, dst=dst)


class SelfSupervisedPayload(BaseModel):
    """hCaptcha payload of the image_label_binary challenge"""

    prompt: str = Field(..., description="challenge prompt")
    challenge_images: List[Base64Bytes] = Field(default_factory=list)
    positive_labels: List[str] | None = Field(default_factory=list, alias="positive")
    negative_labels: List[str] | None = Field(default_factory=list, alias="negative")


class SelfSupervisedResponse(BaseModel):
    """The binary classification result of the image, in the same order as the challenge_images."""

    results: List[bool] = Field(default_factory=list)
