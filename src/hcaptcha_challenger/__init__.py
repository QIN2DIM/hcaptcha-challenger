# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from pathlib import Path

from hcaptcha_challenger.agent.challenger import AgentV, AgentConfig
from hcaptcha_challenger.models import RequestType
from hcaptcha_challenger.tools.challenge_classifier import ChallengeClassifier
from hcaptcha_challenger.tools.challenge_classifier import ChallengeTypeEnum
from hcaptcha_challenger.tools.image_classifier import ImageClassifier
from hcaptcha_challenger.tools.spatial_bbox_reasoning import SpatialBboxReasoner
from hcaptcha_challenger.tools.spatial_path_reasoning import SpatialPathReasoner
from hcaptcha_challenger.tools.spatial_point_reasoning import SpatialPointReasoner
from hcaptcha_challenger.utils import init_log

__all__ = [
    "ChallengeTypeEnum",
    "RequestType",
    "AgentV",
    "AgentConfig",
    "ImageClassifier",
    'ChallengeClassifier',
    'SpatialPathReasoner',
    'SpatialPointReasoner',
    'SpatialBboxReasoner',
]

LOG_DIR = Path(__file__).parent.joinpath("logs", "{time:YYYY-MM-DD}")

init_log(
    runtime=LOG_DIR.joinpath("runtime.log"),
    error=LOG_DIR.joinpath("error.log"),
    serialize=LOG_DIR.joinpath("serialize.log"),
)
