# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from hcaptcha_challenger.components.image_classifier import Classifier as BinaryClassifier
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.utils import init_log

__all__ = ["new_challenger", "BinaryClassifier"]
__version__ = "0.6.0"


@dataclass
class Project:
    at_dir = Path(__file__).parent
    logs = at_dir.joinpath("logs")


project = Project()

init_log(
    runtime=project.logs.joinpath("runtime.log"),
    error=project.logs.joinpath("error.log"),
    serialize=project.logs.joinpath("serialize.log"),
)


def install(upgrade: bool | None = False, username: str = "QIN2DIM", lang: str = "en"):
    modelhub = ModelHub.from_github_repo(username=username, lang=lang)
    modelhub.pull_objects(upgrade=upgrade)
    modelhub.assets.flush_runtime_assets(upgrade=upgrade)


def new_challenger(
    lang: str | None = "en",
    screenshot: bool | None = False,
    debug: bool | None = False,
    slowdown: bool | None = True,
    *args,
    **kwargs,
):
    from hcaptcha_challenger.core import HolyChallenger

    """Soon to be deprecated"""
    return HolyChallenger(
        dir_workspace=project.challenge_cache_dir,
        models_dir=project.models_dir,
        objects_path=project.objects_path,
        lang=lang,
        screenshot=screenshot,
        debug=debug,
        slowdown=slowdown,
    )


def set_reverse_proxy(https_cdn: str):
    parser = urlparse(https_cdn)
    if parser.netloc and parser.scheme.startswith("https"):
        ModelHub.CDN_PREFIX = https_cdn
