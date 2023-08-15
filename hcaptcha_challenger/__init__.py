# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from loguru import logger

from hcaptcha_challenger.core import HolyChallenger
from ._scaffold import get_challenge_ctx, init_log
from ._solutions.kernel import ModelHub
from ._solutions.kernel import PluggableObjects
from ._solutions.yolo import Prefix, YOLO

__all__ = ["HolyChallenger", "new_challenger", "get_challenge_ctx"]
__version__ = "0.4.5"


@dataclass
class Project:
    at_dir = Path(__file__).parent
    datas_dir = at_dir.joinpath("datas")

    logs = datas_dir.joinpath("logs")
    models_dir = datas_dir.joinpath("models")
    assets_dir = models_dir.joinpath("_assets")
    objects_path = datas_dir.joinpath("objects.yaml")

    challenge_cache_dir = datas_dir.joinpath("temp_cache/_challenge")

    def __post_init__(self):
        for ck in [self.assets_dir, self.challenge_cache_dir]:
            ck.mkdir(777, parents=True, exist_ok=True)


project = Project()

init_log(
    runtime=project.logs.joinpath("runtime.log"),
    error=project.logs.joinpath("error.log"),
    serialize=project.logs.joinpath("serialize.log"),
)


def install(
    onnx_prefix: Literal["yolov6n", "yolov6s", "yolov6t"] = "yolov6n", upgrade: bool | None = False
):
    if not hasattr(Prefix, onnx_prefix):
        onnx_prefix = Prefix.YOLOv6n

    if upgrade is True:
        logger.debug(f"Reloading the local cache of Assets", assets_dir=str(project.assets_dir))
        shutil.rmtree(project.assets_dir, ignore_errors=True)

    if (
        upgrade is True
        or not project.objects_path.exists()
        or not project.objects_path.stat().st_size
        or time.time() - project.objects_path.stat().st_mtime > 3600
    ):
        PluggableObjects(path_objects=project.objects_path).sync()
    YOLO(models_dir=project.models_dir, onnx_prefix=onnx_prefix).pull_model().offload()


def new_challenger(
    lang: str | None = "en",
    screenshot: bool | None = False,
    debug: bool | None = False,
    slowdown: bool | None = True,
    *args,
    **kwargs,
) -> HolyChallenger:
    """

    :param slowdown:
    :param lang:
    :param screenshot:
    :param debug:
    :return:
    """
    return HolyChallenger(
        dir_workspace=project.challenge_cache_dir,
        models_dir=project.models_dir,
        objects_path=project.objects_path,
        lang=lang,
        onnx_prefix=Prefix.YOLOv6n,
        screenshot=screenshot,
        debug=debug,
        slowdown=slowdown,
    )


def set_reverse_proxy(https_cdn: str):
    parser = urlparse(https_cdn)
    if parser.netloc and parser.scheme.startswith("https"):
        ModelHub.CDN_PREFIX = https_cdn
