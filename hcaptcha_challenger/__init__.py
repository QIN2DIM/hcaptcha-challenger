# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
import shutil
import time
import typing
from urllib.parse import urlparse

from ._scaffold import init_log, Config, get_challenge_ctx
from ._solutions.kernel import ModelHub
from ._solutions.kernel import PluggableObjects
from ._solutions.yolo import YOLO, Prefix
from .core import HolyChallenger

__all__ = ["HolyChallenger", "new_challenger", "get_challenge_ctx"]
__version__ = "0.4.2.28"

logger = init_log(
    error=os.path.join("datas", "logs", "error.log"),
    runtime=os.path.join("datas", "logs", "runtime.log"),
)


def install(
    onnx_prefix: typing.Optional[str] = Prefix.YOLOv6n, upgrade: typing.Optional[bool] = False
):
    dir_assets = os.path.join("datas", "models", "_assets")
    dir_models = os.path.join("datas", "models")
    path_objects_yaml = os.path.join("datas", "objects.yaml")

    os.makedirs(dir_assets, exist_ok=True)

    if not hasattr(Prefix, onnx_prefix):
        onnx_prefix = Prefix.YOLOv6n

    if upgrade is True:
        logger.debug(f"Reloading the local cache of Assets {dir_assets}")
        shutil.rmtree(dir_assets, ignore_errors=True)

    if (
        upgrade is True
        or not os.path.exists(path_objects_yaml)
        or not os.path.getsize(path_objects_yaml)
        or time.time() - os.path.getmtime(path_objects_yaml) > 3600
    ):
        PluggableObjects(path_objects=path_objects_yaml).sync()
    YOLO(dir_model=dir_models, onnx_prefix=onnx_prefix).pull_model().offload()


def new_challenger(
    dir_workspace: str = "_challenge",
    onnx_prefix: typing.Optional[str] = Prefix.YOLOv6n,
    lang: typing.Optional[str] = "en",
    screenshot: typing.Optional[bool] = False,
    debug: typing.Optional[bool] = False,
) -> HolyChallenger:
    """

    :param dir_workspace:
    :param onnx_prefix:
    :param lang:
    :param screenshot:
    :param debug:
    :return:
    """
    if not isinstance(dir_workspace, str) or not os.path.isdir(dir_workspace):
        dir_workspace = os.path.join("datas", "temp_cache", "_challenge")
        os.makedirs(dir_workspace, exist_ok=True)
    if not hasattr(Prefix, onnx_prefix):
        onnx_prefix = Prefix.YOLOv6n

    return HolyChallenger(
        dir_workspace=dir_workspace,
        dir_model=None,
        path_objects_yaml=None,
        lang=lang,
        onnx_prefix=onnx_prefix,
        screenshot=screenshot,
        debug=debug,
    )


def set_reverse_proxy(https_cdn: str):
    parser = urlparse(https_cdn)
    if parser.netloc and parser.scheme.startswith("https"):
        ModelHub.CDN_PREFIX = https_cdn
