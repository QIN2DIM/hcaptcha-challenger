# -*- coding: utf-8 -*-
# Time       : 2022/4/30 22:34
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import json
import os
import shutil
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx
from loguru import logger


class ChallengeStyle:
    WATERMARK = 144  # onTrigger 128x128
    GENERAL = 128
    GAN = 144


@dataclass
class GitHubUpStream:
    username: str
    GITHUB_RELEASE_API = ""
    URL_REMOTE_OBJECTS = ""

    def __post_init__(self):
        self.GITHUB_RELEASE_API = (
            f"https://api.github.com/repos/{self.username}/hcaptcha-challenger/releases"
        )
        self.URL_REMOTE_OBJECTS = f"https://raw.githubusercontent.com/{self.username}/hcaptcha-challenger/main/src/objects.yaml"


_hook = GitHubUpStream(username="QIN2DIM")


class Memory:
    _fn2memory = {}

    ASSET_TOKEN = "RA_kw"

    def __init__(self, fn: str, memory_dir: Path):
        self.fn = fn
        self.memory_dir = memory_dir

        self._build()

    def _build(self) -> typing.Optional[typing.Dict[str, str]]:
        """:return filename to nodeId"""
        if not self._fn2memory:
            os.makedirs(self.memory_dir, exist_ok=True)
            for memory_name in os.listdir(self.memory_dir):
                fn = memory_name.split(".")[0]
                fn = fn if fn.endswith(".onnx") else f"{fn}.onnx"
                node_id = memory_name.split(".")[-1]
                if node_id.startswith(self.ASSET_TOKEN):
                    self._fn2memory[fn] = node_id
        return self._fn2memory

    def get_node_id(self) -> typing.Optional[str]:
        return self._fn2memory.get(self.fn, "")

    def dump(self, new_node_id: str):
        old_node_id = self._fn2memory.get(self.fn)
        self._fn2memory[self.fn] = new_node_id

        if not old_node_id:
            memory_node = self.memory_dir.joinpath(f"{self.fn}.{new_node_id}")
            memory_node.write_text(str(memory_node))
        else:
            memory_src = self.memory_dir.joinpath(f"{self.fn}.{old_node_id}")
            memory_dst = self.memory_dir.joinpath(f"{self.fn}.{new_node_id}")
            shutil.move(memory_src, memory_dst)

    def is_outdated(self, remote_node_id: str) -> typing.Optional[bool]:
        """延迟反射的诊断步骤，保持分布式网络的两端一致性"""
        local_node_id = self.get_node_id()

        # Invalid judgment
        if (
            not local_node_id
            or not remote_node_id
            or not isinstance(remote_node_id, str)
            or not remote_node_id.startswith(self.ASSET_TOKEN)
        ):
            return

        if local_node_id != remote_node_id:
            return True
        return False


class Assets:
    GITHUB_RELEASE_API = _hook.GITHUB_RELEASE_API

    NAME_ASSETS = "assets"
    NAME_ASSET_NAME = "name"
    NAME_ASSET_SIZE = "size"
    NAME_ASSET_DOWNLOAD_URL = "browser_download_url"
    NAME_ASSET_NODE_ID = "node_id"

    _fn2assets = {}

    # Cache validity period: 2h
    CACHE_CONTROL = 7200

    def __init__(self, fn: str, assets_dir: Path):
        self.fn = fn
        self._assets_dir = assets_dir

        self._pull()

    def _preload(self):
        """仅在 Assets._pull 网络请求发起前调用，将有效的本地缓存代替远端资源"""
        if self._fn2assets:
            return

        os.makedirs(self._assets_dir, exist_ok=True)
        assets = [i for i in os.listdir(self._assets_dir) if i.isdigit()]
        if len(assets) >= 1:
            asset_name = assets[-1]
            if int(asset_name) + self.CACHE_CONTROL > int(time.time()):
                try:
                    record_node = self._assets_dir.joinpath(asset_name)
                    self._fn2assets = json.loads(record_node.read_text())
                except json.decoder.JSONDecodeError as err:
                    logger.warning(err)

    def _offload(self):
        """仅在 Assets._pull 网络请求发起后实现，用于更新本地缓存的内容及时间戳"""
        os.makedirs(self._assets_dir, exist_ok=True)
        for asset_fn in os.listdir(self._assets_dir):
            asset_src = self._assets_dir.joinpath(asset_fn)
            asset_dst = self._assets_dir.joinpath(f"_{asset_fn.replace('_', '')}")
            shutil.move(asset_src, asset_dst)
        record_node = self._assets_dir.joinpath(str(int(time.time())))
        record_node.write_text(json.dumps(self._fn2assets))

    def _pull(self, skip_preload: bool = False) -> typing.Optional[typing.Dict[str, dict]]:
        if not skip_preload:
            self._preload()
        if not self._fn2assets:
            self._request_assets()
        return self._fn2assets

    def _request_assets(self):
        logger.debug(f"Pulling AssetsObject", url=self.GITHUB_RELEASE_API)

        try:
            resp = httpx.get(self.GITHUB_RELEASE_API, timeout=3)
            data = resp.json()[0]
        except (httpx.ConnectError, json.decoder.JSONDecodeError) as err:
            logger.error(err)
        except (AttributeError, IndexError, KeyError) as err:
            logger.error(err)
        else:
            if isinstance(data, dict):
                assets: typing.List[dict] = data.get(self.NAME_ASSETS, [])
                for asset in assets:
                    self._fn2assets[asset[self.NAME_ASSET_NAME]] = asset
        finally:
            self._offload()

    def _get_asset(self, key: str, oncall_default: typing.Any):
        return self._fn2assets.get(self.fn, {}).get(key, oncall_default)

    def sync(self):
        self._fn2assets = {}
        self._request_assets()

    @property
    def assets_dir(self):
        return self._assets_dir

    def get_node_id(self) -> typing.Optional[str]:
        return self._get_asset(self.NAME_ASSET_NODE_ID, "")

    def get_download_url(self) -> typing.Optional[str]:
        return self._get_asset(self.NAME_ASSET_DOWNLOAD_URL, "")

    def get_size(self) -> typing.Optional[int]:
        return self._get_asset(self.NAME_ASSET_SIZE, 0)


class PluggableObjects:
    URL_REMOTE_OBJECTS = _hook.URL_REMOTE_OBJECTS

    def __init__(self, path_objects: Path):
        self.objects_path = path_objects
        self.fn = self.objects_path.name

    def sync(self):
        _request_asset(self.URL_REMOTE_OBJECTS, self.objects_path, self.fn)


class ModelHub:
    _fn2net = {}

    # Reverse proxy
    CDN_PREFIX = ""

    def __init__(self, onnx_prefix: str, name: str, models_dir: Path):
        """
        :param onnx_prefix: 模型文件名，不含有 ext
        :param name: 日志打印显示的标记
        :param models_dir: 模型所在的本地目录
        """
        self.models_dir = models_dir

        self.net = None
        self.flag = name
        self.fn = f"{onnx_prefix}.onnx" if not onnx_prefix.endswith(".onnx") else onnx_prefix
        self.model_path = models_dir.joinpath(self.fn)

        self.memory = Memory(fn=self.fn, memory_dir=models_dir.joinpath("_memory"))
        self.assets = Assets(fn=self.fn, assets_dir=models_dir.joinpath("_assets"))

    def pull_model(self):
        """
        1. node_id: Record the insertion point
        and indirectly judge the changes of the file with the same name

        2. assets.List: Record the item list of the release attachment,
        and directly determine whether there are undownloaded files

        3. assets.size: Record the amount of bytes inserted into the file,
        and directly determine whether the file is downloaded completely
        :return:
        """
        asset_node_id = self.assets.get_node_id()
        asset_download_url = self.assets.get_download_url()
        asset_size = self.assets.get_size()

        # Check for extreme cases
        if (
            not self.fn.endswith(".onnx")
            or not isinstance(asset_download_url, str)
            or not asset_download_url.startswith("https:")
        ):
            return

        # Matching conditions to trigger download tasks
        if (
            not self.model_path.exists()
            or self.model_path.stat().st_size != asset_size
            or self.memory.is_outdated(remote_node_id=asset_node_id)
        ):
            _request_asset(asset_download_url, self.model_path, self.fn)
            self.memory.dump(new_node_id=asset_node_id)

    def register_model(self) -> bool | None:
        """Load and register an existing model"""
        if (
            self.model_path.exists()
            and self.model_path.stat().st_size
            and not self.memory.is_outdated(self.assets.get_node_id())
        ):
            # An error will be reported when reading an incomplete model file
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._fn2net[self.fn] = self.net
            return True
        return False

    def offload(self):
        if self.fn in self._fn2net:
            self._fn2net.pop(self.fn)

    def match_net(self):
        """
        PluggableONNXModel 对象实例化时：
        - 自动读取并注册 objects.yaml 中注明的且已存在指定目录的模型对象，
        - 然而，objects.yaml 中表达的标签组所对应的模型文件不一定都已存在。
        - 初始化时不包含新的网络请求，即，不在初始化阶段下载缺失模型。

        match_net 模型被动拉取：
        - 在挑战进行时被动下载缺失的用于处理特定二分类任务的 ONNX 模型。
        - 匹配的模型会被自动下载、注册并返回。
        - 不在 objects.yaml 名单中的模型不会被下载

        升级的模型不支持热加载，需要重启程序才能读入，但新插入的模型可直接使用。
        :return:
        """
        if not self.net:
            self.pull_model()
            self.register_model()
        return self.net

    @property
    def fn2net(self):
        """check model objs"""
        return self._fn2net

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError


def _request_asset(asset_download_url: str, asset_path: Path, fn_tag: str):
    if isinstance(ModelHub.CDN_PREFIX, str) and ModelHub.CDN_PREFIX.startswith("https://"):
        parser = urlparse(ModelHub.CDN_PREFIX)
        scheme, netloc = parser.scheme, parser.netloc
        asset_download_url = f"{scheme}://{netloc}/{asset_download_url}"

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203"
    }
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        logger.debug(
            f"Downloading assets", fn_tag=fn_tag, url=asset_download_url, to=str(asset_path)
        )
        resp = client.get(asset_download_url)
        asset_path.write_bytes(resp.content)
