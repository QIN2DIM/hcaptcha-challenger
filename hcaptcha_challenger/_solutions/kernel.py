# -*- coding: utf-8 -*-
# Time       : 2022/4/30 22:34
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import json
import os
import shutil
import time
import typing
from os.path import join
from urllib.parse import urlparse
from urllib.request import getproxies

import cv2
import requests
from loguru import logger


class ChallengeStyle:
    WATERMARK = 144  # onTrigger 128x128
    GENERAL = 128
    GAN = 144


class Memory:
    _fn2memory = {}

    ASSET_TOKEN = "RA_kw"

    def __init__(self, fn: str, dir_memory: str = None):
        self.fn = fn
        self._dir_memory = "model/_memory" if dir_memory is None else dir_memory

        self._build()

    def _build(self) -> typing.Optional[typing.Dict[str, str]]:
        """:return filename to nodeId"""
        if not self._fn2memory:
            os.makedirs(self._dir_memory, exist_ok=True)
            for memory_name in os.listdir(self._dir_memory):
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
            memory_name = join(self._dir_memory, f"{self.fn}.{new_node_id}")
            with open(memory_name, "w", encoding="utf8") as file:
                file.write(memory_name)
        else:
            memory_src = join(self._dir_memory, f"{self.fn}.{old_node_id}")
            memory_dst = join(self._dir_memory, f"{self.fn}.{new_node_id}")
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
    GITHUB_RELEASE_API = "https://api.github.com/repos/qin2dim/hcaptcha-challenger/releases"

    NAME_ASSETS = "assets"
    NAME_ASSET_NAME = "name"
    NAME_ASSET_SIZE = "size"
    NAME_ASSET_DOWNLOAD_URL = "browser_download_url"
    NAME_ASSET_NODE_ID = "node_id"

    _fn2assets = {}

    # Cache validity period: 2h
    CACHE_CONTROL = 7200

    def __init__(self, fn: str, dir_assets: str = None):
        self.fn = fn
        self._dir_assets = "model/_assets" if dir_assets is None else dir_assets

        self._pull()

    def _preload(self):
        """仅在 Assets._pull 网络请求发起前调用，将有效的本地缓存代替远端资源"""
        if self._fn2assets:
            return

        os.makedirs(self._dir_assets, exist_ok=True)
        assets = [i for i in os.listdir(self._dir_assets) if i.isdigit()]
        if len(assets) >= 1:
            asset_name = assets[-1]
            if int(asset_name) + self.CACHE_CONTROL > int(time.time()):
                recoded_name = join(self._dir_assets, asset_name)
                try:
                    with open(recoded_name, "r", encoding="utf8") as file:
                        self._fn2assets = json.load(file)
                except json.decoder.JSONDecodeError as err:
                    logger.warning(err)

    def _offload(self):
        """仅在 Assets._pull 网络请求发起后实现，用于更新本地缓存的内容及时间戳"""
        os.makedirs(self._dir_assets, exist_ok=True)
        for asset_fn in os.listdir(self._dir_assets):
            asset_src = join(self._dir_assets, asset_fn)
            asset_dst = join(self._dir_assets, f"_{asset_fn}")
            shutil.move(asset_src, asset_dst)
        recoded_name = join(self._dir_assets, str(int(time.time())))
        with open(recoded_name, "w", encoding="utf8") as file:
            json.dump(self._fn2assets, file)

    def _pull(self, skip_preload: bool = False) -> typing.Optional[typing.Dict[str, dict]]:
        def request_assets():
            logger.debug(f"Pulling AssetsObject from {self.GITHUB_RELEASE_API}")

            try:
                session = requests.session()
                resp = session.get(self.GITHUB_RELEASE_API, proxies=getproxies(), timeout=3)
                data = resp.json()[0]
            except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError) as err:
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

        if not skip_preload:
            self._preload()
        if not self._fn2assets:
            request_assets()
        return self._fn2assets

    def _get_asset(self, key: str, oncall_default: typing.Any):
        return self._fn2assets.get(self.fn, {}).get(key, oncall_default)

    def sync(self, force: typing.Optional[bool] = None, **kwargs):
        raise NotImplementedError

    @property
    def dir_assets(self):
        return self._dir_assets

    def get_node_id(self) -> typing.Optional[str]:
        return self._get_asset(self.NAME_ASSET_NODE_ID, "")

    def get_download_url(self) -> typing.Optional[str]:
        return self._get_asset(self.NAME_ASSET_DOWNLOAD_URL, "")

    def get_size(self) -> typing.Optional[int]:
        return self._get_asset(self.NAME_ASSET_SIZE, 0)


class PluggableObjects:
    URL_REMOTE_OBJECTS = (
        "https://raw.githubusercontent.com/QIN2DIM/hcaptcha-challenger/main/src/objects.yaml"
    )

    def __init__(self, path_objects: str):
        self.path_objects = path_objects
        self.fn = os.path.basename(self.path_objects)
        os.makedirs(os.path.dirname(self.path_objects), exist_ok=True)

    def sync(self):
        _request_asset(self.URL_REMOTE_OBJECTS, self.path_objects, self.fn)


class ModelHub:
    _fn2net = {}

    # Reverse proxy
    CDN_PREFIX = ""

    def __init__(self, onnx_prefix: str, name: str, dir_model: str):
        """
        :param onnx_prefix: 模型文件名，不含有 ext
        :param name: 日志打印显示的标记
        :param dir_model: 模型所在的本地目录
        """
        self._dir_model = "model" if dir_model is None else dir_model

        self.net = None
        self.flag = name
        self.fn = f"{onnx_prefix}.onnx" if not onnx_prefix.endswith(".onnx") else onnx_prefix
        self.path_model = join(dir_model, self.fn)

        self.memory = Memory(fn=self.fn, dir_memory=join(dir_model, "_memory"))
        self.assets = Assets(fn=self.fn, dir_assets=join(dir_model, "_assets"))

    @logger.catch()
    def pull_model(self, fn: str = None, path_model: str = None):
        """
        1. node_id: Record the insertion point
        and indirectly judge the changes of the file with the same name

        2. assets.List: Record the item list of the release attachment,
        and directly determine whether there are undownloaded files

        3. assets.size: Record the amount of bytes inserted into the file,
        and directly determine whether the file is downloaded completely
        :param fn:
        :param path_model:
        :return:
        """
        fn = self.fn if fn is None else fn
        path_model = self.path_model if path_model is None else path_model

        asset_node_id = self.assets.get_node_id()
        asset_download_url = self.assets.get_download_url()
        asset_size = self.assets.get_size()

        # Check for extreme cases
        if (
            not fn.endswith(".onnx")
            or not isinstance(asset_download_url, str)
            or not asset_download_url.startswith("https:")
        ):
            return

        # Matching conditions to trigger download tasks
        if (
            not os.path.exists(path_model)
            or os.path.getsize(path_model) != asset_size
            or self.memory.is_outdated(remote_node_id=asset_node_id)
        ):
            _request_asset(asset_download_url, path_model, fn)
            self.memory.dump(new_node_id=asset_node_id)

    @logger.catch()
    def register_model(self) -> typing.Optional[bool]:
        """Load and register an existing model"""
        # Update AssetsObject local cache
        if os.path.exists(self.path_model) and not self.memory.is_outdated(
            self.assets.get_node_id()
        ):
            self.net = cv2.dnn.readNetFromONNX(self.path_model)
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


def _request_asset(asset_download_url: str, asset_path: str, fn_tag: str):
    _params = {
        "headers": {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.27"
        },
        "stream": True,
        "proxies": getproxies(),
    }
    logger.debug(f"Downloading {fn_tag} from {asset_download_url}")
    if isinstance(ModelHub.CDN_PREFIX, str) and ModelHub.CDN_PREFIX.startswith("https://"):
        parser = urlparse(ModelHub.CDN_PREFIX)
        scheme, netloc = parser.scheme, parser.netloc
        asset_download_url = f"{scheme}://{netloc}/{asset_download_url}"
    with open(asset_path, "wb") as file, requests.get(asset_download_url, **_params) as response:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
