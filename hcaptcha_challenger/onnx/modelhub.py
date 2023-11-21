# -*- coding: utf-8 -*-
# Time       : 2023/8/19 18:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import gc
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import cv2
import httpx
import onnxruntime
import yaml
from cv2.dnn import Net
from loguru import logger
from onnxruntime import InferenceSession
from tenacity import *
from tqdm import tqdm

from hcaptcha_challenger.utils import from_dict_to_model

DEFAULT_KEYPOINT_MODEL = "COCO2020_yolov8m.onnx"


@logger.catch
@retry(
    retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.ConnectError)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=(stop_after_delay(30) | stop_after_attempt(5)),
    reraise=True,
)
def request_resource(url: str, save_path: Path):
    cdn_prefix = os.environ.get("MODELHUB_CDN_PREFIX", "")

    if cdn_prefix and cdn_prefix.startswith("https://"):
        parser = urlparse(cdn_prefix)
        scheme, netloc = parser.scheme, parser.netloc
        url = f"{scheme}://{netloc}/{url}"

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
    }
    with open(save_path, "wb") as download_file:
        with httpx.Client(headers=headers, follow_redirects=True, http2=True) as client:
            with client.stream("GET", url) as response:
                total = int(response.headers["Content-Length"])
                with tqdm(
                    total=total,
                    unit_scale=True,
                    unit_divisor=1024,
                    unit="B",
                    desc=f"Installing {save_path.parent.name}/{save_path.name}",
                ) as progress:
                    num_bytes_downloaded = response.num_bytes_downloaded
                    for chunk in response.iter_bytes():
                        download_file.write(chunk)
                        progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                        num_bytes_downloaded = response.num_bytes_downloaded


@dataclass
class ReleaseAsset:
    id: int
    node_id: str
    name: str
    size: int
    browser_download_url: str


@dataclass
class Assets:
    """
    ONNX model manager
    """

    release_url: str
    """
    GitHub Release URL
    Such as https://api.github.com/repos/QIN2DIM/hcaptcha-challenger/releases
    """

    _assets_dir: Path = Path(__file__).parent.joinpath("models/_assets")
    """
    The local path of index file
    """

    _memory_dir: Path = Path(__file__).parent.joinpath("models/_memory")
    """
    Archive historical versions of a model
    """

    cache_lifetime: timedelta = timedelta(hours=2)
    """
    The effective time of the index file
    """

    _name2asset: Dict[str, ReleaseAsset] = field(default_factory=dict)
    """
    { model_name.onnx: {ReleaseAsset} }
    """

    _name2node: Dict[str, str] = field(default_factory=dict)
    """
    model_name.onnx to asset_node_id
    """

    def __post_init__(self):
        for ck in [self._assets_dir, self._memory_dir]:
            ck.mkdir(mode=0o777, parents=True, exist_ok=True)

    @classmethod
    def from_release_url(cls, release_url: str, **kwargs):
        instance = cls(release_url=release_url, **kwargs)

        # Assets - latest information
        # Called only before the Assets._pull network request is initiated,
        # and the effective local cache will replace the remote resource
        assets = [instance._assets_dir.joinpath(i) for i in os.listdir(instance._assets_dir)]
        if assets:
            resp_path = assets[-1]
            resp_ctime = datetime.fromtimestamp(resp_path.stat().st_ctime)
            if resp_ctime + instance.cache_lifetime > datetime.now():
                try:
                    data = json.loads(resp_path.read_text(encoding="utf8"))
                    instance._name2asset = {
                        k: from_dict_to_model(ReleaseAsset, v) for k, v in data.items()
                    }
                except JSONDecodeError as err:
                    logger.warning(err)

        # Memory - version control
        for x in os.listdir(instance._memory_dir):
            name, node = x.rsplit(".", maxsplit=1)
            instance._name2node[name] = node

        return instance

    def get_focus_asset(self, focus_name: str) -> ReleaseAsset:
        return self._name2asset.get(focus_name)

    @logger.catch
    @retry(
        retry=retry_if_exception_type(httpx.ConnectTimeout),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=(stop_after_delay(30) | stop_after_attempt(5)),
        reraise=True,
    )
    def flush_runtime_assets(self, upgrade: bool = False):
        """Request assets index from remote repository"""
        self._name2asset = {}

        if upgrade is True:
            shutil.rmtree(self._assets_dir, ignore_errors=True)
            self._assets_dir.mkdir(parents=True, exist_ok=True)

        assets_paths = [self._assets_dir.joinpath(i) for i in os.listdir(self._assets_dir)]
        is_outdated = False
        if assets_paths:
            resp_path = assets_paths[-1]
            resp_ctime = datetime.fromtimestamp(resp_path.stat().st_ctime)
            if resp_ctime + self.cache_lifetime < datetime.now():
                is_outdated = True

        # Request assets index from remote repository
        if upgrade is True or not assets_paths or is_outdated:
            try:
                headers = {
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
                }
                client = httpx.Client(timeout=3, http2=True, headers=headers)
                resp = client.get(self.release_url)
                data = resp.json()[0]
                assets: List[dict] = data.get("assets", [])
                for asset in assets:
                    release_asset = from_dict_to_model(ReleaseAsset, asset)
                    self._name2asset[asset["name"]] = release_asset
            except (httpx.ConnectError, JSONDecodeError) as err:
                logger.error(err)
            except (AttributeError, IndexError, KeyError) as err:
                logger.error(err)

        # Only implemented after the Assets._pull network request is initiated,
        # it is used to update the content and timestamp of the local cache
        for asset_fn in os.listdir(self._assets_dir):
            asset_src = self._assets_dir.joinpath(asset_fn)
            asset_dst = self._assets_dir.joinpath(f"_{asset_fn.replace('_', '')}")
            shutil.move(asset_src, asset_dst)
        assets_path = self._assets_dir.joinpath(f"{int(time.time())}.json")
        data = {k: v.__dict__ for k, v in self._name2asset.items()}
        assets_path.write_text(json.dumps(data, ensure_ascii=True, indent=2))

    def archive_memory(self, focus_name: str, new_node_id: str):
        """
        Archive model version to local
        :param focus_name: model name with .onnx suffix
        :param new_node_id: node_id of the GitHub release, ONNX model file.
        :return:
        """
        old_node_id = self._name2node.get(focus_name, "")

        self._name2node[focus_name] = new_node_id

        # Create or update a history tracking node
        if not old_node_id:
            memory_node = self._memory_dir.joinpath(f"{focus_name}.{new_node_id}")
            memory_node.write_text(str(memory_node))
        else:
            memory_src = self._memory_dir.joinpath(f"{focus_name}.{old_node_id}")
            memory_dst = self._memory_dir.joinpath(f"{focus_name}.{new_node_id}")
            shutil.move(memory_src, memory_dst)

    def is_outdated(self, focus_name: str):
        """
        Diagnostic steps for delayed reflection to maintain consistency at both ends of a distributed network
        :param focus_name: model name with .onnx suffix
        :return:
        """
        focus_asset = self._name2asset.get(focus_name)
        if not focus_asset:
            return

        local_node_id = self._name2node.get(focus_name, "")
        if not local_node_id:
            return

        if local_node_id != focus_asset.node_id:
            return True
        return False


@dataclass
class ModelHub:
    """
    Manage pluggable models. Provides high-level interfaces
    such as model download, model cache, and model scheduling.
    """

    models_dir = Path(__file__).parent.joinpath("models")
    assets_dir = models_dir.joinpath("_assets")
    objects_path = models_dir.joinpath("objects.yaml")

    lang: str = "en"

    label_alias: Dict[str, str] = field(default_factory=dict)
    """
    Image classification
    ---
    The most basic function
    Storing a series of mappings from model names to short prompts, 
    I .e., what model to use to handle what challenge is determined by this dictionary.
    """

    yolo_names: List[str] = field(default_factory=list)
    ashes_of_war: Dict[str, List[str]] = field(default_factory=dict)
    """
    Object Detection
    ---
    Provide a series of object detection models applied to special tasks.
    The yolo_names stores the label names of all task objects that the model can process.
    """

    nested_categories: Dict[str, List[str]] = field(default_factory=dict)
    """
    Model Rank.Strategy
    ---
    Provide a string of small model clusters for a prompt to realize 
    "find the ｛z｝ pictures most similar to ｛y｝ in the ｛x_i｝ pictures"
    """

    circle_segment_model: str = field(default=str)
    """
    Image Segmentation
    ---
    A model trained specifically for image segmentation tasks 
    that can separate background and foreground with close to 100 percent accuracy
    """

    datalake: Dict[str, DataLake] = field(default_factory=dict)
    """
    ViT zero-shot image classification
    ---
    Used to generate prompt templates to intensify inserted CLIP model and improve accuracy.
    """

    DEFAULT_CLIP_VISUAL_MODEL: str = "visual_CLIP_RN50.openai.onnx"
    DEFAULT_CLIP_TEXTUAL_MODEL: str = "textual_CLIP_RN50.openai.onnx"
    """
    Available Model
    --- 1180+ MiB
    DEFAULT_CLIP_VISUAL_MODEL: str = "visual_CLIP_ViT-B-32.openai.onnx"
    DEFAULT_CLIP_TEXTUAL_MODEL: str = "textual_CLIP_ViT-B-32.openai.onnx"
    --- 658.3 MiB
    DEFAULT_CLIP_VISUAL_MODEL: str = "visual_CLIP_RN50.openai.onnx"
    DEFAULT_CLIP_TEXTUAL_MODEL: str = "textual_CLIP_RN50.openai.onnx"
    --- 3300+ MiB
    DEFAULT_CLIP_VISUAL_MODEL: str = "visual_CLIP-ViT-L-14-DataComp.XL-s13B-b90K.onnx"
    DEFAULT_CLIP_TEXTUAL_MODEL: str = "textual_CLIP-ViT-L-14-DataComp.XL-s13B-b90K.onnx"
    """

    clip_candidates: Dict[str, List[str]] = field(default_factory=dict)
    """
    CLIP self-supervised candidates
    """

    release_url: str = ""
    objects_url: str = ""

    assets: Assets = None

    _name2net: Dict[str, Net | InferenceSession] = field(default_factory=dict)
    """
    { model_name1.onnx: cv2.dnn.Net }
    { model_name2.onnx: onnxruntime.InferenceSession }
    """

    def __post_init__(self):
        self.assets_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    @classmethod
    def from_github_repo(cls, username: str = "QIN2DIM", lang: str = "en", **kwargs):
        release_url = f"https://api.github.com/repos/{username}/hcaptcha-challenger/releases"
        objects_url = f"https://raw.githubusercontent.com/{username}/hcaptcha-challenger/main/src/objects.yaml"

        instance = cls(release_url=release_url, objects_url=objects_url, lang=lang)
        instance.assets = Assets.from_release_url(release_url)

        return instance

    def pull_objects(self, upgrade: bool = False):
        """Network request"""
        if (
            upgrade
            or not self.objects_path.exists()
            or not self.objects_path.stat().st_size
            or time.time() - self.objects_path.stat().st_mtime > 3600
        ):
            request_resource(self.objects_url, self.objects_path)

    def parse_objects(self):
        """Try to load label_alias from local database"""
        if not self.objects_path.exists():
            return

        data = yaml.safe_load(self.objects_path.read_text(encoding="utf8"))
        if not data:
            os.remove(self.objects_path)
            return

        label_to_i18n_mapping: dict = data.get("label_alias", {})
        if label_to_i18n_mapping:
            for model_name, lang_to_prompts in label_to_i18n_mapping.items():
                for lang, prompts in lang_to_prompts.items():
                    if lang != self.lang:
                        continue
                    self.label_alias.update({prompt.strip(): model_name for prompt in prompts})

        yolo2names: Dict[str, List[str]] = data.get("ashes_of_war", {})
        if yolo2names:
            self.yolo_names = [cl for cc in yolo2names.values() for cl in cc]
            self.ashes_of_war = yolo2names

        nested_categories = data.get("nested_categories", {})
        self.nested_categories = nested_categories or {}

        self.circle_segment_model = data.get(
            "circle_seg", "appears_only_once_2309_yolov8s-seg.onnx"
        )

        datalake = data.get("datalake", {})
        if datalake:
            for prompt, serialized_binary in datalake.items():
                datalake[prompt] = DataLake.from_serialized(serialized_binary)
        self.datalake = datalake or {}

        clip_candidates = data.get("clip_candidates", {})
        self.clip_candidates = clip_candidates or {}

    def pull_model(self, focus_name: str):
        """
        1. node_id: Record the insertion point
        and indirectly judge the changes of the file with the same name

        2. assets.List: Record the item list of the release attachment,
        and directly determine whether there are undownloaded files

        3. assets.size: Record the amount of bytes inserted into the file,
        and directly determine whether the file is downloaded completely

        :param focus_name: model_name.onnx  Such as `mobile.onnx`
        :return:
        """
        focus_asset = self.assets.get_focus_asset(focus_name)
        if not focus_asset:
            return

        # Matching conditions to trigger download tasks
        model_path = self.models_dir.joinpath(focus_name)
        if (
            not model_path.exists()
            or model_path.stat().st_size != focus_asset.size
            or self.assets.is_outdated(focus_name)
        ):
            try:
                request_resource(focus_asset.browser_download_url, model_path.absolute())
            except httpx.ConnectTimeout as err:
                logger.error("Failed to download resource, try again", err=err)
            else:
                self.assets.archive_memory(focus_name, focus_asset.node_id)

    def active_net(self, focus_name: str) -> Net | InferenceSession | None:
        """Load and register an existing model"""
        model_path = self.models_dir.joinpath(focus_name)
        if (
            model_path.exists()
            and model_path.stat().st_size
            and not self.assets.is_outdated(focus_name)
        ):
            if "yolo" in focus_name.lower() or "clip" in focus_name.lower():
                net = onnxruntime.InferenceSession(
                    model_path, providers=onnxruntime.get_available_providers()
                )
            else:
                net = cv2.dnn.readNetFromONNX(str(model_path))
            self._name2net[focus_name] = net
            return net

    def match_net(
        self, focus_name: str, *, install_only: bool = False
    ) -> Net | InferenceSession | None:
        """
        When a PluggableONNXModel object is instantiated:
        ---

        - It automatically reads and registers model objects specified in objects.yaml
        that already exist in the designated directory.
        - However, the model files corresponding to the label groups expressed in objects.yaml
        do not necessarily all exist yet.
        - No new network requests are made during initialization,
        i.e. missing models are not downloaded during the initialization phase.

        match_net models are passively pulled:
        ---

        - Missing ONNX models used for handling specific binary classification tasks are
        passively downloaded during the challenge.
        - Matching models are automatically downloaded, registered, and returned.
        - Models not on the objects.yaml list will not be downloaded.

        [!] The newly inserted model can be used directly.
        :param install_only:
        :param focus_name: model_name with .onnx suffix
        :return:
        """
        net = self._name2net.get(focus_name)
        if not net:
            self.pull_model(focus_name)
            if not install_only:
                net = self.active_net(focus_name)
        return net

    def unplug(self):
        for ash in self.ashes_of_war:
            if ash not in self._name2net:
                continue
            del self._name2net[ash]
            gc.collect()

        for m in [self.DEFAULT_CLIP_TEXTUAL_MODEL, self.DEFAULT_CLIP_VISUAL_MODEL]:
            if m in self._name2net:
                del self._name2net[m]
                gc.collect()

    def apply_ash_of_war(self, ash: str) -> Tuple[str, List[str]]:
        # Prelude - pending DensePose
        if "head of " in ash and "animal" in ash:
            for model_name, covered_class in self.ashes_of_war.items():
                if "head" not in model_name:
                    continue
                for class_name in covered_class:
                    if class_name.replace("-head", "") in ash:
                        return model_name, covered_class

        # Prelude - Ordered dictionary
        for model_name, covered_class in self.ashes_of_war.items():
            for class_name in covered_class:
                if class_name in ash:
                    return model_name, covered_class

        # catch-all rules
        return DEFAULT_KEYPOINT_MODEL, self.ashes_of_war[DEFAULT_KEYPOINT_MODEL]

    def lookup_ash_of_war(self, ash: str):  # fixme
        """catch-all default cases"""
        if "can be eaten" in ash:
            for model_name, covered_class in self.ashes_of_war.items():
                if "can_be_eaten" in model_name:
                    yield model_name, covered_class

        if "not an animal" in ash:
            for model_name, covered_class in self.ashes_of_war.items():
                if "notanimal" in model_name:
                    yield model_name, covered_class

        if "head of " in ash and "animal" in ash:
            for model_name, covered_class in self.ashes_of_war.items():
                if "head" in model_name:
                    yield model_name, covered_class

        if "animal" in ash and "not belong to the sea" in ash:
            for model_name, covered_class in self.ashes_of_war.items():
                if (
                    "notseaanimal" in model_name
                    or "fantasia_elephant" in model_name
                    or "fantasia_cat" in model_name
                ):
                    yield model_name, covered_class

        for model_name, covered_class in self.ashes_of_war.items():
            binder = model_name.split("_")
            if len(binder) > 2 and binder[-2].isdigit():
                binder = " ".join(model_name.split("_")[:-2])
                if binder in ash:
                    yield model_name, covered_class
            else:
                for class_name in covered_class:
                    if class_name in ash:
                        yield model_name, covered_class


@dataclass
class DataLake:
    positive_labels: List[str] = field(default_factory=list)
    """
    Indicate the label with the meaning "True", 
    preferably an independent noun or clause
    """

    negative_labels: List[str] = field(default_factory=list)
    """
    Indicate the label with the meaning "False", 
    preferably an independent noun or clause
    """

    joined_dirs: List[str] | Path | None = None
    """
    Attributes reserved for AutoLabeling
    Used to indicate the directory where the dataset is located

    input_dir = db_dir.joinpath(*joined_dirs).absolute()
    """

    raw_prompt: str = ""
    """
    Challenge prompt or keywords after being divided
    
    !! IMPORT !!
    Only for unsupervised challenges.
    Please do not read in during the initialization phase.
    """

    PREMISED_YES: str = "This is a picture that looks like {}."
    PREMISED_BAD: str = "This is a picture that don't look like {}."
    """
    Insert self-supervised prompt
    """

    @classmethod
    def from_challenge_prompt(cls, raw_prompt: str):
        return cls(raw_prompt=raw_prompt)

    @classmethod
    def from_serialized(cls, fields: Dict[str, List[str]]):
        positive_labels = []
        negative_labels = []
        for kb, labels in fields.items():
            kb = kb.lower()
            if "pos" in kb or kb.startswith("t"):
                positive_labels = labels
            elif "neg" in kb or kb.startswith("f"):
                negative_labels = labels
        return cls(positive_labels=positive_labels, negative_labels=negative_labels)

    @classmethod
    def from_binary_labels(cls, positive_labels: List[str], negative_labels: List[str]):
        return cls(positive_labels=positive_labels, negative_labels=negative_labels)
