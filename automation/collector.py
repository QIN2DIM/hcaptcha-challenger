# -*- coding: utf-8 -*-
# Time       : 2023/8/31 20:54
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
import zipfile
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote

import cv2
import numpy as np
from github import Auth, Github
from github.GitRelease import GitRelease
from github.GitReleaseAsset import GitReleaseAsset
from github.Issue import Issue
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

from hcaptcha_challenger import split_prompt_message, label_cleaning
from hcaptcha_challenger.agents import AgentT, Malenia

TEMPLATE_BINARY_DATASETS = """
> Automated deployment @ utc {now}

| Attributes | Details                      |
| ---------- | ---------------------------- |
| prompt     | {prompt}                     |
| type       | `{type}`                     |
| cases      | {cases_num}                  |
| statistics | [#asset]({statistics})       |
| assets     | [{zip_name}]({download_url}) |

![]({montage_url})

"""

issue_labels = ["🔥 challenge", "🏹 ci: sentinel"]
issue_post_label = "☄️ci: collector"


@dataclass
class Gravitas:
    issue: Issue

    challenge_prompt: str = field(default=str)
    request_type: str = field(default=str)
    sitelink: str = field(default=str)
    mixed_label: str = field(default=str)
    """
    binary --> challenge_prompt
    area_select --> model_name
    """
    parent_prompt: str = field(default=str)

    typed_dir: Path = None
    """
    init by collector
    ./automation/tmp_dir/image_label_binary/{mixed_label}/
    ./automation/tmp_dir/image_label_area_select/{question}/{mixed_label}
    """

    cases_num: int = 0

    def __post_init__(self):
        body = [i for i in self.issue.body.split("\n") if i]
        self.challenge_prompt = body[2]
        self.request_type = body[4]
        self.sitelink = body[6]
        if "@" in self.issue.title:
            self.mixed_label = self.issue.title.split(" ")[1].strip()
            self.parent_prompt = self.issue.title.split("@")[-1].strip()
        else:
            self.mixed_label = split_prompt_message(self.challenge_prompt, lang="en")
            self.parent_prompt = "image_label_binary"

    @classmethod
    def from_issue(cls, issue: Issue):
        return cls(issue=issue)

    def montage(self):
        asset_name = f"{self.typed_dir.parent.name}.{self.typed_dir.name}"
        label_diagnose_name = label_cleaning(asset_name)
        __formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M%f")
        now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])

        thumbnail_path = self.typed_dir.parent.joinpath(
            f"thumbnail.{label_diagnose_name}.{now}.png"
        )

        images = []
        for img_name in os.listdir(self.typed_dir):
            img_path = self.typed_dir.joinpath(img_name)
            image = cv2.imread(str(img_path))
            image = cv2.resize(image, (256, 256))
            images.append(image)
            if len(images) == 9:
                break
        thumbnail = np.hstack(images)
        cv2.imwrite(str(thumbnail_path), thumbnail)

        return thumbnail_path

    def zip(self):
        asset_name = f"{self.typed_dir.parent.name}.{self.typed_dir.name}"
        label_diagnose_name = label_cleaning(asset_name)
        __formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M%f")
        now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])

        zip_path = self.typed_dir.parent.joinpath(f"{label_diagnose_name}.{now}.zip")
        logger.info("pack datasets", mixed=zip_path.name)

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for root, dirs, files in os.walk(self.typed_dir):
                for file in files:
                    zip_file.write(os.path.join(root, file), file)

        return zip_path

    @staticmethod
    def to_asset(archive_release: GitRelease, zip_path: Path) -> GitReleaseAsset:
        logger.info("upload datasets", mixed=zip_path.name)
        res = archive_release.upload_asset(path=str(zip_path))
        return res


@dataclass
class GravitasState:
    done: bool
    cases_num: int
    typed_dir: Path | None = None


def upload_thumbnail(canvas_path: Path, mixed_label: str) -> str | None:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    asset_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")

    asset_path = f"challenge-thumbnail/{time.time()}.{mixed_label}.png"
    branch = "main"
    content = canvas_path.read_bytes()

    asset_repo.update_file(
        path=asset_path,
        message="Automated deployment @ 2023-09-03 01:30:15 Asia/Shanghai",
        content=content,
        sha=hashlib.sha256(content).hexdigest(),
        branch=branch,
    )
    asset_quote_path = quote(asset_path)
    asset_url = (
        f"https://github.com/{asset_repo.full_name}/blob/{branch}/{asset_quote_path}?raw=true"
    )
    logger.info(f"upload challenge-thumbnail", asset_url=asset_url)
    return asset_url


def create_comment(asset: GitReleaseAsset, gravitas: Gravitas, gs: GravitasState):
    montage_path = gravitas.montage()
    montage_url = upload_thumbnail(canvas_path=montage_path, mixed_label=gravitas.mixed_label)

    body = TEMPLATE_BINARY_DATASETS.format(
        now=str(datetime.now()),
        prompt=gravitas.challenge_prompt,
        type=gravitas.request_type,
        cases_num=gravitas.cases_num,
        zip_name=asset.name,
        download_url=asset.browser_download_url,
        statistics=asset.url,
        montage_url=montage_url,
    )
    comment = gravitas.issue.create_comment(body=body)
    logger.success(f"create comment", html_url=comment.html_url)
    if gs.done:
        gravitas.issue.add_to_labels(issue_post_label)


def load_gravitas_from_issues() -> List[Gravitas]:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    issue_repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")

    tasks = []
    for issue in issue_repo.get_issues(
        labels=issue_labels,
        state="all",  # fixme `open`
        since=datetime.now() - timedelta(days=90),  # fixme `24hours`
    ):
        if not isinstance(issue.body, str):
            continue
        if "Automated deployment @" not in issue.body:
            continue
        tasks.append(Gravitas.from_issue(issue))

    return tasks


def get_archive_release() -> GitRelease:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    archive_release = (
        Github(auth=auth).get_repo("CaptchaAgent/hcaptcha-whistleblower").get_release(120534711)
    )
    return archive_release


@dataclass
class Collector:
    per_times: int = 10
    max_batch_times: int = 60

    tmp_dir: Path = Path(__file__).parent.joinpath("tmp_dir")

    pending_gravitas: List[Gravitas] = field(default_factory=list)
    sitelink2gravitas: Dict[str, Gravitas] = field(default_factory=dict)

    task_queue: asyncio.Queue | None = None

    link2collected: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        cpt = os.getenv("COLLECTOR_PER_TIMES", "")
        self.per_times = int(cpt) if cpt.isdigit() else self.per_times
        logger.debug("init collector parameter", per_times=self.per_times)

        if os.getenv("GITHUB_TOKEN"):
            self.pending_gravitas = load_gravitas_from_issues()

        self.task_queue = asyncio.Queue()

    @logger.catch
    async def _collete_datasets(self, context: ASyncContext, sitelink: str):
        if not self.link2collected.get(sitelink):
            self.link2collected[sitelink] = {}

        page = await context.new_page()
        agent = AgentT.from_page(page=page, tmp_dir=self.tmp_dir)

        await page.goto(sitelink)

        await agent.handle_checkbox()

        for pth in range(1, self.per_times + 1):
            with suppress(Exception):
                label = await agent.collect()

                probe = list(agent.qr.requester_restricted_answer_set.keys())
                print(f">> COLLETE - progress=[{pth}/{self.per_times}] {label=} {probe=}")

                mixed_label = probe[0] if len(probe) > 0 else label
                if mixed_label in self.link2collected[sitelink]:
                    self.link2collected[sitelink][mixed_label] += 1
                else:
                    self.link2collected[sitelink][mixed_label] = 1

            await page.wait_for_timeout(500)
            fl = page.frame_locator(agent.HOOK_CHALLENGE)
            await fl.locator("//div[@class='refresh button']").click()

    async def _step_preheat(self, context: ASyncContext):
        for i, gravitas in enumerate(self.sitelink2gravitas.values()):
            logger.debug(
                "preprocessing",
                progress=f"[{i + 1} / {len(self.sitelink2gravitas)}]",
                sitelink=gravitas.sitelink,
            )
            await self._collete_datasets(context, gravitas.sitelink)

    async def _step_reorganize(self):
        logger.info("Reorganize task queue", artifact=self.link2collected)

        for gravitas in self.sitelink2gravitas.values():
            gs = self.all_right(gravitas)
            if gs.done:
                logger.success("task done", mixed_label=gravitas.mixed_label, nums=gs.cases_num)
            else:
                max_value = 0
                best_sitelink = ""
                for sitelink, collected in self.link2collected.items():
                    if collected.get(gravitas.mixed_label, 0) > max_value:
                        max_value = collected[gravitas.mixed_label]
                        best_sitelink = sitelink
                if best_sitelink:
                    loop_times = int((self.max_batch_times - self.per_times) / self.per_times) + 1
                    loop_times = min(loop_times, 2)
                    for _ in range(loop_times):
                        await self.task_queue.put(best_sitelink)
                        logger.info(
                            "recur task",
                            challenge_prompt=gravitas.challenge_prompt,
                            nums=gs.cases_num,
                        )

    async def _step_focus(self, context: ASyncContext):
        max_qsize = self.task_queue.qsize()
        logger.info("focus on the mission", max_qsize=max_qsize)

        while not self.task_queue.empty():
            sitelink = self.task_queue.get_nowait()
            logger.debug(
                "focusing", qsize=f"[{self.task_queue.qsize()}/{max_qsize}]", sitelink=sitelink
            )
            await self._collete_datasets(context, sitelink)

    async def startup_collector(self):
        if not self.sitelink2gravitas:
            logger.info("exits", reseon="No pending tasks")
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(locale="en-US")
            await Malenia.apply_stealth(context)

            await self._step_preheat(context)
            await self._step_reorganize()
            await self._step_focus(context)

            await context.close()

    def all_right(self, gravitas: Gravitas) -> GravitasState:
        cases_num = 0
        for root, _, _ in os.walk(self.tmp_dir):
            root_dir = Path(root)
            if (
                gravitas.mixed_label != root_dir.name
                or gravitas.parent_prompt not in root_dir.parent.name
            ):
                continue
            cases_num = len(os.listdir(root))
            if "binary" in gravitas.request_type:
                done = bool(cases_num > 300)
                return GravitasState(typed_dir=root_dir, done=done, cases_num=cases_num)
            if "area_select" in gravitas.request_type:
                done = bool(cases_num > 20)
                return GravitasState(typed_dir=root_dir, done=done, cases_num=cases_num)
        return GravitasState(done=False, cases_num=cases_num)

    def prelude_tasks(self):
        for gravitas in self.pending_gravitas:
            gs = self.all_right(gravitas)
            if gs.done:
                logger.success("task done", prompt=gravitas.challenge_prompt, progress=gs.cases_num)
            else:
                self.sitelink2gravitas[gravitas.sitelink] = gravitas
                logger.info(
                    "parse task from issues",
                    prompt=gravitas.challenge_prompt,
                    sitelink=gravitas.sitelink,
                    progress=gs.cases_num,
                )

    def post_datasets(self):
        if not self.pending_gravitas:
            return

        archive_release = get_archive_release()
        for gravitas in self.pending_gravitas:
            gs = self.all_right(gravitas)
            if not gs.typed_dir:
                continue
            logger.info("posting datasets", pending_gravitas=gravitas)
            gravitas.typed_dir = gs.typed_dir
            gravitas.cases_num = gs.cases_num
            zip_path = gravitas.zip()
            asset = gravitas.to_asset(archive_release, zip_path)
            create_comment(asset, gravitas, gs)
            shutil.rmtree(zip_path, ignore_errors=True)

    async def bytedance(self):
        self.prelude_tasks()
        await self.startup_collector()
        self.post_datasets()


if __name__ == "__main__":
    collector = Collector()
    asyncio.run(collector.bytedance())
