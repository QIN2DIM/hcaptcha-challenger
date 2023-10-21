# -*- coding: utf-8 -*-
# Time       : 2023/9/5 9:30
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set, List
from urllib.parse import quote

import httpx
from github import Github, Auth
from github.Repository import Repository
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright, Page

import hcaptcha_challenger as solver
from hcaptcha_challenger import label_cleaning, split_prompt_message
from hcaptcha_challenger.agents import AgentT, QuestionResp, Malenia
from hcaptcha_challenger.onnx.yolo import is_matched_ash_of_war
from hcaptcha_challenger.utils import SiteKey

solver.install(upgrade=True, flush_yolo=False)

TEMPLATE_BINARY_CHALLENGE = """
> Automated deployment @ utc {now}

### Prompt[en]

{prompt}

### request_type

`{request_type}`

### Sitelink

{sitelink}

### Screenshot of the challenge

![{prompt}]({screenshot_url})

"""


@dataclass
class Pigeon:
    mixed_label: str
    """
    area_select --> probe @ question
    binary --> not diagnosed label name 
    """

    qr: QuestionResp
    sitekey: str
    canvas_path: Path

    issue_repo: Repository | None = None
    asset_repo: Repository | None = None

    issue_prompt: str = field(default=str)
    """
    area_select --> probe @ question
    binary --> question
    """

    request_type: str = field(default=str)
    challenge_prompt: str = field(default=str)
    issue_labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    issue_title: str = field(default=str)

    def __post_init__(self):
        auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
        self.issue_repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")
        # self.issue_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")
        self.asset_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")

        self.request_type = self.qr.request_type
        if shape_type := self.qr.request_config.get("shape_type"):
            self.request_type = f"{self.request_type}: {shape_type}"

        challenge_prompt = list(self.qr.requester_question.values())[0]
        self.challenge_prompt = label_cleaning(challenge_prompt)

        self.issue_prompt = self.challenge_prompt
        if ":" in self.request_type:
            self.issue_prompt = self.mixed_label

        self.issue_title = f"[Challenge] {self.issue_prompt}"

        self.issue_labels = ["🔥 challenge"]
        self.issue_post_label = "🏹 ci: sentinel"
        self.assignees = ["QIN2DIM"]

    @classmethod
    def build(cls, label, qr, sitekey, canvas_path):
        return cls(mixed_label=label, qr=qr, sitekey=sitekey, canvas_path=canvas_path)

    def _upload_asset(self) -> str | None:
        asset_path = f"captcha/{int(time.time())}.{self.mixed_label}.png"
        branch = "main"
        content = Path(self.canvas_path).read_bytes()

        self.asset_repo.update_file(
            path=asset_path,
            message="Automated deployment @ 2023-09-03 01:30:15 Asia/Shanghai",
            content=content,
            sha=hashlib.sha256(content).hexdigest(),
            branch=branch,
        )
        asset_quote_path = quote(asset_path)
        asset_url = f"https://github.com/{self.asset_repo.full_name}/blob/{branch}/{asset_quote_path}?raw=true"
        logger.success(f"upload screenshot", asset_url=asset_url)
        return asset_url

    def _create_challenge_issues(self):
        asset_url = self._upload_asset()

        body = TEMPLATE_BINARY_CHALLENGE.format(
            now=str(datetime.now()),
            prompt=self.issue_prompt,
            request_type=self.request_type,
            screenshot_url=asset_url,
            sitelink=SiteKey.as_sitelink(self.sitekey),
        )

        resp = self.issue_repo.create_issue(
            title=self.issue_title,
            body=body,
            assignees=self.assignees,
            labels=self.issue_labels + [self.issue_post_label],
        )
        logger.success(f"create issue", html_url=resp.html_url)

    def _bypass_motion(self):
        """
        mixed_label in issue.title
        ------------

        area_select
        ------------

                    squirrelwatercolor-lmv2 @ please click on the squirrel
                    ↓
        [Challenge] squirrelwatercolor-lmv2 @ please click on the squirrel

        binary
        ------------

                                                         chess piece
                                                         ↓
        [Challenge] Please click each image containing a chess piece

        :return:
        """
        for issue in self.issue_repo.get_issues(
            labels=self.issue_labels,
            state="all",
            since=datetime.now() - timedelta(days=14),
            assignee=self.assignees[0],
        ):
            mixed_label = split_prompt_message(self.issue_prompt, lang="en")
            if issue.created_at + timedelta(hours=24) > datetime.now():
                issue.add_to_labels("🏹 ci: sentinel")
            if mixed_label in issue.title.lower():
                return True

    def notify(self):
        if not self._bypass_motion():
            self._create_challenge_issues()
        else:
            logger.info("bypass issue", issue_prompt=self.issue_prompt)


@dataclass
class Sentinel:
    per_times: int = 8
    tmp_dir: Path = Path(__file__).parent.joinpath("tmp_dir")
    nt_screenshot_dir = tmp_dir.joinpath("sentinel_challenge")

    lookup_labels: Set[str] = field(default_factory=set)
    pending_pigeon: asyncio.Queue[Pigeon] = None

    pending_sitekey: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.nt_screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.pending_pigeon = asyncio.Queue()

        spt = os.getenv("SENTINEL_PER_TIMES", "")
        self.per_times = int(spt) if spt.isdigit() else 8

        for skn in os.environ:
            if skn.startswith("SITEKEY_"):
                sk = os.environ[skn]
                logger.info("get sitekey from env", name=skn, sitekey=sk)
                self.pending_sitekey.append(sk)

        self.pending_sitekey = list(set(self.pending_sitekey))

    async def register_pigeon(self, page: Page, mixed_label: str, agent, sitekey):
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        canvas = fl.locator("//div[@class='challenge-view']")

        canvas_path = self.nt_screenshot_dir.joinpath(f"{mixed_label}.png")
        await canvas.screenshot(type="png", path=canvas_path)

        pigeon = Pigeon.build(mixed_label, agent.qr, sitekey, canvas_path)
        self.pending_pigeon.put_nowait(pigeon)

        self.lookup_labels.add(mixed_label)

    @logger.catch
    async def collete_datasets(self, context: ASyncContext, sitekey: str, batch: int = per_times):
        page = await context.new_page()
        agent = AgentT.from_page(page=page, tmp_dir=self.tmp_dir)
        label_alias = agent.modelhub.label_alias

        sitelink = SiteKey.as_sitelink(sitekey)
        await page.goto(sitelink)

        await agent.handle_checkbox()

        for pth in range(1, batch + 1):
            try:
                label = await agent.collect()
            except httpx.HTTPError as err:
                logger.warning(f"Collection speed is too fast", reason=err)
            except FileNotFoundError:
                pass
            else:
                probe = list(agent.qr.requester_restricted_answer_set.keys())
                mixed_label = label
                print(f">> COLLETE - progress=[{pth}/{batch}] {label=} {probe=}")

                if not mixed_label:
                    pass
                # Match: image_label_area_select
                elif probe:
                    # probe --> one-step model
                    mixed_label = f"{probe[0]} @ {label}"
                    if mixed_label not in self.lookup_labels and not any(
                        is_matched_ash_of_war(agent.ash, c) for c in agent.modelhub.yolo_names
                    ):
                        logger.info(f"lookup new challenge", label=mixed_label, sitelink=sitelink)
                        await self.register_pigeon(page, mixed_label, agent, sitekey)
                # Match: image_label_binary
                elif agent.qr.request_type == "image_label_binary":
                    if mixed_label not in self.lookup_labels and mixed_label not in label_alias:
                        logger.info(f"lookup new challenge", label=label, sitelink=sitelink)
                        await self.register_pigeon(page, label, agent, sitekey)

            # Update MQ
            await page.wait_for_timeout(500)
            fl = page.frame_locator(agent.HOOK_CHALLENGE)
            await fl.locator("//div[@class='refresh button']").click()

        await page.close()

    async def bytedance(self):
        if not self.pending_sitekey:
            logger.info("No pending tasks, sentinel exits", tasks=self.pending_sitekey)
            return
        logger.info("create tasks", pending_sitekey=self.pending_sitekey)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(locale="en-US")
            await Malenia.apply_stealth(context)
            for sitekey in self.pending_sitekey:
                await self.collete_datasets(context, sitekey)
                while not self.pending_pigeon.empty():
                    pigeon = self.pending_pigeon.get_nowait()
                    pigeon.notify()
            await context.close()


if __name__ == "__main__":
    if os.getenv("GITHUB_TOKEN"):
        sentinel = Sentinel()
        asyncio.run(sentinel.bytedance())
    else:
        logger.critical("Failed to startup sentinel, miss GITHUB TOKEN.")
