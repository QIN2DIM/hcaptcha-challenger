# -*- coding: utf-8 -*-
# Time       : 2023/9/5 9:30
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set
from urllib.parse import quote

import httpx
from github import Github, Auth
from github.Repository import Repository
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright, Page

import hcaptcha_challenger as solver
from hcaptcha_challenger import AgentT, QuestionResp, Malenia, label_cleaning, split_prompt_message
from hcaptcha_challenger.utils import SiteKey

solver.install(upgrade=True, flush_yolo=False)

TEMPLATE_BINARY_CHALLENGE = """
> Automated deployment @ utc {now}

### Prompt[en]

{prompt}

### New types of challenge

image_label_binary

### Sitelink

{sitelink}

### Screenshot of the challenge

![{prompt}]({screenshot_url})

"""

# Find new binary challenge from here
PENDING_SITEKEY = []


@dataclass
class Pigeon:
    label: str
    qr: QuestionResp
    sitekey: str
    canvas_path: Path

    issue_repo: Repository | None = None
    asset_repo: Repository | None = None

    binary_challenge_label = "🔥 challenge"

    def __post_init__(self):
        auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
        self.issue_repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")
        # self.issue_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")
        self.asset_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")

    @classmethod
    def build(cls, label, qr, sitekey, canvas_path):
        return cls(label=label, qr=qr, sitekey=sitekey, canvas_path=canvas_path)

    def _upload_asset(self) -> str | None:
        asset_path = f"captcha/{int(time.time())}.{self.label}.png"
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

    def _create_challenge_issues(self, challenge_prompt: str):
        asset_url = self._upload_asset()

        body = TEMPLATE_BINARY_CHALLENGE.format(
            now=str(datetime.now()),
            prompt=challenge_prompt,
            screenshot_url=asset_url,
            sitelink=SiteKey.as_sitelink(self.sitekey),
        )

        issue_title = f"[Challenge] {challenge_prompt}"
        resp = self.issue_repo.create_issue(
            title=issue_title,
            body=body,
            assignees=["QIN2DIM"],
            labels=[self.binary_challenge_label],
        )
        logger.success(f"create issue", html_url=resp.html_url)

    def _bypass_motion(self, challenge_prompt: str):
        for issue in self.issue_repo.get_issues(
            labels=[self.binary_challenge_label],
            state="all",
            since=datetime.now() - timedelta(days=14),
        ):
            label = split_prompt_message(challenge_prompt, lang="en")
            if label in issue.title:
                return True

    def notify(self):
        challenge_prompt = list(self.qr.requester_question.values())[0]
        challenge_prompt = label_cleaning(challenge_prompt)
        if not self._bypass_motion(challenge_prompt):
            self._create_challenge_issues(challenge_prompt)
        else:
            logger.info("bypass issue", prompt=challenge_prompt)


@dataclass
class Sentinel:
    per_times: int = 8
    tmp_dir: Path = Path(__file__).parent.joinpath("tmp_dir")
    nt_screenshot_dir = tmp_dir.joinpath("sentinel_challenge")

    lookup_labels: Set[str] = field(default_factory=set)
    pending_pigeon: asyncio.Queue[Pigeon] = None

    pending_sitekey = PENDING_SITEKEY

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
        logger.info("create tasks", pending_sitekey=self.pending_sitekey)

    async def register_pigeon(self, page: Page, label: str, agent, sitekey):
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        canvas = fl.locator("//div[@class='challenge-view']")

        canvas_path = self.nt_screenshot_dir.joinpath(f"{label}.png")
        await canvas.screenshot(type="png", path=canvas_path)

        pigeon = Pigeon.build(label, agent.qr, sitekey, canvas_path)
        self.pending_pigeon.put_nowait(pigeon)

        self.lookup_labels.add(label)

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
                print(f">> COLLETE - progress=[{pth}/{batch}] {label=} {probe=}")
                # {{< Sentinel Notify >}}
                if (
                    label
                    and "binary" in agent.qr.request_type
                    and label not in self.lookup_labels
                    and label not in label_alias
                ):
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
