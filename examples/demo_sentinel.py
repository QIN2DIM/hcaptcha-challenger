# -*- coding: utf-8 -*-
# Time       : 2023/9/5 9:30
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set

import httpx
from github import Github, Auth
from github.Repository import Repository
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

from hcaptcha_challenger import install
from hcaptcha_challenger.agents.playwright.control import AgentT, QuestionResp
from hcaptcha_challenger.agents.playwright.tarnished import Malenia
from hcaptcha_challenger.components.prompt_handler import label_cleaning
from hcaptcha_challenger.utils import SiteKey

install(upgrade=True, flush_yolo=False)

PENDING_SITEKEY = [SiteKey.epic, SiteKey.discord, SiteKey.user, SiteKey.top_level]

TEMPLATE_BINARY_CHALLENGE = """
### Prompt[en]

{prompt}

### New types of challenge

New prompt (for ex. Please select all the 45th President of the US)

### Sitelink

{sitelink}

### Screenshot of the challenge

![{prompt}]({screenshot_url})

"""


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
        # self.issue_repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")
        self.issue_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")
        self.asset_repo = Github(auth=auth).get_repo("QIN2DIM/cdn-relay")

    @classmethod
    def build(cls, label, qr, sitekey, canvas_path):
        return cls(label=label, qr=qr, sitekey=sitekey, canvas_path=canvas_path)

    def _upload_asset(self) -> str:
        asset_path = f"captcha/{self.label}.png"
        branch = "main"
        content = Path(self.canvas_path).read_bytes()

        self.asset_repo.update_file(
            path=asset_path,
            message="Automated deployment @ 2023-09-03 01:30:15 Asia/Shanghai",
            content=content,
            sha=hashlib.sha256(content).hexdigest(),
            branch=branch,
        )
        asset_url = (
            f"https://github.com/{self.asset_repo.full_name}/blob/{branch}/{asset_path}?raw=true"
        )
        logger.success(f"upload screenshot", asset_url=asset_url)
        return asset_url

    def _create_challenge_issues(self, challenge_prompt: str):
        asset_url = self._upload_asset()

        body = TEMPLATE_BINARY_CHALLENGE.format(
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
        issues = self.issue_repo.get_issues(
            labels=[self.binary_challenge_label],
            state="all",
            since=datetime.now() - timedelta(days=14),
        )
        issue_titles = [i.title for i in issues]
        if challenge_prompt in issue_titles:
            return True
        return False

    def notify(self):
        if not os.getenv("GITHUB_TOKEN"):
            logger.error("Failed to update issue, miss GITHUB TOKEN.")
            return

        challenge_prompt = list(self.qr.requester_question.values())[0]
        challenge_prompt = label_cleaning(challenge_prompt)
        if not self._bypass_motion(challenge_prompt):
            self._create_challenge_issues(challenge_prompt)


@dataclass
class Sentinel:
    per_times: int = 20
    tmp_dir: Path = Path(__file__).parent.joinpath("tmp_dir")
    nt_screenshot_dir = tmp_dir.joinpath("sentinel_challenge")

    lookup_labels: Set[str] = field(default_factory=set)
    pending_pigeon: asyncio.Queue[Pigeon] = None

    def __post_init__(self):
        self.nt_screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.pending_pigeon = asyncio.Queue()

    @logger.catch
    async def collete_datasets(self, context: ASyncContext, sitekey: str, batch: int = per_times):
        page = await context.new_page()
        agent = AgentT.from_page(page=page, tmp_dir=self.tmp_dir)

        sitelink = SiteKey.as_sitelink(sitekey)
        await page.goto(sitelink)

        await agent.handle_checkbox()

        for pth in range(1, batch + 1):
            try:
                label = await agent.collect()
                print(f">> COLLETE - progress={pth}/{batch} {label=} {sitelink=}")
            except httpx.HTTPError as err:
                logger.warning(f"Collection speed is too fast - reason={err}")
            except FileNotFoundError:
                pass
            else:
                # {{< Sentinel Notify >}}
                label_alias = {}

                if (
                    label
                    and "binary" in agent.qr.request_type
                    and label not in self.lookup_labels
                    and label not in label_alias
                ):
                    logger.success(f"lookup new challenge - {label=} {sitelink=}")
                    fl = page.frame_locator(agent.HOOK_CHALLENGE)
                    canvas = fl.locator("//div[@class='challenge-view']")
                    canvas_path = self.nt_screenshot_dir.joinpath(f"{label}.png")
                    await canvas.screenshot(type="png", path=canvas_path)
                    pigeon = Pigeon.build(label, agent.qr, sitekey, canvas_path)
                    self.pending_pigeon.put_nowait(pigeon)
                    self.lookup_labels.add(label)

            # Update MQ
            await page.wait_for_timeout(500)
            fl = page.frame_locator(agent.HOOK_CHALLENGE)
            await fl.locator("//div[@class='refresh button']").click()

        await page.close()

    async def bytedance(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(locale="en-US")
            await Malenia.apply_stealth(context)
            for sitekey in PENDING_SITEKEY:
                await self.collete_datasets(context, sitekey)
                while not self.pending_pigeon.empty():
                    pigeon = self.pending_pigeon.get_nowait()
                    pigeon.notify()
            await context.close()


if __name__ == "__main__":
    sentinel = Sentinel()
    asyncio.run(sentinel.bytedance())
