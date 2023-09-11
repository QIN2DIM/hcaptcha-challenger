# -*- coding: utf-8 -*-
# Time       : 2023/8/31 20:54
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import httpx
from github import Auth, Github
from github.GitRelease import GitRelease
from github.GitReleaseAsset import GitReleaseAsset
from github.Issue import Issue
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

from hcaptcha_challenger import AgentT, Malenia
from hcaptcha_challenger import split_prompt_message, diagnose_task

TEMPLATE_BINARY_DATASETS = """
> Automated deployment @ utc {now}

| Attributes | Details                      |
| ---------- | ---------------------------- |
| prompt     | {prompt}                     |
| type       | `{type}`                     |
| cases      | {cases_num}                  |
| statistics | [#asset]({statistics})       |
| assets     | [{zip_name}]({download_url}) |

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
        else:
            self.mixed_label = split_prompt_message(self.challenge_prompt, lang="en")

    @classmethod
    def from_issue(cls, issue: Issue):
        return cls(issue=issue)

    @property
    def zip_path(self) -> Path:
        label_diagnose_name = diagnose_task(self.typed_dir.name)
        now = datetime.strptime(str(datetime.now()), "%Y-%m-%d %H:%M:%S.%f").strftime("%Y%m%d%H%M")
        zip_path = self.typed_dir.parent.joinpath(f"{label_diagnose_name}.{now}.zip")
        return zip_path

    def zip(self):
        logger.info("pack datasets", mixed=self.zip_path.name)
        with zipfile.ZipFile(self.zip_path, "w") as zip_file:
            for root, dirs, files in os.walk(self.typed_dir):
                for file in files:
                    zip_file.write(os.path.join(root, file), file)

    def to_asset(self, archive_release: GitRelease) -> GitReleaseAsset:
        logger.info("upload datasets", mixed=self.zip_path.name)
        res = archive_release.upload_asset(path=str(self.zip_path))
        return res


def create_comment(asset: GitReleaseAsset, gravitas: Gravitas, sign_label: bool = False):
    body = TEMPLATE_BINARY_DATASETS.format(
        now=str(datetime.now()),
        prompt=gravitas.challenge_prompt,
        type=gravitas.request_type,
        cases_num=gravitas.cases_num,
        zip_name=asset.name,
        download_url=asset.browser_download_url,
        statistics=asset.url,
    )
    comment = gravitas.issue.create_comment(body=body)
    logger.success(f"create comment", html_url=comment.html_url)
    if sign_label:
        gravitas.issue.add_to_labels(issue_post_label)


def load_gravitas_from_issues() -> List[Gravitas]:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    issue_repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")

    tasks = []
    for issue in issue_repo.get_issues(
        labels=issue_labels,
        state="open",  # fixme `open`
        since=datetime.now() - timedelta(hours=24),  # fixme `24hours`
    ):
        if "Automated deployment @" not in issue.body:
            continue
        tasks.append(Gravitas.from_issue(issue))

    return tasks


def get_archive_release() -> GitRelease:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    archive_release = (
        Github(auth=auth)
        .get_repo("captcha-challenger/hcaptcha-whistleblower")
        .get_release(120534711)
    )
    return archive_release


@dataclass
class GravitasState:
    done: bool
    cases_num: int
    typed_dir: Path | None = None


# noinspection DuplicatedCode
@dataclass
class Collector:
    per_times: int = 3
    loop_times: int = 1
    tmp_dir: Path = Path(__file__).parent.joinpath("tmp_dir")

    pending_gravitas: List[Gravitas] = field(default_factory=list)
    sitelink2gravitas: Dict[str, Gravitas] = field(default_factory=dict)

    task_queue: asyncio.Queue | None = None

    def __post_init__(self):
        cpt = os.getenv("COLLECTOR_PER_TIMES", "")
        self.per_times = int(cpt) if cpt.isdigit() else self.per_times
        logger.debug("init collector parameter", per_times=self.per_times)

        clt = os.getenv("COLLECTOR_LOOP_TIMES", "")
        self.loop_times = int(clt) if clt.isdigit() else self.loop_times
        logger.debug("init collector parameter", loop_times=self.loop_times)

        if os.getenv("GITHUB_TOKEN"):
            self.pending_gravitas = load_gravitas_from_issues()

    @logger.catch
    async def _collete_datasets(self, context: ASyncContext, sitelink: str):
        page = await context.new_page()
        agent = AgentT.from_page(page=page, tmp_dir=self.tmp_dir)

        await page.goto(sitelink)

        await agent.handle_checkbox()

        for pth in range(1, self.per_times + 1):
            try:
                label = await agent.collect()
            except (httpx.HTTPError, httpx.ConnectTimeout) as err:
                logger.warning(f"Collection speed is too fast", reason=err)
                await page.wait_for_timeout(500)
            except FileNotFoundError:
                pass
            except Exception as err:
                print(err)
            else:
                probe = list(agent.qr.requester_restricted_answer_set.keys())
                print(f">> COLLETE - progress=[{pth}/{self.per_times}] {label=} {probe=}")

            await page.wait_for_timeout(500)
            fl = page.frame_locator(agent.HOOK_CHALLENGE)
            await fl.locator("//div[@class='refresh button']").click()

    async def startup_collector(self):
        if not self.sitelink2gravitas:
            logger.info("exits", reseon="No pending tasks")
            return

        self.task_queue = asyncio.Queue()
        for item in self.sitelink2gravitas.items():
            self.task_queue.put_nowait(item)

        link2outdated = {}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(locale="en-US")
            await Malenia.apply_stealth(context)

            while not self.task_queue.empty():
                sitelink, gravitas = self.task_queue.get_nowait()
                await self._collete_datasets(context, sitelink)
                gs = self.all_right(gravitas)
                if not gs.done:
                    if sitelink not in link2outdated:
                        link2outdated[sitelink] = 1
                    else:
                        link2outdated[sitelink] += 1
                    if link2outdated[sitelink] < 3:
                        self.task_queue.put_nowait((sitelink, gravitas))

            await context.close()

    def all_right(self, gravitas: Gravitas) -> GravitasState:
        cases_num = 0
        for root, _, _ in os.walk(self.tmp_dir):
            root_dir = Path(root)
            if gravitas.mixed_label != root_dir.name:
                continue
            cases_num = len(os.listdir(root))
            if "binary" in gravitas.request_type and cases_num > 300:
                return GravitasState(typed_dir=root_dir, done=True, cases_num=cases_num)
            if "area_select" in gravitas.request_type and cases_num > 50:
                return GravitasState(typed_dir=root_dir, done=True, cases_num=cases_num)
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
            gravitas.typed_dir = gs.typed_dir
            gravitas.cases_num = gs.cases_num
            gravitas.zip()
            asset = gravitas.to_asset(archive_release)
            create_comment(asset, gravitas, sign_label=gs.done)
            shutil.rmtree(gravitas.zip_path, ignore_errors=True)

    async def bytedance(self):
        self.prelude_tasks()
        await self.startup_collector()
        self.post_datasets()


if __name__ == "__main__":
    collector = Collector()
    asyncio.run(collector.bytedance())
