# -*- coding: utf-8 -*-
# Time       : 2023/8/25 14:05
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Page, Response, FrameLocator
from playwright.sync_api import TimeoutError as NinjaTimeout

from hcaptcha_challenger.agents.playwright.binary_challenge import PlaywrightAgent, QuestionResp
from hcaptcha_challenger.onnx.yolo import YOLO_CLASSES


@dataclass
class OnClickAgent(PlaywrightAgent):
    def handler(self, response: Response):
        if response.url.startswith("https://hcaptcha.com/getcaptcha/"):
            with suppress(Exception):
                self.question_resp = QuestionResp.from_response(response)

    def challenge(self, frame_challenge: FrameLocator, detector, *args, **kwargs):
        locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
        locator.wait_for(state="visible")

        page: Page = kwargs.get("page")
        page.wait_for_timeout(1000)

        path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
        locator.screenshot(path=path, type="png")

        # {{< Please click on the X >}}
        res = detector(Path(path))
        for name, (center_x, center_y), _ in res:
            # Bypass unfocused objects
            if name not in self._label:
                continue
            # Bypass invalid area
            if center_y < 20 or center_y > 520 or center_x < 91 or center_x > 400:
                continue
            # Click canvas
            position = {"x": center_x, "y": center_y}
            locator.click(delay=500, position=position)
            # print(f">>[{kwargs.get('pth')}] onclick - {name} - {position=}")
            page.wait_for_timeout(500)
            break

        # {{< Verify >}}
        with suppress(NinjaTimeout):
            fl = frame_challenge.locator("//div[@class='button-submit button']")
            fl.click()

    def tactical_retreat(self, **kwargs) -> str | None:
        if any(c in self._label for c in YOLO_CLASSES):
            return
        return self.status.CHALLENGE_BACKCALL

    def anti_hcaptcha(
            self, page: Page, window: str = "login", recur_url=None, *args, **kwargs
    ) -> bool | str:
        # Switch to hCaptcha challenge iframe
        frame_challenge = self.switch_to_challenge_frame(page, window)

        # Reset state of the OnClick Challenger
        self.challenge_resp = None
        self.question_resp = None
        with page.expect_response("**getcaptcha**") as resp:
            self.question_resp = QuestionResp.from_response(resp.value)

        # Parse challenge prompt
        self.get_label(frame_challenge)

        # Bypass `image_label_binary` challenge
        if self.question_resp.request_type != "image_label_area_select":
            if self.question_resp.request_type == "image_label_binary":
                # return super().anti_hcaptcha(
                #     page=page, window=window, recur_url=recur_url, *args, **kwargs
                # )
                pass
            return self.status.CHALLENGE_BACKCALL

        # [Optional] Download challenge image as dataset
        # The input of the task should be a screenshot of challenge-view
        # rather than the challenge image itself
        self.download_images()

        # Bypass Low-mAP Detection tasks
        if result := self.tactical_retreat() in [self.status.CHALLENGE_BACKCALL]:
            return result

        # Load YOLOv8 model from local or remote repo
        detector = self.match_solution(select="yolo")

        # Execute the detection task for twice
        for pth in range(2):
            frame_challenge = self.switch_to_challenge_frame(page, window)
            self.challenge(frame_challenge, detector, pth=pth, page=page)

        # Check challenge result
        with suppress(TypeError):
            result, message = self.is_success(
                page, frame_challenge, window=window, hook_url=recur_url
            )
            if result in [
                self.status.CHALLENGE_SUCCESS,
                self.status.CHALLENGE_CRASH,
                self.status.CHALLENGE_RETRY,
            ]:
                return result
