# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import time

from selenium.common.exceptions import (
    ElementNotInteractableException,
    ElementClickInterceptedException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

import hcaptcha_challenger as solver
from hcaptcha_challenger import HolyChallenger
from hcaptcha_challenger.agents.exceptions import ChallengePassed

# Existing user data
email = "plms-123@tesla.com"
country = "Hong Kong"
headless = False

# Init local-side of the ModelHub
solver.install()


def hit_challenge(ctx, challenger: HolyChallenger, retries: int = 10) -> str | None:
    """
    Use `anti_checkbox()` `anti_hcaptcha()` to be flexible to challenges
    :param ctx:
    :param challenger:
    :param retries:
    :return:
    """
    if challenger.utils.face_the_checkbox(ctx):
        challenger.anti_checkbox(ctx)
        if res := challenger.utils.get_hcaptcha_response(ctx):
            return res

    for _ in range(retries):
        try:
            if (resp := challenger.anti_hcaptcha(ctx)) is None:
                continue
            if resp == challenger.CHALLENGE_SUCCESS:
                return challenger.utils.get_hcaptcha_response(ctx)
        except ChallengePassed:
            return challenger.utils.get_hcaptcha_response(ctx)
        challenger.utils.refresh(ctx)
        time.sleep(1)


def bytedance():
    # New Challenger
    challenger = solver.new_challenger(screenshot=True, debug=True)

    # Replace selenium.webdriver.Chrome with CTX
    ctx = solver.get_challenge_ctx(silence=headless)
    ctx.get("https://dashboard.hcaptcha.com/signup")
    try:
        # Populate test data
        WebDriverWait(ctx, 15, ignored_exceptions=(ElementNotInteractableException,)).until(
            EC.presence_of_element_located((By.ID, "email"))
        ).send_keys(email)
        WebDriverWait(ctx, 15, ignored_exceptions=(ElementNotInteractableException,)).until(
            EC.presence_of_element_located((By.ID, "country"))
        ).send_keys(country)
        # Handling context validation
        hit_challenge(ctx=ctx, challenger=challenger)
        # Submit test data
        WebDriverWait(ctx, 5, ignored_exceptions=(ElementClickInterceptedException,)).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-cy]"))
        ).click()

        ctx.save_screenshot(f"datas/bytedance{' - headless' if headless else ''}.png")
    finally:
        ctx.quit()


if __name__ == "__main__":
    bytedance()
