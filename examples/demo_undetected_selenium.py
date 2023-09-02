# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description: No longer maintained
from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path

from loguru import logger
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

import hcaptcha_challenger as solver
from hcaptcha_challenger.agents.exceptions import ChallengePassed
from hcaptcha_challenger.agents.selenium import ArmorUtils
from hcaptcha_challenger.agents.selenium import SeleniumAgent
from hcaptcha_challenger.agents.selenium import get_challenge_ctx


# Existing user data
email = "plms-123@tesla.com"
country = "Hong Kong"

# Init local-side of the ModelHub
solver.install()
headless = False

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")


@logger.catch
def hit_challenge(ctx, agent: SeleniumAgent, retries: int = 2) -> bool | None:
    """
    Use `anti_checkbox()` `anti_hcaptcha()` to be flexible to challenges
    :param ctx:
    :param agent:
    :param retries:
    :return:
    """
    if ArmorUtils.checkbox_visible(ctx):
        agent.anti_checkbox(ctx)

    for pth in range(1, retries):
        with suppress(ChallengePassed):
            result = agent.anti_hcaptcha(ctx)
            print(f">> {pth} - Challenge Result: {result}")
            if result == agent.status.CHALLENGE_BACKCALL:
                time.sleep(0.5)
                ctx.find_element(By.XPATH, "//div[@class='refresh button']").click()
                continue
            if result == agent.status.CHALLENGE_RETRY:
                continue
            if result == agent.status.CHALLENGE_SUCCESS:
                # Not related action
                WebDriverWait(ctx, 30).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Submit']"))
                ).click()
                return True

    ctx.switch_to.default_content()
    return True


def bytedance():
    # New Challenger
    challenger = SeleniumAgent.from_modelhub(tmp_dir=tmp_dir)

    # Replace selenium.webdriver.Chrome with CTX
    ctx = get_challenge_ctx(silence=headless)
    ctx.get("https://dashboard.hcaptcha.com/signup")
    try:
        WebDriverWait(ctx, 15, ignored_exceptions=(ElementNotInteractableException,)).until(
            EC.presence_of_element_located((By.ID, "email"))
        ).send_keys(email)

        WebDriverWait(ctx, 15, ignored_exceptions=(ElementNotInteractableException,)).until(
            EC.presence_of_element_located((By.ID, "country"))
        ).send_keys(country)

        # Handling context validation
        hit_challenge(ctx=ctx, agent=challenger)
    except Exception as err:
        logger.exception(err)
    finally:
        sp = tmp_dir.joinpath(f"bytedance{' - headless' if headless else ''}.png")
        ctx.save_screenshot(sp)
        time.sleep(3)
        ctx.quit()


# No longer maintained
if __name__ == "__main__":
    bytedance()
