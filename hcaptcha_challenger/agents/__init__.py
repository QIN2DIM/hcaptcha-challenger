# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:17
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from hcaptcha_challenger.agents.pipline.control import AgentR
from hcaptcha_challenger.agents.playwright.control import AgentT
from hcaptcha_challenger.agents.playwright.tarnished import Malenia, Tarnished

__all__ = ["AgentT", "Malenia", "Tarnished", "AgentR"]
