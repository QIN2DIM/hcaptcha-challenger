# -*- coding: utf-8 -*-
# Time       : 2023/9/12 14:45
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import re
from pathlib import Path

import pytest

from hcaptcha_challenger import split_prompt_message, label_cleaning

pattern = re.compile(r"[^\x00-\x7F]")

prompts = []

qa_data_path = Path(__file__).parent.joinpath("qa_data.txt")
if qa_data_path.exists():
    prompts = qa_data_path.read_text(encoding="utf8").split("\n")


@pytest.mark.parametrize("prompt", prompts)
def test_split_prompt_message(prompt: str):
    result = split_prompt_message(prompt, lang="en")
    assert result != prompt


@pytest.mark.parametrize("prompt", prompts)
def test_is_illegal_chars(prompt: str):
    result = label_cleaning(prompt)
    assert not pattern.search(result)
