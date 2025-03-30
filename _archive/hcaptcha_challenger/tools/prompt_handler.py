# -*- coding: utf-8 -*-
# Time       : 2023/8/19 18:04
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from hcaptcha_challenger.constant import BAD_CODE, INV


def regularize_prompt_message(prompt_message: str) -> str:
    """Detach label from challenge prompt"""
    prompt_message = prompt_message.lower()
    if prompt_message.endswith("."):
        prompt_message = prompt_message[:-1]
    prompt_message = prompt_message.strip()
    return prompt_message


def label_cleaning(raw_label: str) -> str:
    """cleaning errors-unicode"""
    clean_label = raw_label
    for c in BAD_CODE:
        clean_label = clean_label.replace(c, BAD_CODE[c])
    return clean_label


def diagnose_task(words: str) -> str:
    """from challenge label to focus model name"""
    if not words or not isinstance(words, str) or len(words) < 2:
        raise TypeError(f"({words})TASK should be string type data")

    # Filename contains illegal characters
    if s := set(words) & INV:
        raise TypeError(f"({words})TASK contains invalid characters({s})")

    # Normalized separator
    rnv = {" ", ",", "-"}
    for s in rnv:
        words = words.replace(s, "_")

    for code, right_code in BAD_CODE.items():
        words.replace(code, right_code)

    words = words.strip()

    return words


def prompt2task(prompt: str) -> str:
    prompt = regularize_prompt_message(prompt)
    prompt = label_cleaning(prompt)
    prompt = diagnose_task(prompt)
    return prompt


def handle(x):
    return regularize_prompt_message(label_cleaning(x))
