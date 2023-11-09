# -*- coding: utf-8 -*-
# Time       : 2023/8/19 18:04
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import re

BAD_CODE = {
    "а": "a",
    "е": "e",
    "e": "e",
    "i": "i",
    "і": "i",
    "ο": "o",
    "с": "c",
    "ԁ": "d",
    "ѕ": "s",
    "һ": "h",
    "у": "y",
    "р": "p",
    "ϳ": "j",
    "х": "x",
    "\u0405": "S",
    "\u0042": "B",
    "\u0052": "R",
    "\u0049": "I",
    "\u0043": "C",
    "\u004b": "K",
    "\u039a": "K",
    "\u0053": "S",
    "\u0421": "C",
    "\u006c": "l",
    "\u0399": "I",
    "\u0392": "B",
    "ー": "一",
    "土": "士",
}


def split_prompt_message(prompt_message: str, lang: str) -> str:
    """Detach label from challenge prompt"""
    if lang.startswith("zh"):
        if "中包含" in prompt_message or "上包含" in prompt_message:
            return re.split(r"击|(的每)", prompt_message)[2]
        if "的每" in prompt_message:
            return re.split(r"(包含)|(的每)", prompt_message)[3]
        if "包含" in prompt_message:
            return re.split(r"(包含)|(的图)", prompt_message)[3]
    elif lang.startswith("en"):
        prompt_message = prompt_message.replace(".", "").lower()
        if "containing" in prompt_message:
            th = re.split(r"containing", prompt_message)[-1][1:].strip()
            return th[2:].strip() if th.startswith("a") else th
        if prompt_message.startswith("select all") and "images" not in prompt_message:
            return prompt_message.split("select all")[-1].strip()
        if "select all" in prompt_message:
            return re.split(r"all (.*) images", prompt_message)[1].strip()
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
    inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
    if s := set(words) & inv:
        raise TypeError(f"({words})TASK contains invalid characters({s})")

    # Normalized separator
    rnv = {" ", ",", "-"}
    for s in rnv:
        words = words.replace(s, "_")

    for code, right_code in BAD_CODE.items():
        words.replace(code, right_code)

    words = words.strip()

    return words


def prompt2task(prompt: str, lang: str = "en") -> str:
    prompt = split_prompt_message(prompt, lang)
    prompt = label_cleaning(prompt)
    prompt = diagnose_task(prompt)
    return prompt


def handle(x):
    return split_prompt_message(label_cleaning(x), "en")
