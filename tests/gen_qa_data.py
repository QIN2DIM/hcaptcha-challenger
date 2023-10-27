# -*- coding: utf-8 -*-
# Time       : 2023/10/27 23:46
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: 合并工厂和挑战者的 record_json
import json
import os
import shutil
from pathlib import Path

output_dir = Path(__file__).parent.joinpath("record_json")
output_dir.mkdir(parents=True, exist_ok=True)

input_dirs = [os.environ[i] for i in os.environ if i.lower().startswith("input_dir_")]


def copy_from_factory():
    count = 0

    for idr in input_dirs:
        if not (input_dir := Path(idr)).exists():
            print(f"Directory does not exist - {input_dir=}")
            continue

        print(f"MIGRATE DIRECTORY - {input_dir=}")
        for jn in os.listdir(input_dir):
            jp = input_dir.joinpath(jn)
            if not jp.is_file() or not jp.name.endswith(".json") or len(jn) < 10:
                continue
            output_path = output_dir.joinpath(jn)
            if output_path.exists():
                continue
            shutil.copyfile(jp, output_path)
            count += 1
            print(f"Merge Files[{count}] --> {jn=}")


def merge_prompts():
    prompts = []

    if (prompts_json := Path(__file__).parent.joinpath("prompts.json")).exists():
        prompts = json.loads(prompts_json.read_text(encoding="utf8"))

    if (json_dir := output_dir).exists():
        for json_name in os.listdir(json_dir):
            jf = json_dir.joinpath(json_name)
            data = json.loads(jf.read_text(encoding="utf8"))
            prompt: str = data["requester_question"].get("en", "")
            prompts.append(prompt)

    if prompts:
        prompts = list(set(prompts))
        df = json.dumps(prompts, indent=2, ensure_ascii=True, sort_keys=True)
        prompts_json.write_text(df, encoding="utf8")

    print(f"Extract prompt - {len(prompts)}")


if __name__ == "__main__":
    copy_from_factory()
    merge_prompts()
