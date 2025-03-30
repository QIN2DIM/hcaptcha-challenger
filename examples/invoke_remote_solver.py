# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import base64
import random
from pathlib import Path
from typing import List

# pip install pandas tabulate
import pandas as pd
from httpx import Client

BASE_URL = "http://localhost:33777"
client = Client(base_url=BASE_URL, timeout=30)


def invoke_remove_tool(self_supervised_payload: dict):
    response = client.post("/challenge/image_label_binary", json=self_supervised_payload)
    response.raise_for_status()
    results = response.json()["results"]

    return results


def show_and_cache(image_paths: List[Path], results: List[str], prompt: str):
    output = [
        {"image": f"![]({image_path})", "result": result}
        for image_path, result in zip(image_paths, results)
    ]
    mdk = pd.DataFrame.from_records(output).to_markdown()
    mdk = f"- prompt: `{prompt}`\n\n{mdk}"
    print(mdk)

    fp = Path(f"results {prompt}.md")
    fp.write_text(mdk, encoding="utf8")
    print(f"\nsaved ->> {fp.resolve()}")


def run():
    images_dir = Path(__file__).parent.parent.joinpath("assets/image_label_binary/streetlamp")
    image_paths = list(images_dir.glob("*.jpeg"))
    if not image_paths:
        return
    random.shuffle(image_paths)
    image_paths = image_paths[:5]

    prompt = "streetlamp"
    challenge_images = [base64.b64encode(fp.read_bytes()).decode() for fp in image_paths]
    self_supervised_payload = {
        "prompt": prompt,
        "challenge_images": challenge_images,
        "positive_labels": ["streetlamp", "light"],
        "negative_labels": ["duck", "shark", "swan"],
    }

    results = invoke_remove_tool(self_supervised_payload)

    show_and_cache(image_paths, results, prompt)


if __name__ == "__main__":
    run()
