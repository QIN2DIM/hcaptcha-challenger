import json
import os
from pathlib import Path
from uuid import uuid4

import dotenv
from google import genai
from google.genai import types

dotenv.load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_models = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-lite",
]

output_dir = Path("generate_content")
output_dir.mkdir(parents=True, exist_ok=True)


def test_count_tokens():
    har_path = Path("har/hcaptcha.html")
    if har_path.is_file():
        contents = har_path.read_text(encoding="utf8")
        for model in gemini_models:
            response = client.models.count_tokens(model=model, contents=contents)
            print(model, response)


def test_count_prompts_tokens():
    from hcaptcha_challenger.tools import (
        challenge_classifier,
        image_classifier,
        spatial_point_reasoning,
        spatial_path_reasoning,
    )

    model = gemini_models[0]

    contents = [
        challenge_classifier.CHALLENGE_CLASSIFIER_INSTRUCTIONS + challenge_classifier.USER_PROMPT,
        image_classifier.SYSTEM_INSTRUCTION + image_classifier.USER_PROMPT,
        spatial_point_reasoning.THINKING_PROMPT,
        spatial_path_reasoning.THINKING_PROMPT,
    ]
    for text in contents:
        response = client.models.count_tokens(model=model, contents=text)
        print(response)


def test_list_models():
    cf = genai.Client(api_key=os.getenv("GEMINI_API_KEY_FREE"))
    pf = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    print("--- FREE PLAN ---")
    for model in cf.models.list():
        if "gemini-2." in model.name:
            print(model)

    print("--- PAID PLAN ---")
    for model in pf.models.list():
        if "gemini-2." in model.name:
            print(model)
