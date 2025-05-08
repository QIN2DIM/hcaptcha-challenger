import json
import os
from pathlib import Path

import dotenv
from google import genai
from google.genai import types
from google.genai.types import ThinkingConfig

dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

output_dir = Path(__file__).parent.joinpath("generate_content")
output_dir.mkdir(parents=True, exist_ok=True)


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


def test_generate_content_fast():
    cc = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = cc.models.generate_content(model="gemini-2.0-flash-lite", contents="Hello, world!")

    output_path = output_dir.joinpath("generate_content_fast.json")
    text = json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False)
    output_path.write_text(text, encoding="utf8")


def test_generate_content_reasoning():
    cc = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = cc.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents="Hello, world!",
        config=types.GenerateContentConfig(
            temperature=0,
            thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=100),
        ),
    )

    output_path = output_dir.joinpath("generate_content_non_thoughts.json")
    text = json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False)
    output_path.write_text(text, encoding="utf8")
