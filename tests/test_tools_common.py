import os
from pathlib import Path

import dotenv
from google import genai

dotenv.load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_models = [
    "gemini-2.0-flash",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-lite",
]


def test_count_tokens():
    har_path = Path("har/hcaptcha.html")
    if har_path.is_file():
        contents = har_path.read_text(encoding="utf8")
        for model in gemini_models:
            response = client.models.count_tokens(model=model, contents=contents)
            print(model, response)
