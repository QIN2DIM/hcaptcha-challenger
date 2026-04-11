import asyncio
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcaptcha_challenger.tools.internal.providers.openai_compat import OpenAICompatProvider


class VisionCheck(BaseModel):
    ok: bool
    seen_text: str


async def main() -> None:
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", os.getenv("CODEX_BASE_URL", "http://127.0.0.1:8787"))
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", os.getenv("CODEX_API_KEY", "relay-dev-token"))
    model = os.getenv("OPENAI_COMPAT_MODEL", os.getenv("CODEX_MODEL", "gpt-5.4"))
    timeout = int(os.getenv("OPENAI_COMPAT_TIMEOUT", os.getenv("CODEX_TIMEOUT", "180")))

    scratch = Path("tmp/.smoke")
    scratch.mkdir(parents=True, exist_ok=True)
    image_path = scratch / "vision-smoke.png"

    image = Image.new("RGB", (140, 70), "white")
    drawer = ImageDraw.Draw(image)
    drawer.text((10, 20), "OK", fill=(0, 0, 0))
    image.save(image_path)

    provider = OpenAICompatProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        extra_headers={},
    )

    result = await provider.generate_with_images(
        images=[image_path],
        response_schema=VisionCheck,
        user_prompt="Look at the image. Return seen_text exactly from the image and ok=true.",
        description="Return strict JSON only.",
    )
    print(result.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
