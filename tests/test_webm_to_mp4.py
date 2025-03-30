from pathlib import Path

from hcaptcha_challenger.helper import webm_to_mp4


def test_webm_to_mp4():
    record_dir = Path("record")
    if record_dir.is_dir():
        for wp in record_dir.rglob("*.webm"):
            webm_to_mp4.invoke(input_path=str(wp.resolve()))
