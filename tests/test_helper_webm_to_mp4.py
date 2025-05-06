from pathlib import Path

from hcaptcha_challenger.helper import webm_to_mp4
from hcaptcha_challenger.helper.webm_to_mp4 import check_ffmpeg
from loguru import logger


def test_webm_to_mp4():
    record_dir = Path(__file__).parent.joinpath("record")
    if not check_ffmpeg():
        logger.critical("ffmpeg not found. Please install ffmpeg before running this function.")
        return

    if record_dir.is_dir():
        for wp in record_dir.rglob("*.webm"):
            webm_to_mp4.invoke(input_path=str(wp.resolve()))
