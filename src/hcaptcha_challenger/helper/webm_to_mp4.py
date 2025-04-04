"""
WebM to MP4 conversion tool

This script can convert .webm video files to .mp4 format.
Supports single file conversion or batch conversion of all .webm files in the entire directory.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List

from loguru import logger


def convert_webm_to_mp4(input_file: str, output_file: Optional[str] = None) -> bool:
    """
    Convert a single WebM file to MP4 format

    Args:
        input_file: The input WebM file path
        output_file: The output MP4 file path, automatically generated if not specified

    Returns:
        bool: Is the conversion successful?
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return False

    if not output_file:
        # If no output file is specified, the same file name is used but the extension is changed to .mp4
        output_file = str(Path(input_file).with_suffix('.mp4'))

    try:
        logger.info(f"Converting: {input_file} -> {output_file}")

        # Use ffmpeg for conversion
        cmd = [
            'ffmpeg',
            '-i',
            input_file,  # 输入文件
            '-c:v',
            'libx264',  # 视频编码器
            '-c:a',
            'aac',  # 音频编码器
            '-strict',
            'experimental',
            '-b:a',
            '192k',  # 音频比特率
            '-y',  # 覆盖输出文件（如果存在）
            output_file,
        ]

        # 执行命令
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return False

        logger.success(f"Conversion successfully: {output_file}")
        return True

    except Exception as e:
        logger.exception(f"An error occurred during the conversion process: {str(e)}")
        return False


def batch_convert(input_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Batch convert all WebM files in the directory

    Args:
        input_dir: Input directory containing WebM files
        output_dir: The directory that outputs the MP4 file, and if not specified, the input directory is used.
    """
    if not os.path.isdir(input_dir):
        logger.error(f"The input directory does not exist: {input_dir}")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Create output directory: {output_dir}")

    # Get all .webm files
    webm_files = list(Path(input_dir).glob("**/*.webm"))

    if not webm_files:
        logger.warning(f"No .webm file was found in {input_dir}")
        return

    logger.info(f"Found {len(webm_files)} .webm files")

    success_count = 0
    for webm_file in webm_files:
        if output_dir:
            # Calculate relative paths and maintain directory structure
            rel_path = webm_file.relative_to(input_dir)
            output_file = Path(output_dir) / rel_path.with_suffix('.mp4')

            # Make sure the output directory exists
            os.makedirs(output_file.parent, exist_ok=True)

            if convert_webm_to_mp4(str(webm_file), str(output_file)):
                success_count += 1
        else:
            if convert_webm_to_mp4(str(webm_file)):
                success_count += 1

    logger.info(
        f"Conversion completed: {success_count}/{len(webm_files)} files were successfully converted"
    )


def check_ffmpeg() -> bool:
    """Check if the system is installed ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def invoke(
    input_path: str, output_path: Optional[str] = None, is_directory: bool = False
) -> List[dict]:
    """
    Call the conversion function programmatically

        Args:
            input_path: The input WebM file path or the directory path containing the WebM file
            output_path: The output MP4 file path or directory, automatically generated if not specified
            is_directory: Whether to batch process the directory

        Returns:
            List[dict]: A list of conversion results, each dictionary contains input file, output file, and conversion status
    """
    # Check if ffmpeg is installed
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found. Please install ffmpeg before running this function.")

    results = []

    if is_directory:
        # Batch conversion directory
        if not os.path.isdir(input_path):
            raise ValueError(f"Input directory does not exist: {input_path}")

        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)

        # Get all .webm files
        webm_files = list(Path(input_path).glob("**/*.webm"))

        for webm_file in webm_files:
            if output_path:
                # Calculate relative paths and maintain directory structure
                rel_path = webm_file.relative_to(input_path)
                output_file = Path(output_path) / rel_path.with_suffix('.mp4')

                # Make sure the output directory exists
                os.makedirs(output_file.parent, exist_ok=True)

                success = convert_webm_to_mp4(str(webm_file), str(output_file))
            else:
                output_file = webm_file.with_suffix('.mp4')
                success = convert_webm_to_mp4(str(webm_file))

            results.append(
                {"input_file": str(webm_file), "output_file": str(output_file), "success": success}
            )
    else:
        # Single file conversion
        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")

        success = convert_webm_to_mp4(input_path, output_path)

        if not output_path:
            output_path = str(Path(input_path).with_suffix('.mp4'))

        results.append({"input_file": input_path, "output_file": output_path, "success": success})

    return results
