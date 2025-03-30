#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebM 转 MP4 转换工具

这个脚本可以将 .webm 视频文件转换为 .mp4 格式。
支持单个文件转换或批量转换整个目录中的所有 .webm 文件。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

from loguru import logger


def convert_webm_to_mp4(input_file: str, output_file: Optional[str] = None) -> bool:
    """
    将单个 WebM 文件转换为 MP4 格式

    Args:
        input_file: 输入的 WebM 文件路径
        output_file: 输出的 MP4 文件路径，如果不指定则自动生成

    Returns:
        bool: 转换是否成功
    """
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return False

    if not output_file:
        # 如果没有指定输出文件，则使用相同的文件名但扩展名改为 .mp4
        output_file = str(Path(input_file).with_suffix('.mp4'))

    try:
        logger.info(f"正在转换: {input_file} -> {output_file}")

        # 使用 ffmpeg 进行转换
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
            logger.error(f"转换失败: {result.stderr}")
            return False

        logger.success(f"转换成功: {output_file}")
        return True

    except Exception as e:
        logger.exception(f"转换过程中发生错误: {str(e)}")
        return False


def batch_convert(input_dir: str, output_dir: Optional[str] = None) -> None:
    """
    批量转换目录中的所有 WebM 文件

    Args:
        input_dir: 包含 WebM 文件的输入目录
        output_dir: 输出 MP4 文件的目录，如果不指定则使用输入目录
    """
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    # 获取所有 .webm 文件
    webm_files = list(Path(input_dir).glob("**/*.webm"))

    if not webm_files:
        logger.warning(f"在 {input_dir} 中没有找到 .webm 文件")
        return

    logger.info(f"找到 {len(webm_files)} 个 .webm 文件")

    success_count = 0
    for webm_file in webm_files:
        if output_dir:
            # 计算相对路径，保持目录结构
            rel_path = webm_file.relative_to(input_dir)
            output_file = Path(output_dir) / rel_path.with_suffix('.mp4')

            # 确保输出目录存在
            os.makedirs(output_file.parent, exist_ok=True)

            if convert_webm_to_mp4(str(webm_file), str(output_file)):
                success_count += 1
        else:
            if convert_webm_to_mp4(str(webm_file)):
                success_count += 1

    logger.info(f"转换完成: {success_count}/{len(webm_files)} 个文件成功转换")


def check_ffmpeg() -> bool:
    """检查系统是否安装了 ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将 WebM 视频转换为 MP4 格式')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='要转换的单个 WebM 文件')
    group.add_argument('-d', '--directory', help='包含 WebM 文件的目录（将批量转换）')

    parser.add_argument('-o', '--output', help='输出文件或目录（可选）')

    args = parser.parse_args()

    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    # 检查 ffmpeg 是否已安装
    if not check_ffmpeg():
        logger.error("未找到 ffmpeg。请先安装 ffmpeg 后再运行此脚本。")
        logger.info("安装指南: https://ffmpeg.org/download.html")
        return

    if args.file:
        # 单文件转换
        convert_webm_to_mp4(args.file, args.output)
    else:
        # 批量转换
        batch_convert(args.directory, args.output)


def invoke(
    input_path: str, output_path: Optional[str] = None, is_directory: bool = False
) -> List[dict]:
    """
    以编程方式调用转换功能

    Args:
        input_path: 输入的WebM文件路径或包含WebM文件的目录路径
        output_path: 输出的MP4文件路径或目录，如果不指定则自动生成
        is_directory: 是否批量处理目录

    Returns:
        List[dict]: 转换结果列表，每个字典包含输入文件、输出文件和转换状态
    """
    # 检查 ffmpeg 是否已安装
    if not check_ffmpeg():
        raise RuntimeError("未找到 ffmpeg。请先安装 ffmpeg 后再运行此函数。")

    results = []

    if is_directory:
        # 批量转换目录
        if not os.path.isdir(input_path):
            raise ValueError(f"输入目录不存在: {input_path}")

        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)

        # 获取所有 .webm 文件
        webm_files = list(Path(input_path).glob("**/*.webm"))

        for webm_file in webm_files:
            if output_path:
                # 计算相对路径，保持目录结构
                rel_path = webm_file.relative_to(input_path)
                output_file = Path(output_path) / rel_path.with_suffix('.mp4')

                # 确保输出目录存在
                os.makedirs(output_file.parent, exist_ok=True)

                success = convert_webm_to_mp4(str(webm_file), str(output_file))
            else:
                output_file = webm_file.with_suffix('.mp4')
                success = convert_webm_to_mp4(str(webm_file))

            results.append(
                {"input_file": str(webm_file), "output_file": str(output_file), "success": success}
            )
    else:
        # 单文件转换
        if not os.path.exists(input_path):
            raise ValueError(f"输入文件不存在: {input_path}")

        success = convert_webm_to_mp4(input_path, output_path)

        if not output_path:
            output_path = str(Path(input_path).with_suffix('.mp4'))

        results.append({"input_file": input_path, "output_file": output_path, "success": success})

    return results


if __name__ == "__main__":
    main()
