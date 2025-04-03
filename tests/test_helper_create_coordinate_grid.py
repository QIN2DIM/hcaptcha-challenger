import itertools
import logging
import os
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from hcaptcha_challenger.helper.create_coordinate_grid import create_coordinate_grid

# --- 配置常量 ---
BASE_PATH = Path("challenge_view")
DATASET_IMAGE_DRAG_DROP = BASE_PATH / "image_drag_drop"
DATASET_IMAGE_LABEL_AREA_SELECT = BASE_PATH / "image_label_area_select"
# 确保目录存在 (可选，但良好实践)
DATASET_IMAGE_DRAG_DROP.mkdir(parents=True, exist_ok=True)
DATASET_IMAGE_LABEL_AREA_SELECT.mkdir(parents=True, exist_ok=True)

# 固定的 Bounding Box
DEFAULT_BBOX: Dict[str, int] = {"x": 0, "y": 0, "width": 501, "height": 431}

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PREFIX_ = "coordinate_grid"
PIL_AVAILABLE = True


def process_and_save_grid(challenge_screenshot: Path, bbox: Dict[str, int]):
    """
    为单个图像创建坐标网格并保存结果。

    Args:
        challenge_screenshot: 输入图像文件的路径。
        bbox: 用于 create_coordinate_grid 的边界框。
    """
    try:
        grid_divisions_path = challenge_screenshot.parent / f'{PREFIX_}_{challenge_screenshot.name}'

        # 调用核心处理函数 (假设它返回一个适合保存的图像数据，例如 NumPy 数组)
        result_data = create_coordinate_grid(challenge_screenshot, bbox)

        if PIL_AVAILABLE and isinstance(result_data, np.ndarray):
            # 确保数据类型适合 Pillow 保存 (通常是 uint8)
            if result_data.dtype != np.uint8:
                # 如果是浮点数 (0-1范围)，转换为 uint8
                if result_data.max() <= 1.0 and result_data.min() >= 0.0:
                    result_data = (result_data * 255).astype(np.uint8)
                else:
                    # 其他情况可能需要更复杂的归一化或类型转换
                    logging.warning(
                        f"Result data for {challenge_screenshot.name} has dtype {result_data.dtype}. Attempting direct conversion to uint8."
                    )
                    result_data = result_data.astype(np.uint8)  # 可能不准确，取决于原始范围

            # 使用 Pillow 保存 NumPy 数组
            img = Image.fromarray(result_data)
            img.save(grid_divisions_path)
            logging.info(f"Saved grid image to: {grid_divisions_path} using Pillow")
        else:
            logging.error(
                f"Processing failed for {challenge_screenshot.name}: create_coordinate_grid did not return a NumPy array."
            )

    except FileNotFoundError:
        logging.error(f"Input image not found: {challenge_screenshot}")
    except Exception as e:
        logging.error(
            f"Failed to process {challenge_screenshot.name}: {e}", exc_info=True
        )  # exc_info=True 记录堆栈跟踪


def test_create_coordinate_grid_parallel(max_workers: int = None):
    """
    查找所有挑战截图，并行处理并保存坐标网格图像。

    Args:
        max_workers: 使用的最大进程数。默认为系统 CPU 核心数。
    """
    if not PIL_AVAILABLE:
        logging.warning(
            "Consider installing Pillow (`pip install Pillow`) for potentially better performance and reduced dependencies."
        )

    if max_workers is None:
        max_workers = os.cpu_count()
        logging.info(f"Using default max_workers: {max_workers}")

    # 获取所有图像路径列表 (并行处理通常需要先收集所有任务)
    all_image_paths = list(
        itertools.chain(
            DATASET_IMAGE_DRAG_DROP.rglob("*.png"), DATASET_IMAGE_LABEL_AREA_SELECT.rglob("*.png")
        )
    )

    valid_image_paths = [
        p for p in all_image_paths if p.is_file() and not p.name.startswith(PREFIX_)
    ]
    logging.info(f"Found {len(valid_image_paths)} image files to process.")

    if not valid_image_paths:
        logging.warning("No valid image files found.")
        return

    processed_count = 0
    # 使用 ProcessPoolExecutor 进行 CPU 密集型任务并行化
    # 注意：process_and_save_grid 函数必须可以在子进程中被 pickle
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_and_save_grid, img_path, DEFAULT_BBOX): img_path
            for img_path in valid_image_paths
        }

        # 处理完成的任务
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                # 如果 process_and_save_grid 有返回值，可以在这里获取 future.result()
                future.result()  # 检查任务是否抛出异常
                processed_count += 1
                # 可以在这里添加更详细的成功日志，但 process_and_save_grid 内部已有日志
            except Exception as e:
                # 异常已经在 process_and_save_grid 中记录，这里可以再记录一次或采取其他措施
                logging.error(f"Error processing {img_path} in parallel worker: {e}")

    logging.info(
        f"Finished parallel processing. Processed {processed_count}/{len(valid_image_paths)} images."
    )


# --- 主程序入口 ---
if __name__ == "__main__":
    # Windows/macOS 下使用 multiprocessing 需要保护入口点
    test_create_coordinate_grid_parallel()
    # 或者指定 worker 数量: test_create_coordinate_grid_parallel(max_workers=4)
