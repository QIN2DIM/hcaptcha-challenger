from pathlib import Path
from typing import Union, TypedDict, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class FloatRect(TypedDict):
    x: float
    y: float
    width: float
    height: float


def _create_adaptive_contrast_grid(
    image: np.ndarray,
    bbox: Union[FloatRect, Tuple[float, float, float, float], List[float]],
    **kwargs,
) -> np.ndarray:
    """
    Create coordinate grids with adaptive contrast colors

    Args:
        image: input image (numpy array)
        bbox: Bounding box of image in web page (x, y, width, height)
        **kwargs: Additional parameters

    Returns:
        Processed image with adaptive contrasting color coordinate grid
    """
    img = image.copy()

    if isinstance(bbox, dict):
        x, y = bbox['x'], bbox['y']
        width, height = bbox['width'], bbox['height']
    else:
        x, y, width, height = bbox

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray) / 255.0

    grid_color = 'black' if avg_brightness > 0.5 else 'white'

    cmap_name = 'hot' if avg_brightness < 0.5 else 'cool'

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img, extent=[x, x + width, y + height, y])

    ax.set_xlim(x, x + width)
    ax.set_ylim(y + height, y)

    ax.spines['left'].set_position(('data', x))
    ax.spines['bottom'].set_position(('data', y + height))

    for spine in ax.spines.values():
        spine.set_color(grid_color)

    ax.tick_params(axis='x', colors=grid_color)
    ax.tick_params(axis='y', colors=grid_color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x_line_space_num = kwargs.get("x_line_space_num", 11)
    y_line_space_num = kwargs.get("y_line_space_num", 20)
    x_ticks = np.linspace(x, x + width, x_line_space_num)
    y_ticks = np.linspace(y, y + height, y_line_space_num)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels([str(round(tick)) for tick in x_ticks], color=grid_color)
    ax.set_yticklabels([str(round(tick)) for tick in y_ticks], color=grid_color)

    ax.grid(True, color=grid_color, alpha=0.7, linestyle='-', linewidth=1.0)

    n_colors = x_line_space_num * y_line_space_num
    colors = plt.cm.get_cmap(cmap_name, n_colors)

    for i, x_val in enumerate(x_ticks[:-1]):
        for j, y_val in enumerate(y_ticks[:-1]):
            color_idx = i + j * (x_line_space_num - 1)
            cell_color = colors(color_idx / n_colors)
            ax.add_patch(
                plt.Rectangle(
                    (x_val, y_val),
                    x_ticks[i + 1] - x_val,
                    y_ticks[j + 1] - y_val,
                    fill=True,
                    alpha=0.15,
                    color=cell_color,
                    zorder=0,
                )
            )

    ax.set_xlabel('X Coordinate', color=grid_color)
    ax.set_ylabel('Y Coordinate', color=grid_color)

    ax.set_title('Adaptive Contrast Coordinate Grid', color=grid_color)

    plt.tight_layout()

    fig.canvas.draw()
    img_with_grid = np.array(fig.canvas.renderer.buffer_rgba())

    plt.close(fig)

    img_with_grid = cv2.cvtColor(img_with_grid, cv2.COLOR_RGBA2RGB)

    return img_with_grid


def create_coordinate_grid(
    image: Union[str, np.ndarray, Path],
    bbox: Union[FloatRect, Tuple[float, float, float, float], List[float]],
    **kwargs,
) -> np.ndarray:
    """
    Convert a web image to a scientific-style coordinate system image.

    Args:
        image: Input image (path or numpy array)
        bbox: Bounding box (x, y, width, height) of the image in the webpage
        **kwargs: Additional parameters including:
          - x_line_space_num: Number of vertical grid lines (default: 11)
          - y_line_space_num: Number of horizontal grid lines (default: 20)
          - adaptive_contrast: Whether to use adaptive contrast grid (default: False)

    Returns:
        Processed image with coordinate grid
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    # Check if adaptive contrast mode is enabled
    adaptive_contrast = kwargs.get("adaptive_contrast", False)
    if adaptive_contrast:
        return _create_adaptive_contrast_grid(img, bbox, **kwargs)

    # Extract bbox parameters
    if isinstance(bbox, dict):
        x, y = bbox['x'], bbox['y']
        width, height = bbox['width'], bbox['height']
    else:
        x, y, width, height = bbox

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(img, extent=[x, x + width, y + height, y])  # Note the y-axis inversion

    # Set axis limits
    ax.set_xlim(x, x + width)
    ax.set_ylim(y + height, y)  # Inverted y-axis to match image coordinates

    # Set origin at the top-left corner
    ax.spines['left'].set_position(('data', x))
    ax.spines['bottom'].set_position(('data', y + height))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create grid lines
    x_line_space_num = kwargs.get("x_line_space_num", 11)
    y_line_space_num = kwargs.get("y_line_space_num", 20)
    x_ticks = np.linspace(x, x + width, x_line_space_num)
    y_ticks = np.linspace(y, y + height, y_line_space_num)

    # Set ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Format tick labels as rounded integers
    ax.set_xticklabels([str(round(tick)) for tick in x_ticks])
    ax.set_yticklabels([str(round(tick)) for tick in y_ticks])

    # Add grid with semi-transparent purple lines
    ax.grid(True, color='gray', alpha=0.5, linestyle='-', linewidth=1.0)
    # ax.grid(True, color='gray', alpha=0.5, linestyle='--', linewidth=1.0)
    # ax.grid(True, color='lightgray', alpha=0.2, linestyle='-', linewidth=0.8)

    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Add title
    ax.set_title('Image with Coordinate Grid')

    # Tight layout
    plt.tight_layout()

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    img_with_grid = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the figure to free memory
    plt.close(fig)

    # Convert RGBA to RGB
    img_with_grid = cv2.cvtColor(img_with_grid, cv2.COLOR_RGBA2RGB)

    return img_with_grid
