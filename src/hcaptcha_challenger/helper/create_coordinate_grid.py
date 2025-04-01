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


def create_coordinate_grid(
    image: Union[str, np.ndarray, Path],
    bbox: Union[FloatRect, Tuple[float, float, float, float], List[float]],
) -> np.ndarray:
    """
    Convert a web image to a scientific-style coordinate system image.

    Args:
        image: Input image (path or numpy array)
        bbox: Bounding box (x, y, width, height) of the image in the webpage

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
    x_ticks = np.linspace(x, x + width, 11)
    y_ticks = np.linspace(y, y + height, 11)

    # Set ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

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
