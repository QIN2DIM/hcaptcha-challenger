from typing import Tuple

import cv2
import numpy as np


def create_grid_reference(
    image_size: Tuple[int, int],
    bounding_box: Tuple[Tuple[int, int], Tuple[int, int]],
    grid_divisions: int,
    line_color: Tuple[int, int, int] = (0, 255, 0),  # Green in BGR
    line_thickness: int = 1,
    background_color: Tuple[int, int, int] = (0, 0, 0),  # Black background
    alpha: float = 0.5,  # Transparency level for overlay
) -> np.ndarray:
    """
    Create a grid reference layer within a specified bounding box

    Args:
        image_size: Size of the image (width, height)
        bounding_box: Tuple of two points ((x1, y1), (x2, y2)) defining the bounding box
        grid_divisions: Number of divisions for the grid (0 = no grid, 1 = 4 cells, 2 = 9 cells, etc.)
        line_color: Color of the grid lines in BGR format
        line_thickness: Thickness of the grid lines
        background_color: Background color of the grid layer
        alpha: Transparency level for the grid layer (0.0 to 1.0)

    Returns:
        Grid reference layer as a numpy array
    """
    # Extract bounding box coordinates
    (x1, y1), (x2, y2) = bounding_box

    # Ensure x1 < x2 and y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Create a transparent layer for the grid
    grid_layer = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)

    # Draw the bounding box rectangle
    cv2.rectangle(grid_layer, (x1, y1), (x2, y2), line_color, line_thickness)

    # If grid_divisions > 0, draw the grid lines
    if grid_divisions > 0:
        # Calculate the step size for grid lines
        x_step = (x2 - x1) / (grid_divisions + 1)
        y_step = (y2 - y1) / (grid_divisions + 1)

        # Draw vertical grid lines
        for i in range(1, grid_divisions + 1):
            x = int(x1 + i * x_step)
            cv2.line(grid_layer, (x, y1), (x, y2), line_color, line_thickness)

        # Draw horizontal grid lines
        for i in range(1, grid_divisions + 1):
            y = int(y1 + i * y_step)
            cv2.line(grid_layer, (x1, y), (x2, y), line_color, line_thickness)

    return grid_layer


def overlay_grid_on_image(
    image: np.ndarray,
    bounding_box: Tuple[Tuple[int, int], Tuple[int, int]],
    grid_divisions: int,
    line_color: Tuple[int, int, int] = (255, 0, 0),
    line_thickness: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a grid reference on an existing image

    Args:
        image: Input image as numpy array
        bounding_box: Tuple of two points ((x1, y1), (x2, y2)) defining the bounding box
        grid_divisions: Number of divisions for the grid
        line_color: Color of the grid lines in BGR format
        line_thickness: Thickness of the grid lines
        alpha: Transparency level for the grid overlay

    Returns:
        Image with grid overlay
    """
    # Create grid layer
    grid_layer = create_grid_reference(
        (image.shape[1], image.shape[0]),
        bounding_box,
        grid_divisions,
        line_color,
        line_thickness,
        (0, 0, 0),
        alpha,
    )

    # Create a mask where grid lines are drawn
    mask = np.any(grid_layer != [0, 0, 0], axis=2)

    # Create output image
    result = image.copy()

    # Apply grid lines to the image
    result[mask] = cv2.addWeighted(image, 1 - alpha, grid_layer, alpha, 0)[mask]

    return result
