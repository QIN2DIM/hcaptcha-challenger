from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np


def mark_points_on_image(
    image_path: Union[str, Path],
    points: List[Tuple[int, int]],
    output_path: Optional[Union[str, Path]] = None,
    point_radius: int = 5,
    text_offset: Tuple[int, int] = (10, 5),
    text_scale: float = 0.5,
    text_thickness: int = 1,
) -> np.ndarray:
    """
    Mark coordinates on an image with colored points and coordinate labels

    Args:
        image_path: Path to the image file
        points: List of coordinates to mark, each point in (x, y) format
        output_path: Path to save the output image, if None, image won't be saved
        point_radius: Radius of the circle marker
        text_offset: Offset (x, y) for the text label from the point
        text_scale: Scale of the text label
        text_thickness: Thickness of the text

    Returns:
        Image array with marks and labels
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Draw a point and label at each coordinate
    for i, (x, y) in enumerate(points):
        # Generate a unique color for each point using HSV color space
        # This ensures good color distribution and visibility
        hue = (i * 30) % 180  # Rotate through hues, avoiding similar adjacent colors
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()

        # Draw a filled circle at the point
        cv2.circle(image, (x, y), point_radius, color, -1)  # -1 means filled

        # Add coordinate text
        label = f"({x}, {y})"
        cv2.putText(
            image,
            label,
            (x + text_offset[0], y + text_offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    # Save the image if output path is specified
    if output_path:
        cv2.imwrite(str(output_path), image)

    return image
