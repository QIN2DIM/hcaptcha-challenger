import math
import random
from typing import List, Tuple

def _generate_bezier_trajectory(
    start: Tuple[float, float], end: Tuple[float, float], steps: int
) -> List[Tuple[float, float]]:
    """
    Generates a quadratic bezier curve trajectory between start and end points.
    """
    points = []

    # Calculate distance between points
    distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

    # Create control point(s) for the bezier curve
    # For longer distances, we use a higher control point offset
    offset_factor = min(0.3, max(0.1, distance / 1000))

    # Random control point that's offset from the midpoint
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # Create slight randomness in the control point
    control_x = mid_x + random.uniform(-1, 1) * distance * offset_factor
    control_y = mid_y + random.uniform(-1, 1) * distance * offset_factor

    # Generate points along the bezier curve
    for i in range(steps + 1):
        t = i / steps
        # Quadratic bezier formula
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control_x + t**2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control_y + t**2 * end[1]
        points.append((x, y))

    return points


def _generate_dynamic_delays(steps: int, base_delay: int) -> List[float]:
    """
    Generates dynamic delays between mouse movements to simulate human-like acceleration/deceleration.
    """
    delays = []

    # Acceleration profile: slower at start and end, faster in the middle
    for i in range(steps + 1):
        progress = i / steps

        # Ease in-out function (slow start, fast middle, slow end)
        if progress < 0.5:
            factor = 2 * progress * progress  # Accelerate
        else:
            progress = progress - 1
            factor = 1 - (-2 * progress * progress)  # Decelerate

        # Adjust delay based on position in the curve (1.5x at ends, 0.6x in middle)
        delay_factor = 1.5 - 0.9 * factor

        # Add slight randomness to delays (±10%)
        random_factor = random.uniform(0.9, 1.1)

        delays.append(base_delay * delay_factor * random_factor)

    return delays
