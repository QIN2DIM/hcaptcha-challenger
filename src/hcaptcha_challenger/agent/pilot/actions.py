import asyncio
import random
from typing import Tuple, Optional
from playwright.async_api import Page, Locator
from hcaptcha_challenger.agent.utils import _generate_bezier_trajectory, _generate_dynamic_delays

class PilotActions:
    def __init__(self, page: Page, arm):
        self.page = page
        self.arm = arm

    async def click_by_mouse(self, locator: Optional[Locator] = None, coords: Optional[Tuple[float, float]] = None):
        if locator:
            bbox = await locator.bounding_box()
            if not bbox: return
            x = bbox['x'] + bbox['width'] / 2
            y = bbox['y'] + bbox['height'] / 2
        else:
            x, y = coords

        # Jitter sutil para humanização (Premium Parity)
        jx = x + random.uniform(-2, 2)
        jy = y + random.uniform(-2, 2)
        
        await self.page.mouse.move(jx, jy, steps=random.randint(5, 10))
        await self.page.mouse.click(jx, jy, delay=random.randint(150, 250))

    async def perform_drag_drop(self, path, delay_ms: int = 15, steps: int = 25):
        # Validação de integridade do objeto path (Portabilidade Final)
        if not hasattr(path, 'start_point') or not hasattr(path, 'end_point'):
            raise ValueError("O objeto Path deve ter os atributos start_point e end_point")
            
        start_x, start_y = path.start_point.x, path.start_point.y
        end_x, end_y = path.end_point.x, path.end_point.y
        
        # Move to the starting position
        await self.page.mouse.move(start_x, start_y)

        # Small random delay before pressing down (human reaction time)
        await asyncio.sleep(random.uniform(0.05, 0.15))

        # Press the mouse button down
        await self.page.mouse.down()
        
        # Soul Alignment: Official Bezier trajectory (Portado da linha 553 do original)
        points = _generate_bezier_trajectory((start_x, start_y), (end_x, end_y), steps)
        # Add velocity variation (slow start, fast middle, slow end)
        delays = _generate_dynamic_delays(steps, base_delay=delay_ms)

        # Perform the drag with human-like movement
        for i, ((current_x, current_y), delay) in enumerate(zip(points, delays)):
            # Add slight "noise" to the path (more pronounced near the end)
            if i > steps * 0.7:  # In the last 30% of the movement
                # More micro-adjustments near the end
                noise_factor = 0.5 if i > steps * 0.9 else 0.2
                current_x += random.uniform(-noise_factor, noise_factor)
                current_y += random.uniform(-noise_factor, noise_factor)
            
            await self.page.mouse.move(current_x, current_y)
            await asyncio.sleep(delay / 1000)

        # Ensure we end exactly at the target position
        await self.page.mouse.move(end_x, end_y)
        
        # Small pause before releasing (human precision adjustment)
        await asyncio.sleep(random.uniform(0.05, 0.1))
        
        # Release the mouse button at the destination
        await self.page.mouse.up()
        
        # Small pause between drag operations
        await asyncio.sleep(random.uniform(0.08, 0.12))

    async def click_checkbox(self):
        """Localiza e clica no checkbox inicial (Portado do baseline)."""
        checkbox_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"
        checkbox_frame = self.page.frame_locator(checkbox_selector)
        checkbox_element = checkbox_frame.locator("//div[@id='checkbox']")
        await self.click_by_mouse(checkbox_element)
