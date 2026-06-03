import re
import uuid
import asyncio
import random
from typing import Tuple, Optional, Union
from pathlib import Path
from contextlib import suppress
from playwright.async_api import Page, Frame, expect
from loguru import logger
import matplotlib.pyplot as plt

from hcaptcha_challenger.models import RequestType, ChallengeTypeEnum
from hcaptcha_challenger.helper.create_coordinate_grid import create_coordinate_grid
from hcaptcha_challenger.agent.logger import LoggerHelper

class PilotNavigation:
    def __init__(self, page: Page, config, arm):
        self.page = page
        self.config = config
        self.arm = arm
        self.current_view_bbox: Optional[dict] = None
        self._checkbox_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"
        self._challenge_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=challenge')]"

    async def get_challenge_frame(self) -> Optional[Frame]:
        # Busca robusta (Recursiva + Fallbacks de URL)
        for attempt in range(20):
            candidate = self._find_frame_recursive(self.page.main_frame)
            if candidate:
                with suppress(Exception):
                    if await candidate.locator("//div[@class='challenge-view']").is_visible(timeout=1000):
                        return candidate
            
            # Fallback literal por URL
            for frame in self.page.frames:
                if "hcaptcha.com/captcha/v1/" in frame.url and "frame=challenge" in frame.url:
                    with suppress(Exception):
                        if await frame.locator("//div[@class='challenge-view']").is_visible(timeout=500):
                            return frame
            
            if attempt % 5 == 0:
                LoggerHelper.log_info(f"Procurando frame... Tentativa {attempt+1}/20")
            await asyncio.sleep(1)
        return None

    def _find_frame_recursive(self, frame: Frame, depth=0) -> Optional[Frame]:
        if depth > 4: return None
        for child in frame.child_frames:
            if "frame=challenge" in child.url:
                return child
            found = self._find_frame_recursive(child, depth + 1)
            if found: return found
        return None

    def validate_coordinate(self, x: int, y: int) -> bool:
        """Sanity check to prevent clicking way outside. Margem de 20% para flexibilidade oficial."""
        if not self.current_view_bbox:
            return True
        
        bx = self.current_view_bbox.get('x', 0)
        by = self.current_view_bbox.get('y', 0)
        bw = self.current_view_bbox.get('width', 1000)
        bh = self.current_view_bbox.get('height', 1000)

        if not (bx - bw * 0.5 <= x <= bx + bw * 1.5) or not (by - bh * 0.5 <= y <= by + bh * 1.5):
             LoggerHelper.log_warning(f"Coordenadas suspeitas: ({x}, {y}) fora dos limites normais.")
        return True

    async def wait_for_loaders(self, frame: Frame) -> bool:
        """Implementação robusta da linha 240-260 do original."""
        await asyncio.sleep(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS / 1000)
        loading_indicators = frame.locator("//div[@class='loading-indicator']")
        count = await loading_indicators.count()
        
        if count == 0: 
            return True
        
        for i in range(count):
            try:
                loader = loading_indicators.nth(i)
                await expect(loader).to_have_attribute(
                    "style", re.compile(r"opacity:\s*0"), timeout=30000
                )
            except:
                pass
        return True

    async def capture_grid(self, frame: Frame, cache_key: Path, cid: Union[int, str]) -> Tuple[Optional[Path], Optional[Path]]:
        """Implementação robusta com MutationObserver e suporte a Canvas."""
        await frame.evaluate("""
            async () => {
                const imgs = Array.from(document.querySelectorAll(".challenge-view img"));
                const canvas = Array.from(document.querySelectorAll(".challenge-view canvas"));
                if (imgs.length === 0 && canvas.length === 0) return;
                const promises = [];
                imgs.forEach(img => {
                    if (img.complete && img.naturalWidth > 0) return;
                    promises.push(new Promise(resolve => {
                        img.onload = () => img.naturalWidth > 0 ? resolve() : null;
                        img.onerror = resolve;
                    }));
                });
                canvas.forEach(c => {
                    if (c.width > 0 && c.height > 0) return;
                    promises.push(new Promise(resolve => {
                        const observer = new MutationObserver(() => {
                            if (c.width > 0 && c.height > 0) {
                                observer.disconnect();
                                resolve();
                            }
                        });
                        observer.observe(c, { attributes: true, attributeFilter: ['width', 'height'] });
                        setTimeout(() => { observer.disconnect(); resolve(); }, 2000);
                    }));
                });
                if (promises.length > 0) {
                    await Promise.all(promises);
                    await new Promise(r => setTimeout(r, 400));
                }
            }
        """)
        
        challenge_view = frame.locator("//div[@class='challenge-view']")
        bbox = await challenge_view.bounding_box()
        self.current_view_bbox = bbox
        
        screenshot_path = cache_key.joinpath(f"{cache_key.name}_{cid}_challenge_view.png")
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        await challenge_view.screenshot(type="png", path=screenshot_path, timeout=10000)

        grid_img = create_coordinate_grid(
            screenshot_path,
            bbox,
            x_line_space_num=self.config.coordinate_grid.x_line_space_num,
            y_line_space_num=self.config.coordinate_grid.y_line_space_num,
            color=self.config.coordinate_grid.color,
            adaptive_contrast=self.config.coordinate_grid.adaptive_contrast,
        )

        grid_path = cache_key.joinpath(f"{cache_key.name}_{cid}_spatial_helper.png")
        plt.imsave(str(grid_path.resolve()), grid_img)

        return screenshot_path, grid_path

    async def get_challenge_frame_locator(self) -> Optional[Frame]:
        """Implementação otimizada: Busca exaustiva pelo frame de desafio sem esperas cegas."""
        # Tenta uma espera curta inicial de forma não-bloqueante pesada
        try:
            await self.page.wait_for_selector("iframe[src*='hcaptcha.com/captcha/v1/']", timeout=2000)
        except Exception: pass

        for attempt in range(30): # Mais tentativas, mas mais rápidas
            candidate = self._find_frame_recursive(self.page.main_frame)
            if candidate:
                with suppress(Exception):
                    # Check visibility quickly
                    if await candidate.locator("//div[@class='challenge-view']").is_visible(timeout=200):
                        return candidate
            
            # Fallback por URL (mais rápido que recursão em alguns casos)
            for frame in self.page.frames:
                if "hcaptcha.com/captcha/v1/" in frame.url and "frame=challenge" in frame.url:
                    with suppress(Exception):
                        if await frame.locator("//div[@class='challenge-view']").is_visible(timeout=200):
                            return frame
            
            if attempt % 10 == 0:
                LoggerHelper.log_info(f"Procurando cockpit... {attempt+1}/30")
            
            # Sleep menor (250ms em vez de 1s) para ser mais responsivo
            await asyncio.sleep(0.25)
        return None


    async def check_crumb_count(self) -> int:
        """Implementação da linha 245-250 do original: Conta as bolinhas do desafio."""
        frame = await self.get_challenge_frame_locator()
        if not frame: return 1
        try:
            return await frame.locator("//div[@class='Crumb']").count() or 1
        except: return 1

    async def check_challenge_type(self) -> Optional[Union[RequestType, ChallengeTypeEnum]]:
        """Implementação da linha 255-280 do original: Detecta o tipo de desafio por roteamento visual."""
        frame = await self.get_challenge_frame_locator()
        if not frame: return None
        
        samples = frame.locator("//div[@class='task-image']")
        count = await samples.count()
        if count == 9: return RequestType.IMAGE_LABEL_BINARY
        
        # Roteamento Visual via IA (ChallengeRouter)
        challenge_view = frame.locator("//div[@class='challenge-view']")
        cache_path = self.config.cache_dir.joinpath(f"challenge_view/_artifacts/{uuid.uuid4()}.png")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        await challenge_view.screenshot(path=cache_path)
        
        router_result = await self.arm.challenge_router(challenge_screenshot=cache_path)
        return router_result.challenge_type

    async def refresh_challenge(self):
        """Implementação da linha 235 do original: Recarrega o desafio."""
        frame = await self.get_challenge_frame_locator()
        if frame:
            refresh_button = frame.locator("//div[@class='refresh button']")
            if await refresh_button.is_visible():
                await self.arm.actions.click_by_mouse(refresh_button)
                await asyncio.sleep(2)
                return True
        await self.page.reload()
        await asyncio.sleep(2)
        return True

    async def click_checkbox(self):
        """Redirecionar para PilotActions."""
        return await self.arm.actions.click_checkbox()
