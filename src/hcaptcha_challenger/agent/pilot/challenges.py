import asyncio
import time
import random
import os
import re
from pathlib import Path
from loguru import logger
from typing import Union, Tuple, Optional
from contextlib import suppress
from playwright.async_api import Page, Frame, Locator, expect, FrameLocator
from hcaptcha_challenger.models import ChallengeTypeEnum, RequestType
from hcaptcha_challenger.agent.logger import LoggerHelper, log_method_call, ChallengeTracker

class PilotChallenges:
    def __init__(self, arm):
        self.arm = arm
        self.tracker = ChallengeTracker()

    def get_tracker(self):
        return self.tracker

    async def _wait_for_all_loaders_complete(self, frame: Frame):
        """Implementação da linha 240-260 do original: Garante que as imagens do desafio carregaram."""
        await asyncio.sleep(self.arm.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS / 1000)
        loading_indicators = frame.locator("//div[@class='loading-indicator']")
        count = await loading_indicators.count()
        if count == 0: return True
        
        for i in range(count):
            try:
                await expect(loading_indicators.nth(i)).to_have_attribute("style", re.compile(r"opacity:\s*0"), timeout=30000)
            except: pass
        return True

    async def _capture_burst_frames(self, frame: FrameLocator | Frame, cache_key: Path, cid: int, count: int = 4) -> list[Path]:
        """
        Captura uma sequência de screenshots para desafios de movimento.
        """
        screenshots = []
        challenge_view = frame.locator("//div[@class='challenge-view']")
        
        for i in range(count):
            path = cache_key.joinpath(f"{cache_key.name}_{cid}_burst_{i}.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            await challenge_view.screenshot(type="png", path=path)
            screenshots.append(path)
            if i < count - 1:
                await asyncio.sleep(0.5) # 500ms entre frames para pegar a animação inteira
        
        return screenshots

    async def _capture_spatial_mapping(self, frame: Frame, cache_key: Path, cid: Union[int, str]) -> Tuple[Optional[Path], Optional[Path]]:
        """Implementação robusta da linha 270-340: Captura screenshot com MutationObserver e suporte a Canvas."""
        await frame.evaluate("""
            async () => {
                const imgs = Array.from(document.querySelectorAll(".challenge-view img"));
                const canvas = Array.from(document.querySelectorAll(".challenge-view canvas"));
                const promises = [];
                imgs.forEach(img => {
                    if (!(img.complete && img.naturalWidth > 0)) {
                        promises.push(new Promise(resolve => {
                            img.onload = resolve;
                            img.onerror = resolve;
                        }));
                    }
                });
                canvas.forEach(c => {
                    if (!(c.width > 0 && c.height > 0)) {
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
                    }
                });
                if (promises.length > 0) await Promise.all(promises);
                await new Promise(r => setTimeout(r, 400));
            }
        """)
        
        challenge_view = frame.locator("//div[@class='challenge-view']")
        bbox = await challenge_view.bounding_box()
        self.arm.navigation.current_view_bbox = bbox

        screenshot_path = cache_key.joinpath(f"{cache_key.name}_{cid}_challenge_view.png")
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        await challenge_view.screenshot(type="png", path=screenshot_path, timeout=5000)

        from hcaptcha_challenger.helper import create_coordinate_grid
        import matplotlib.pyplot as plt
        grid_img = create_coordinate_grid(
            screenshot_path, bbox,
            x_line_space_num=self.arm.config.coordinate_grid.x_line_space_num,
            y_line_space_num=self.arm.config.coordinate_grid.y_line_space_num,
            color=self.arm.config.coordinate_grid.color,
            adaptive_contrast=self.arm.config.coordinate_grid.adaptive_contrast,
        )

        grid_path = cache_key.joinpath(f"{cache_key.name}_{cid}_spatial_helper.png")
        plt.imsave(str(grid_path.resolve()), grid_img)

        return screenshot_path, grid_path

    async def _click_submit(self, frame):
        """Implementação da linha 980 do original: Clica no botão submit com simulação humana."""
        selectors = [
            "//div[@class='button-submit button']",
            "//div[contains(@class, 'button-submit')]",
            "text=Submit",
            "text=Verify",
            "button:has-text('Verify')"
        ]
        
        for selector in selectors:
            try:
                btn = frame.locator(selector).first
                if await btn.is_visible(timeout=1000):
                    await asyncio.sleep(random.uniform(0.7, 1.4))
                    await self.arm.actions.click_by_mouse(btn)
                    
                    await asyncio.sleep(2)
                    with suppress(Exception):
                        error_locator = frame.locator("//div[contains(@class, 'error-text')]")
                        if await error_locator.is_visible(timeout=1000):
                            LoggerHelper.log_error("hCaptcha recusou solução!", emoji='boom')
                            return False
                    
                    LoggerHelper.log_info("Ação enviada. Aguardando veredito...", emoji='hourglass')
                    return True
            except: continue
        return False

    @log_method_call(emoji='🧩', color='magenta')
    async def handle_drag_drop(self, job_type: ChallengeTypeEnum):
        frame = await self.arm.navigation.get_challenge_frame_locator()
        if not frame: return False
        
        cache_key = self.arm.config.create_cache_key(self.arm.captcha_payload)
        user_prompt = self.arm._match_user_prompt(job_type)
        self.tracker.start_challenge(user_prompt)

        for cid in range(self.arm.crumb_count):
            round_start = time.time()
            LoggerHelper.log_round_start(cid + 1, self.arm.crumb_count)
            await self._wait_for_all_loaders_complete(frame)
            
            raw, projection = await self._capture_spatial_mapping(frame, cache_key, cid)
            
            img_hash = self.arm.core.image_cache.get_hash(raw)
            if img_hash and img_hash in self.arm.core.image_cache.cache:
                LoggerHelper.log_info("Usando Cache (HIT 🎯)", emoji='kermit')
                response = self.arm.core.image_cache.cache[img_hash]
                ai_duration = 0
                model_used = "cache"
            else:
                model, available_keys = await self.arm._get_available_model_and_keys(
                    preferred_model=self.arm.config.SPATIAL_PATH_REASONER_MODEL
                )
                
                # Soul Alignment: Instrução de alta precisão (Portado da linha 825)
                ai_hint = (
                    f"{user_prompt}\n"
                    "STEP-BY-STEP INSTRUCTIONS:\n"
                    "1. The RIGHT side has draggable line segments stacked vertically.\n"
                    "2. The LEFT side has incomplete lines with visible GAPS.\n"
                    "3. Match each segment to its gap by ANGLE and CURVATURE continuity.\n"
                    "4. FROM = center of draggable piece (right side, higher X). TO = center of gap (left side, lower X).\n"
                    "5. Read all coordinates from the GRID AXIS LABELS, not pixel estimation.\n"
                    "6. VALIDATION: start_point.x MUST be > end_point.x (right to left movement)."
                )
                
                try:
                    start_ai = time.time()
                    response = await self.arm.spatial_path_reasoner(
                        challenge_screenshot=raw,
                        grid_divisions=projection,
                        auxiliary_information=ai_hint
                    )
                    ai_duration = time.time() - start_ai
                    model_used = model
                    if img_hash: self.arm.core.image_cache.cache[img_hash] = response
                    
                    # Feedback de quota: Sucesso
                    if model and available_keys:
                        k0 = available_keys[0]
                        self.arm.core.quota_manager.mark_success(k0, model)
                except Exception as e:
                    # Feedback de quota: Erro (Portabilidade Premium)
                    total_keys = len(self.arm.config.GEMINI_API_KEYS) if self.arm.config.GEMINI_API_KEYS else 1
                    current_key_idx = 1
                    
                    if model and available_keys:
                        k0 = available_keys[0]
                        err_msg = str(e).lower()
                        if "429" in err_msg or "exhausted" in err_msg:
                            self.arm.core.quota_manager.mark_exhausted(k0, model)
                        else:
                            self.arm.core.quota_manager.mark_failure(k0, model)
                        
                        # Tentar encontrar o índice real no config original
                        for i, key_obj in enumerate(self.arm.config.GEMINI_API_KEYS):
                            val = key_obj.get_secret_value() if hasattr(key_obj, "get_secret_value") else str(key_obj)
                            if val == k0:
                                current_key_idx = i + 1
                                break
                    
                    self.arm.log_provider_error(current_key_idx, total_keys, e)
                    raise e

            self.arm._log_ai_response(response, cid + 1, self.arm.crumb_count)
            LoggerHelper.log_ai_performance(model_used, ai_duration, len(response.paths))
            self.arm.metrics.log_ai_call(ai_duration)

            for path in response.paths:
                # Validação: corrigir coordenadas invertidas (FROM deve ser à direita, TO à esquerda)
                if path.start_point.x < path.end_point.x:
                    path.start_point, path.end_point = path.end_point, path.start_point
                if self.arm._validate_coordinate(path.start_point.x, path.start_point.y):
                    LoggerHelper.log_info(f"DRAG: ({int(path.start_point.x)}, {int(path.start_point.y)}) -> ({int(path.end_point.x)}, {int(path.end_point.y)})", emoji='drag')
                    await self.arm.actions.perform_drag_drop(path, delay_ms=random.randint(15, 25))
                    await asyncio.sleep(random.uniform(0.5, 0.8))

            await self._click_submit(frame)
            self.tracker.log_round(cid+1, True, time.time()-round_start, ai_duration, len(response.paths))

    @log_method_call(emoji='🎯', color='cyan')
    async def handle_label_select(self, job_type: ChallengeTypeEnum):
        frame = await self.arm.navigation.get_challenge_frame_locator()
        if not frame: return False
        
        cache_key = self.arm.config.create_cache_key(self.arm.captcha_payload)
        user_prompt = self.arm._match_user_prompt(job_type)
        self.tracker.start_challenge(user_prompt)

        for cid in range(self.arm.crumb_count):
            round_start = time.time()
            LoggerHelper.log_round_start(cid + 1, self.arm.crumb_count)
            
            await self._wait_for_all_loaders_complete(frame)

            # Soul Alignment: Detecção de Motion Pattern -> Burst Mode 📸
            real_prompt = self.arm.captcha_payload.get_requester_question().lower() if self.arm.captcha_payload else ""
            logger.info(f"DEBUG: Checking motion. Real Prompt: '{real_prompt}'")
            
            motion_keywords = [
                "motion", "pattern", "move", "moving",
                "different", "differently", "fastest", "slowest",
                "change", "changes", "changing",
                "transform", "transforms",
                "morph", "morphs",
                "becomes", "turns", "turning",
                "appear", "disappear",
                "animate", "animation",
                "shift", "rotate", "rotating",
            ]
            is_motion = any(k in real_prompt for k in motion_keywords)
            logger.info(f"DEBUG: is_motion={is_motion}")
            
            # Garantir que bbox seja definido antes
            bbox = await frame.locator("//div[@class='challenge-view']").bounding_box()
            self.arm.navigation.current_view_bbox = bbox

            if is_motion:
                num_frames = getattr(self.arm.config, "BURST_MODE_FRAMES", 6)
                LoggerHelper.log_info(f"Desafio de Movimento detectado! Ativando Burst Mode ({num_frames} frames)...", emoji='📸')
                challenge_screenshots = await self._capture_burst_frames(frame, cache_key, cid, count=num_frames)
                
                # Para grid, usamos o último frame do burst (bbox já definido acima)
                
                from hcaptcha_challenger.helper import create_coordinate_grid
                import matplotlib.pyplot as plt
                
                grid_result = create_coordinate_grid(
                    challenge_screenshots[-1], # Usa o último frame para o grid
                    bbox,

                    x_line_space_num=self.arm.config.coordinate_grid.x_line_space_num,
                    y_line_space_num=self.arm.config.coordinate_grid.y_line_space_num,
                    color=self.arm.config.coordinate_grid.color,
                    adaptive_contrast=self.arm.config.coordinate_grid.adaptive_contrast,
                )
                projection = cache_key.joinpath(f"{cache_key.name}_{cid}_spatial_helper.png")
                projection.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(str(projection.resolve()), grid_result)
                
                raw = challenge_screenshots # Passa a LISTA de paths
                LoggerHelper.log_info(f"Burst Mode concluído: {len(raw)} frames capturados.", emoji='🎞️')
                
                ai_hint = (
                    f"{user_prompt}\n"
                    "STEP-BY-STEP INSTRUCTIONS FOR MOTION DETECTION:\n"
                    f"1. You are given {num_frames} sequential frames of a 3x3 image grid.\n"
                    "2. Examine the 9 sub-images very carefully across the frames.\n"
                    "3. Look for the single sub-image where the object ANIMATES, MOVES, or CHANGES into a completely DIFFERENT object.\n"
                    "4. The change might be subtle (e.g. an object turning into another, or an animal morphing). Most will stay completely still.\n"
                    "5. Return the exact (X, Y) coordinates of the center of that specific changing sub-image using the provided coordinate grid."
                )
            else:
                raw, projection = await self._capture_spatial_mapping(frame, cache_key, cid)
                ai_hint = user_prompt
            
            model, available_keys = await self.arm._get_available_model_and_keys(
                preferred_model=self.arm.config.SPATIAL_POINT_REASONER_MODEL
            )
            
            try:
                start_ai = time.time()
                response = await self.arm.spatial_point_reasoner(
                    challenge_screenshot=raw,
                    grid_divisions=projection,
                    auxiliary_information=ai_hint
                )
                ai_duration = time.time() - start_ai
                if model and available_keys:
                    k0 = available_keys[0]
                    self.arm.core.quota_manager.mark_success(k0, model)
            except Exception as e:
                # Feedback de quota: Erro (Portabilidade Premium)
                total_keys = len(self.arm.config.GEMINI_API_KEYS) if self.arm.config.GEMINI_API_KEYS else 1
                current_key_idx = 1
                
                if model and available_keys:
                    k0 = available_keys[0]
                    err_msg = str(e).lower()
                    if "429" in err_msg or "exhausted" in err_msg:
                        self.arm.core.quota_manager.mark_exhausted(k0, model)
                    else:
                        self.arm.core.quota_manager.mark_failure(k0, model)
                    
                    for i, key_obj in enumerate(self.arm.config.GEMINI_API_KEYS):
                        val = key_obj.get_secret_value() if hasattr(key_obj, "get_secret_value") else str(key_obj)
                        if val == k0:
                            current_key_idx = i + 1
                            break
                
                self.arm.log_provider_error(current_key_idx, total_keys, e)
                raise e
            
            self.arm._log_ai_response(response, cid + 1, self.arm.crumb_count)
            points = getattr(response, 'points', [])
            LoggerHelper.log_ai_performance(model, ai_duration, len(points))
            self.arm.metrics.log_ai_call(ai_duration)

            for point in points:
                # SOUL ALIGNMENT: O SpatialPointReasoner retorna coordenadas GLOBAIS
                # A imagem grid_divisions contém labels de coordenadas absolutas
                LoggerHelper.log_info(f"IA Escolheu: ({point.x}, {point.y}) - BBox: {bbox}", emoji='🖱️')
                
                try:
                    tasks = frame.locator("//div[contains(@class, 'task')]")
                    count = await tasks.count()
                    if count in [9, 16]:
                        # ── Grade 3x3 ou 4x4: mapear coordenada IA para célula ──────────
                        grid_size = int(count ** 0.5)
                        rel_x = point.x - bbox['x']
                        rel_y = point.y - bbox['y']
                        cell_w = bbox['width'] / grid_size
                        cell_h = bbox['height'] / grid_size
                        col = max(0, min(int(rel_x // cell_w), grid_size - 1))
                        row = max(0, min(int(rel_y // cell_h), grid_size - 1))
                        idx = row * grid_size + col
                        LoggerHelper.log_info(f"Grid {grid_size}x{grid_size}: clicando célula {idx+1} ({int(rel_x)},{int(rel_y)})", emoji='🎯')
                        await self.arm.actions.click_by_mouse(tasks.nth(idx))

                    elif count >= 1:
                        # ── Single-select ou área de clique direto ───────────────────────
                        # image_label_single_select tem 1 task div; não é grade.
                        # A IA retorna coordenadas absolutas de tela → clicar diretamente.
                        LoggerHelper.log_info(f"Single-select (count={count}): clicando em ({point.x}, {point.y})", emoji='🎯')
                        await self.arm.page.mouse.click(point.x, point.y, delay=180)

                    else:
                        # ── count=0: canvas gigante (ex: image_label_single_select) ──────
                        LoggerHelper.log_info(f"Alvo único detectado (canvas): clicando diretamente em ({point.x}, {point.y})", emoji='🎯')
                        await self.arm.page.mouse.click(point.x, point.y, delay=180)
                except Exception as e:
                    LoggerHelper.log_warning(f"Fallback para clique absoluto devido a erro: {e}")
                    await self.arm.page.mouse.click(point.x, point.y, delay=180)
                    
                await asyncio.sleep(random.uniform(0.4, 0.6))

            await self._click_submit(frame)
            self.tracker.log_round(cid+1, True, time.time()-round_start, ai_duration, len(points))

    @log_method_call(emoji='🖼️', color='green')
    async def handle_binary(self):
        frame = await self.arm.navigation.get_challenge_frame_locator()
        if not frame: return False
        
        cache_key = self.arm.config.create_cache_key(self.arm.captcha_payload)
        user_prompt = self.arm._match_user_prompt(RequestType.IMAGE_LABEL_BINARY)
        self.tracker.start_challenge(user_prompt)

        for cid in range(self.arm.crumb_count):
            round_start = time.time()
            LoggerHelper.log_round_start(cid + 1, self.arm.crumb_count)
            await self._wait_for_all_loaders_complete(frame)
            
            raw = await self.arm._get_challenge_image(frame, cache_key, cid)
            model, available_keys = await self.arm._get_available_model_and_keys(
                preferred_model=self.arm.config.IMAGE_CLASSIFIER_MODEL
            )
            
            try:
                start_ai = time.time()
                response = await self.arm.image_classifier(
                    challenge_screenshot=raw, 
                    auxiliary_information=user_prompt
                )
                ai_duration = time.time() - start_ai
                if model and available_keys:
                    k0 = available_keys[0]
                    self.arm.core.quota_manager.mark_success(k0, model)
            except Exception as e:
                # Feedback de quota: Erro (Portabilidade Premium)
                total_keys = len(self.arm.config.GEMINI_API_KEYS) if self.arm.config.GEMINI_API_KEYS else 1
                current_key_idx = 1
                
                if model and available_keys:
                    k0 = available_keys[0]
                    err_msg = str(e).lower()
                    if "429" in err_msg or "exhausted" in err_msg:
                        self.arm.core.quota_manager.mark_exhausted(k0, model)
                    else:
                        self.arm.core.quota_manager.mark_failure(k0, model)
                    
                    for i, key_obj in enumerate(self.arm.config.GEMINI_API_KEYS):
                        val = key_obj.get_secret_value() if hasattr(key_obj, "get_secret_value") else str(key_obj)
                        if val == k0:
                            current_key_idx = i + 1
                            break
                
                self.arm.log_provider_error(current_key_idx, total_keys, e)
                raise e
            
            matrix = response.convert_box_to_boolean_matrix()
            LoggerHelper.log_ai_performance(model, ai_duration, sum(matrix))
            self.arm.metrics.log_ai_call(ai_duration)
            
            for i, should_click in enumerate(matrix):
                if should_click:
                    selector = f"//div[@class='task' and contains(@aria-label, '{i+1}')]"
                    await self.arm.actions.click_by_mouse(frame.locator(selector))
            
            await self._click_submit(frame)
            self.tracker.log_round(cid+1, True, time.time()-round_start, ai_duration, sum(matrix))


    async def debug_find_captcha(self):
        """
        Implementação da linha 345 original: Tenta encontrar componentes do captcha para debug.
        Sistema "Swiss Army Knife" para detecção e ativação.
        """
        page = self.arm.page
        LoggerHelper.log_info("Procurando captcha na página...", emoji='🔍')
        
        # 1. Tentar encontrar o Cockpit (Challenge Frame) diretamente primeiro
        frame = await self.arm.navigation.get_challenge_frame_locator()
        if frame:
            LoggerHelper.log_success(f"Cockpit encontrado: {frame.url[:60]}...", emoji='🎯')
            return True
            
        # 2. Varredura exaustiva de iframes (Parity Restoration)
        iframes = await page.query_selector_all('iframe')
        LoggerHelper.log_info(f"Total de iframes detectados: {len(iframes)}", emoji='📋')
        
        for i, iframe in enumerate(iframes):
            try:
                src = await iframe.get_attribute('src') or ''
                if 'hcaptcha' in src.lower() or 'captcha' in src.lower():
                    LoggerHelper.log_success(f"Iframe {i} POSSÍVEL CAPTCHA: {src[:60]}...", emoji='🎯')
                    
                    content_frame = await iframe.content_frame()
                    if content_frame:
                        # Verificar Checkbox
                        checkbox = content_frame.locator('div#checkbox')
                        if await checkbox.is_visible(timeout=1000):
                            LoggerHelper.log_success("Checkbox hCaptcha visível! Clicando...", emoji='✅')
                            await self.arm.actions.click_by_mouse(checkbox)
                            await asyncio.sleep(2) # Aguarda transição
                            return True
            except:
                continue

        # 3. Verificar por seletores de dados (data-sitekey, etc)
        selectors = [
            'div[class*="h-captcha"]', 'div[class*="hcaptcha"]', 
            'div[data-sitekey]', 'div#h-captcha', '.h-captcha'
        ]
        for selector in selectors:
            if await page.locator(selector).count() > 0:
                LoggerHelper.log_success(f"Elemento captcha detectado via seletor: {selector}", emoji='🎯')
                # Tenta clicar no checkbox se encontrar
                try: 
                    await self.arm.actions.click_checkbox() 
                    return True
                except: pass
                return True

        LoggerHelper.log_warning("Cockpit não localizado.")
        return False
