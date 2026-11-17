import asyncio
import json
import msgpack
import hashlib
import time
from datetime import datetime
from pathlib import Path
from asyncio import Queue
from typing import List, Optional, Any, Union
from loguru import logger
from playwright.async_api import Page, Response

from hcaptcha_challenger.models import CaptchaResponse, CaptchaPayload, ChallengeSignal, RequestType, ChallengeTypeEnum
from hcaptcha_challenger.agent.logger import LoggerHelper, NetworkLogger
from hcaptcha_challenger.agent.quota_manager import QuotaManager

class ImageCache:
    def __init__(self):
        self.cache = {}
    
    def get_hash(self, image_path: Path):
        if not image_path or not image_path.exists():
            return None
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

class PilotCore:
    def __init__(self, page: Page, arm, config):
        self.page = page
        self.arm = arm
        self.config = config
        
        self.captcha_payload_queue: Queue[Optional[CaptchaPayload]] = Queue()
        self.captcha_response_queue: Queue[CaptchaResponse] = Queue()
        self.cr_list: List[CaptchaResponse] = []
        
        # Infrastructure Resources
        self.quota_manager = QuotaManager(config.cache_dir)
        self.image_cache = ImageCache()
        self.network_logger = NetworkLogger(interval_seconds=5.0)
        
        # Soul Alignment: Evitar multiplicação de listeners em páginas reutilizadas (Singleton Browser)
        # Portado da lógica de paridade: remove o handler anterior antes de registrar o novo
        old_handler = getattr(self.page, "_h_handler", None)
        if old_handler:
            try:
                self.page.remove_listener("response", old_handler)
            except: pass
        
        self.page.on("response", self.task_handler)
        self.page._h_handler = self.task_handler

    async def task_handler(self, response: Response):
        self.network_logger.log_request()
        
        # 1. HSW Injection (Dual Context)
        if response.url.endswith("/hsw.js"):
            try:
                LoggerHelper.log_info("Injetando script HSW (Dual Context)...", emoji='inject')
                hsw_text = await response.text()
                await self.page.evaluate(hsw_text)
                if response.frame: await response.frame.evaluate(hsw_text)
                await self.page.add_init_script(hsw_text)
            except Exception as e:
                logger.error(f"Erro ao injetar HSW: {e}")
                
        # 2. GetCaptcha Interception (Msgpack Support)
        elif "/getcaptcha/" in response.url:
            try:
                if response.headers.get("content-type") == "application/json":
                    data = await response.json()
                    if data.get("pass"):
                        while not self.captcha_response_queue.empty(): self.captcha_response_queue.get_nowait()
                        self.captcha_response_queue.put_nowait(CaptchaResponse(**data))
                    elif data.get("request_config"):
                        self.captcha_payload_queue.put_nowait(CaptchaPayload(**data))
                else:
                    raw_data = await response.body()
                    context = response.frame if response.frame else self.page
                    has_hsw = await context.evaluate("() => typeof hsw === 'function'")
                    if not has_hsw: logger.warning("HSW ausente durante binário.")
                        
                    result = await context.evaluate("async (data) => { try { const res = await hsw(0, new Uint8Array(data)); return Array.from(res); } catch(e) { return null; } }", list(raw_data))
                    if result:
                        unpacked = msgpack.unpackb(bytes(result))
                        self.captcha_payload_queue.put_nowait(CaptchaPayload(**unpacked))
                    else: self.captcha_payload_queue.put_nowait(None)
            except Exception as e:
                logger.error(f"Erro no processador de Captcha (Get): {e}")
                self.captcha_payload_queue.put_nowait(None)
                
        # 3. CheckCaptcha Result
        elif "/checkcaptcha/" in response.url:
            try:
                data = await response.json()
                self.captcha_response_queue.put_nowait(CaptchaResponse(**data))
            except: pass

    async def review_challenge_type(self) -> Optional[Union[RequestType, ChallengeTypeEnum]]:
        """Implementação da linha 1090-1140 do AgentV: Decodifica o payload e decide a missão."""
        try:
            payload = await asyncio.wait_for(self.captcha_payload_queue.get(), timeout=30)
            if not payload: return None
            
            self.arm.captcha_payload = payload
            prompt = payload.get_requester_question().lower()
            
            # Soul Alignment: Verificação de questões ignoradas (Portado da linha 1160)
            if self.config.ignore_request_questions:
                for q in self.config.ignore_request_questions:
                    if q in prompt:
                        LoggerHelper.log_warning(f"Ignorando desafio por questão proibida: '{q}'", emoji='skip')
                        await self.arm.navigation.refresh_challenge()
                        return None

            # Keyword Overrides (Soul Alignment: Refined to avoid leakage)
            # Apenas substitui se o prompt for EXTREMAMENTE específico ou se o tipo original for ambíguo
            drag_keywords = ["drag", "arraste", "puzzle", "segment", "arraste-o", "mova", "piece"]
            
            # Se já for um tipo de drag conhecido pela rede, não precisamos de override pro básico
            is_already_drag = payload.request_type in [RequestType.IMAGE_DRAG_DROP]
            
            if any(k in prompt for k in drag_keywords) or is_already_drag:
                # Se detectado por palavra-chave mas o tipo é visualmente outro (ex: select), logar aviso
                if payload.request_type not in [RequestType.IMAGE_DRAG_DROP] and not is_already_drag:
                    LoggerHelper.log_warning(f"Override agressivo detectado: '{prompt}' (Tipo real: {payload.request_type})", emoji='⚠️')
                
                LoggerHelper.log_info(f"Roteamento: Arraste detectado: '{prompt}'", emoji='target')
                self.arm.crumb_count = len(payload.tasklist)
                try: 
                    return ChallengeTypeEnum.IMAGE_DRAG_SINGLE if len(payload.tasklist[0].entities) == 1 else ChallengeTypeEnum.IMAGE_DRAG_MULTI
                except:
                    return ChallengeTypeEnum.IMAGE_DRAG_MULTI
            
            # Motion/Video Detection (Soul Alignment: Video Motion Support)
            # CORREÇÃO: "Select" + "motion" = image_label_single_select
            if "select" in prompt and "motion" in prompt:
                LoggerHelper.log_info(f"Detectado: Seleção de objeto com padrão de movimento: '{prompt}'", emoji='target')
                self.arm.crumb_count = len(payload.tasklist)
                return ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT
            
            video_keywords = ["video", "clip"]
            if any(k in prompt for k in video_keywords):
                 LoggerHelper.log_info(f"Detectado: Desafio de vídeo (não suportado, tentando como imagem): '{prompt}'", emoji='⚠️')
                 self.arm.crumb_count = 1
                 return ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT
                
            match payload.request_type:
                case RequestType.IMAGE_LABEL_BINARY:
                    self.arm.crumb_count = int(len(payload.tasklist) / 9)
                    return RequestType.IMAGE_LABEL_BINARY
                case RequestType.IMAGE_DRAG_DROP:
                    self.arm.crumb_count = len(payload.tasklist)
                    return ChallengeTypeEnum.IMAGE_DRAG_SINGLE
                case RequestType.IMAGE_LABEL_AREA_SELECT:
                    self.arm.crumb_count = len(payload.tasklist)
                    max_shapes = payload.request_config.max_shapes_per_image if payload.request_config else 1
                    return ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT if max_shapes == 1 else ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT

            # Fallback para roteamento visual delegado ao RoboticArm
            return await self.arm.check_challenge_type()
        except: return None

    async def solve_captcha(self, ctype: Union[RequestType, ChallengeTypeEnum]):
        """Implementação da linha 1150-1210 do AgentV: Delega para o especialista correto."""
        if ctype == RequestType.IMAGE_LABEL_BINARY:
            await self.arm.challenges.handle_binary()
        elif ctype in [ChallengeTypeEnum.IMAGE_DRAG_SINGLE, ChallengeTypeEnum.IMAGE_DRAG_MULTI]:
            await self.arm.challenges.handle_drag_drop(ctype)
        elif ctype in [ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT, ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT]:
            await self.arm.challenges.handle_label_select(ctype)
        return True

    def cache_validated_response(self, cr: CaptchaResponse):
        """Implementação da linha 1055-1065 do AgentV: Salva o token de sucesso."""
        if not cr.is_pass: return
        self.cr_list.append(cr)
        try:
            current_time = datetime.now().strftime("%Y%m%d/%Y%m%d%H%M%S%f")
            path = self.config.captcha_response_dir.joinpath(f"{current_time}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cr.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False), encoding="utf-8")
        except: pass
