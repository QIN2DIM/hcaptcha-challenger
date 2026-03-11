# -*- coding: utf-8 -*-
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union
from playwright.async_api import Page, Frame

from hcaptcha_challenger.models import CaptchaPayload, ChallengeTypeEnum, RequestType
from hcaptcha_challenger.skills import SkillManager
from hcaptcha_challenger.agent.logger import LoggerHelper, MetricsLogger, console
from hcaptcha_challenger.agent.pilot import PilotActions, PilotNavigation, PilotChallenges, PilotCore
from hcaptcha_challenger.tools import ImageClassifier, ChallengeRouter, SpatialPathReasoner, SpatialPointReasoner
from rich.panel import Panel
from rich.text import Text
from rich import box

class ImageCache:
    def __init__(self):
        self.cache = {}
    
    def get_hash(self, image_path: Path) -> Optional[str]:
        if not image_path or not image_path.exists():
            return None
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

class RoboticArm:
    """
    O Cockpit (Interface de Controle).
    Centraliza todos os recursos e coordenadores.
    """
    def __init__(self, page: Page, config):
        self.page = page
        self.config = config
        self.crumb_count = 1
        self.captcha_payload: Optional[CaptchaPayload] = None
        self.signal_crumb_count: Optional[int] = None
        
        # Infraestrutura de recursos
        self.metrics = MetricsLogger()
        self._skill_manager = SkillManager(agent_config=config)
        self._image_cache = ImageCache()
        
        # IA Reasoners (Support for Groq and Gemini)
        self._init_reasoners()
        
        # Sub-pilotos especializados
        self.actions = PilotActions(page, self)
        self.navigation = PilotNavigation(page, config, self)
        self.challenges = PilotChallenges(self)
        self.core = PilotCore(page, self, config)

    def _init_reasoners(self):
        """Inicializa os motores de IA baseados no provedor configurado."""
        if self.config.AI_PROVIDER == "groq":
            from hcaptcha_challenger.tools.internal.providers.groq import GroqProvider
            groq_keys = [k.get_secret_value() if hasattr(k, "get_secret_value") else str(k) 
                        for k in self.config.GROQ_API_KEYS] if self.config.GROQ_API_KEYS else []
            
            model = self.config.IMAGE_CLASSIFIER_MODEL or "llama-4-scout-17b-16e-instruct"
            provider = GroqProvider(api_key=groq_keys, model=model)
            
            self._image_classifier = ImageClassifier(gemini_api_key="", model=model, provider=provider)
            self._challenge_router = ChallengeRouter(gemini_api_key="", model=model, provider=provider)
            self._spatial_path_reasoner = SpatialPathReasoner(gemini_api_key="", model=model, provider=provider)
            self._spatial_point_reasoner = SpatialPointReasoner(gemini_api_key="", model=model, provider=provider)
        else:
            api_keys = [k.get_secret_value() if hasattr(k, "get_secret_value") else str(k) 
                       for k in self.config.GEMINI_API_KEYS] if self.config.GEMINI_API_KEYS else []

            self._image_classifier = ImageClassifier(
                gemini_api_key=api_keys,
                model=self.config.IMAGE_CLASSIFIER_MODEL,
            )
            self._challenge_router = ChallengeRouter(
                gemini_api_key=api_keys,
                model=self.config.CHALLENGE_CLASSIFIER_MODEL,
            )
            self._spatial_path_reasoner = SpatialPathReasoner(
                gemini_api_key=api_keys,
                model=self.config.SPATIAL_PATH_REASONER_MODEL,
            )
            self._spatial_point_reasoner = SpatialPointReasoner(
                gemini_api_key=api_keys,
                model=self.config.SPATIAL_POINT_REASONER_MODEL,
            )

    async def _get_available_model_and_keys(self, preferred_model: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
        """
        Retorna modelo e chaves não esgotadas baseado em prioridade.
        Portado das linhas 75-90 do original.
        """
        api_keys = self.config.GEMINI_API_KEYS
        
        # Prioridade de modelos (mais leves primeiro para garantir velocidade)
        model_priority = [
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite",
            "gemini-3-flash",
        ]

        
        # Se um modelo específico foi solicitado, usar ele
        if preferred_model:
            model_priority = [preferred_model] + [m for m in model_priority if m != preferred_model]
        
        for model in model_priority:
            available_keys = []
            for key in api_keys:
                key_str = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
                if not self.core.quota_manager.is_exhausted(key_str, model):
                    available_keys.append(key_str)
            
            if available_keys:
                return model, available_keys
        
        # Fallback
        if api_keys:
            k0 = api_keys[0].get_secret_value() if hasattr(api_keys[0], "get_secret_value") else str(api_keys[0])
            return model_priority[0], [k0]
        return None, []

    def _match_user_prompt(self, job_type: ChallengeTypeEnum) -> str:
        """
        Obtém o prompt específico do skill manager.
        Portado das linhas 215-225 do original.
        """
        try:
            challenge_prompt = (
                self.captcha_payload.get_requester_question()
                if self.captcha_payload
                else "hCaptcha Challenge"
            )
            if challenge_prompt and isinstance(challenge_prompt, str):
                return self._skill_manager.get_skill(challenge_prompt, job_type)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Erro ao processar prompt do captcha: {e}")

        return f"Please note that the current task type is: {job_type.value}"

    async def _get_challenge_image(self, frame: Frame, cache_key: Path, cid: Union[int, str]) -> Optional[Path]:
        """
        Captura a screenshot da área do desafio (3x3 grid).
        Portado das linhas 320-340 do original.
        """
        try:
            challenge_container = frame.locator("//div[contains(@class, 'challenge-container')]")
            if not await challenge_container.is_visible(timeout=2000):
                challenge_container = frame.locator("//div[@class='challenge-view']")
            
            output_path = cache_key.joinpath(f"{cache_key.name}_{cid}_binary_grid.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await challenge_container.screenshot(path=output_path, timeout=5000)
            return output_path
        except Exception as e:
            LoggerHelper.log_error(f"Erro ao capturar imagem do desafio: {str(e)[:100]}")
            return None

    def get_skill_manager(self):
        """Getter para o skill manager."""
        return self._skill_manager

    @property
    def image_classifier(self) -> ImageClassifier:
        return self._image_classifier

    @property
    def challenge_router(self) -> ChallengeRouter:
        return self._challenge_router

    @property
    def spatial_path_reasoner(self) -> SpatialPathReasoner:
        return self._spatial_path_reasoner

    @property
    def spatial_point_reasoner(self) -> SpatialPointReasoner:
        return self._spatial_point_reasoner

    def _log_ai_response(self, response, current_round: int, total_rounds: int):
        """Implementação da linha 130-160 do original: Log detalhado da resposta da IA."""
        title = f"🧠 IA ROUND {current_round}/{total_rounds}"
        if not response: return
        
        try:
            log_msg = getattr(response, 'log_message', None)
            if log_msg:
                if isinstance(log_msg, (dict, list)):
                    if isinstance(log_msg, dict) and "Challenge Propt" in log_msg:
                        log_msg["Challenge Prompt"] = log_msg.pop("Challenge Propt")
                    LoggerHelper.log_json(log_msg, title=title)
                else:
                    import json
                    import re
                    msg = str(log_msg).strip()
                    # 1. Sequências de escape ANSI tradicionais
                    msg = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', msg)
                    # 2. Resíduos de códigos de cores sem prefixo (ex: [32m, ;40m, 0m)
                    msg = re.sub(r'\[?\d+(?:;\d+)*m', '', msg)
                    # Remove caracteres de controle invisíveis
                    msg = "".join(ch for ch in msg if ord(ch) >= 32 or ch in "\n\r\t")
                    
                    if msg.startswith('{') or msg.startswith('['):

                        try:
                            data = json.loads(msg)
                            if isinstance(data, dict) and "Challenge Propt" in data:
                                data["Challenge Prompt"] = data.pop("Challenge Propt")
                            LoggerHelper.log_json(data, title=title)
                        except json.JSONDecodeError:
                            LoggerHelper.log_info(f"[{title}] {msg[:200]}...")
                    else:
                        LoggerHelper.log_info(f"[{title}] {msg[:200]}...")
        except Exception as e:
            LoggerHelper.log_debug(f"Falha ao formatar log da IA: {e}")
            LoggerHelper.log_info(f"[{title}] {response}")

    def _validate_coordinate(self, x: int, y: int) -> bool:
        """Sanity check portado da linha 180 do original."""
        bbox = self.navigation.current_view_bbox
        if not bbox: 
            LoggerHelper.log_warning(f"BBox não definido, coordenada ({x}, {y}) aceita.")
            return True
        
        bx, by = bbox.get('x', 0), bbox.get('y', 0)
        bw, bh = bbox.get('width', 1000), bbox.get('height', 1000)
        
        # Verificar se as coordenadas estão razoavelmente dentro da viewport
        # Margem de 10% para permitir pequenos desvios
        margin = 0.1
        min_x = bx - (bw * margin)
        max_x = bx + bw + (bw * margin)
        min_y = by - (bh * margin)
        max_y = by + bh + (bh * margin)
        
        is_valid = (min_x <= x <= max_x) and (min_y <= y <= max_y)
        
        if not is_valid:
            LoggerHelper.log_warning(
                f"Coordenadas suspeitas: ({x}, {y}) fora dos limites normais "
                f"({int(min_x)}-{int(max_x)}, {int(min_y)}-{int(max_y)})"
            )
            # Em vez de rejeitar, aplicar correção para dentro dos limites
            return False # Fallback será acionado pelo PilotChallenges
        
        return True

    def log_provider_error(self, current_key_index: int, total_keys: int, error: Exception):
        """
        Log de erro de provedor de IA portado das linhas 95-120 do original.
        """
        error_msg = str(error).lower()
        key_info = f"🔑 Chave {current_key_index}/{total_keys}"
        
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            LoggerHelper.log_error(f"{key_info}: Quota excedida", emoji='💸')
        elif "500" in error_msg or "internal" in error_msg:
            LoggerHelper.log_error(f"{key_info}: Erro interno do servidor", emoji='🔄')
        elif "timeout" in error_msg:
            LoggerHelper.log_warning(f"{key_info}: Timeout", emoji='⏰')
        else:
            LoggerHelper.log_error(f"{key_info}: {str(error)[:100]}", emoji='❌')

    def log_failure_summary(self, duration: float, error: str, retry_count: int, total_retries: int):
        """
        Resumo elegante de falha portado da funcionalidade premium original.
        """
        # REMOVER CÓDIGOS ANSI DO ERRO
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_error = ansi_escape.sub('', error)
        
        summary_text = Text()
        summary_text.append(f"⏱️  Duração: {duration:.1f}s\n", style="yellow")
        summary_text.append(f"🔄 Tentativa: {retry_count}/{total_retries}\n", style="cyan")
        summary_text.append(f"❌ Erro: {clean_error[:150]}", style="red")
        
        console.print(
            Panel(
                summary_text,
                title="[bold red]Resumo da Falha[/]",
                border_style="red",
                box=box.ROUNDED
            )
        )

    async def check_challenge_type(self) -> Optional[Union[RequestType, ChallengeTypeEnum]]:
        """Delega a detecção para o piloto de navegação - Bridge Pattern."""
        return await self.navigation.check_challenge_type()

    async def debug_find_captcha(self):
        """Delega a busca de componentes para o piloto de desafios - Bridge Pattern."""
        return await self.challenges.debug_find_captcha()

    async def check_crumb_count(self):
        """Delega para o piloto de navegação - Bridge Pattern."""
        return await self.navigation.check_crumb_count()

    async def wait_for_all_loaders_complete(self, frame: Frame):
        """Delega para o piloto de navegação - Bridge Pattern."""
        return await self.navigation.wait_for_loaders(frame)

    async def capture_spatial_mapping(self, frame: Frame, cache_key: Path, cid: Union[int, str]):
        """Delega para o piloto de navegação - Bridge Pattern."""
        return await self.navigation.capture_grid(frame, cache_key, cid)

    async def _perform_drag_drop(self, path, delay_ms: int = 15, steps: int = 40):
        """
        Método legado para manter compatibilidade com o baseline original.
        Delega para PilotActions.perform_drag_drop.
        """
        return await self.actions.perform_drag_drop(path, delay_ms=delay_ms, steps=steps)

    # Métodos de logging portados do original
    def _log_state_change(self, from_state: str, to_state: str):
        LoggerHelper.log_info(
            f"Estado: [yellow]{from_state}[/] → [green]{to_state}[/]",
            emoji='flag'
        )
    
    def _log_challenge_info(self, challenge_type: str, crumb_count: int):
        LoggerHelper.log_challenge_start(challenge_type, crumb_count, None)
    
    def _log_mouse_action(self, action: str, coordinates: Tuple[float, float], element: str = None):
        x, y = coordinates if coordinates else (0, 0)
        LoggerHelper.log_mouse_action(action, int(x), int(y), element)
