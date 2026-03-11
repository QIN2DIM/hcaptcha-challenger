import os
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple, Union
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from hcaptcha_challenger.models import (
    RequestType,
    ChallengeTypeEnum,
    SCoTModelType,
    DEFAULT_SCOT_MODEL,
    DEFAULT_FAST_SHOT_MODEL,
    FastShotModelType,
    IGNORE_REQUEST_TYPE_LITERAL,
    CoordinateGrid,
    CaptchaPayload,
    INV
)

VERSION = "0.20.0"

SINGLE_IGNORE_TYPE = Union[IGNORE_REQUEST_TYPE_LITERAL, RequestType, ChallengeTypeEnum]
IGNORE_REQUEST_TYPE_LIST = List[SINGLE_IGNORE_TYPE]

class SolveState(Enum):
    INIT = "init"
    CHALLENGE_PENDING = "challenge_pending"
    CHALLENGING = "challenging"
    SUBMITTED = "submitted"
    SUCCESS = "success"
    FAILURE = "failure"

class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    GEMINI_API_KEYS: Any = Field(
        default=None,
        description="Comma-separated list of Gemini API keys",
    )
    GROQ_API_KEYS: Any = Field(
        default=None,
        description="Comma-separated list of Groq API keys",
    )
    AI_PROVIDER: str = Field(
        default="gemini",
        description="AI provider to use (gemini, groq)",
    )
    USER_AGENT: str | None = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        description="Consistent User-Agent to avoid bot detection",
    )

    cache_dir: Path = Path("tmp/.cache")
    challenge_dir: Path = Path("tmp/.challenge")
    captcha_response_dir: Path = Path("tmp/.captcha")
    ignore_request_types: IGNORE_REQUEST_TYPE_LIST | None = Field(default_factory=list)
    ignore_request_questions: List[str] | None = Field(default_factory=list)

    DEFAULT_SCOT_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL,
        description="Default SCoT model for various tasks"
    )

    DISABLE_BEZIER_TRAJECTORY: bool = Field(
        default=False,
        description="If you use Camoufox, it is recommended to turn off "
        "the custom Bessel track generator of hcaptcha-challenger "
        "and use Camoufox(humanize=True)",
    )

    MAX_CRUMB_COUNT: int = Field(
        default=2,
        description="""
        CRUMB_COUNT: The number of challenge rounds you need to solve once the challenge starts.
        In the vast majority of cases this value will be 2, some specialized sites will set this value to 3.
        In most cases you don't need to change this value, the `_review_challenge_type` task determines the exact value of `CRUMB_COUNT` based on the information of the assigned task.
        Only manually change this value if you are working on a very specific task that prevents the `_review_challenge_type` from hijacking the task information and the maximum number of tasks > 2.
        """,
    )

    MAX_CHALLENGE_ATTEMPTS: int = Field(
        default=4,
        description="Maximum number of attempts to click the checkbox etc.",
    )

    MAX_RESETS: int = Field(
        default=1,
        description="Maximum number of challenge resets (new challenge after submit) before aborting.",
    )

    EXECUTION_TIMEOUT: float = Field(
        default=180,
        description="When your local network is poor, increase this value appropriately [unit: second]",
    )
    RESPONSE_TIMEOUT: float = Field(
        default=45,
        description="When your local network is poor, increase this value appropriately [unit: second]",
    )
    RETRY_ON_FAILURE: bool = Field(
        default=True, description="Re-execute the challenge when it fails"
    )
    WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS: int = Field(
        default=1500,
        description="When your local network is poor, increase this value appropriately [unit: millisecond]",
    )

    CHALLENGE_CLASSIFIER_MODEL: FastShotModelType = Field(
        default="gemini-2.5-flash",
        description="For the challenge classification task \n"
        "Used as last resort when HSW decoding fails.",
    )
    IMAGE_CLASSIFIER_MODEL: SCoTModelType = Field(
        default="gemini-2.5-flash", description="For the challenge type: `image_label_binary`"
    )
    SPATIAL_POINT_REASONER_MODEL: SCoTModelType = Field(
        default="gemini-2.5-flash",
        description="For the challenge type: `image_label_area_select` (single/multi)",
    )
    SPATIAL_PATH_REASONER_MODEL: SCoTModelType = Field(
        default="gemini-2.5-flash",
        description="For the challenge type: `image_drag_drop` (single/multi)",
    )

    coordinate_grid: CoordinateGrid | None = Field(default_factory=CoordinateGrid)

    enable_challenger_debug: bool | None = Field(default=False, description="Enable debug mode")

    # == Skills Configuration == #
    custom_skills_path: Path | None = Field(
        default=None, description="Path to custom skills rules.yaml"
    )
    enable_skills_update: bool = Field(
        default=False, description="Enable auto-update of skills from GitHub"
    )
    skills_update_repo: str = Field(
        default="QIN2DIM/hcaptcha-challenger", description="GitHub repo for skills update"
    )
    skills_update_branch: str = Field(default="main", description="GitHub branch for skills update")

    @field_validator('GEMINI_API_KEYS', mode="before")
    @classmethod
    def validate_api_keys(cls, v: Any) -> List[SecretStr]:
        """
        Validates and parses Gemini API keys.
        Supports comma-separated strings (from env) or lists.
        """
        # 1. Handle string input (e.g. from environment variable)
        if isinstance(v, str):
            v = [k.strip() for k in v.split(",") if k.strip()]
        
        # 2. Fallback to environment variables if still empty
        if not v:
            env_keys = os.environ.get("GEMINI_API_KEYS", "")
            if env_keys:
                v = [k.strip() for k in env_keys.split(",") if k.strip()]
            else:
                single_key = os.environ.get("GEMINI_API_KEY")
                if single_key:
                    v = [single_key]
        
        # 3. Final validation
        if not v:
            raise ValueError(
                "GEMINI_API_KEYS is required but not provided. "
                "Please set the GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY environment variable."
            )
            
        # 4. Ensure everything is a SecretStr
        return [SecretStr(k) if isinstance(k, str) else k for k in v]

    @property
    def spatial_grid_cache(self):
        return self.cache_dir.joinpath("spatial_grid")

    def create_cache_key(
        self,
        captcha_payload: CaptchaPayload | None = None,
        request_type: str = "type",
        prompt: str = "unknown",
    ) -> Path:
        """
        Cria uma chave de cache estruturada e descritiva.
        Portado das linhas 231-278 do baseline original.
        """
        import json
        from datetime import datetime
        
        current_datetime = datetime.now()
        # Formato: 20240108/20240108185504123456
        current_time = current_datetime.strftime("%Y%m%d/%Y%m%d%H%M%S%f")

        # Limpar o prompt para ser usado em nomes de diretórios
        prompt = prompt.translate(str.maketrans("", "", "".join(INV)))

        if not captcha_payload:
            cache_key = self.challenge_dir.joinpath(request_type, prompt, current_time)
            return cache_key

        # Extrair metadados para estrutura de diretórios
        job_type = captcha_payload.request_type.value if captcha_payload.request_type else "unknown"
        question = captcha_payload.get_requester_question()
        
        # Estrutura: tmp/.challenge / request_type / prompt / current_time
        cache_key = self.challenge_dir.joinpath(job_type, question, current_time)

        try:
            # Salvar o payload original para depuração (Soul Alignment)
            payload_path = cache_key.joinpath(f"{cache_key.name}_captcha.json")
            payload_path.parent.mkdir(parents=True, exist_ok=True)

            payload_data = captcha_payload.model_dump(mode="json")
            payload_path.write_text(
                json.dumps(payload_data, indent=2, ensure_ascii=False), 
                encoding="utf8"
            )
        except Exception:
            pass

        return cache_key
