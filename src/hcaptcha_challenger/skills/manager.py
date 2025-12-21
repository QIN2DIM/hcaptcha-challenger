from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import yaml
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from hcaptcha_challenger.models import ChallengeTypeEnum
from hcaptcha_challenger.skills.schema import SkillManifest, SkillRule

if TYPE_CHECKING:
    from hcaptcha_challenger.agent.challenger import AgentConfig


class SkillManager:
    """
    Manages skill templates and rules for challenge solving.

    Skills are loaded with layered priority: User > Cache > Built-in.
    Supports remote updates from GitHub with progress tracking.
    """

    def __init__(self, agent_config: "AgentConfig | None" = None):
        self._config = agent_config
        self._rules: list[SkillRule] = []
        self._manifest: SkillManifest | None = None
        self._template_cache: dict[str, str] = {}

        # Built-in paths (always available)
        self._builtin_dir = Path(__file__).parent
        self._builtin_rules_path = self._builtin_dir / "rules.yaml"
        self._builtin_library_path = self._builtin_dir / "library"
        self._current_library_path: Path = self._builtin_library_path

        self._init_skills()

    # ==================== Properties ====================

    @cached_property
    def _user_rules_path(self) -> Path | None:
        """User-defined custom skills path (highest priority)."""
        if self._config and self._config.custom_skills_path:
            return Path(self._config.custom_skills_path)
        return None

    @cached_property
    def _cache_dir(self) -> Path:
        """Cache directory for downloaded skills."""
        if self._config and self._config.cache_dir:
            return self._config.cache_dir / "skills"
        return Path("tmp/.cache/skills")

    @property
    def _cache_rules_path(self) -> Path:
        return self._cache_dir / "rules.yaml"

    @property
    def _cache_library_path(self) -> Path:
        return self._cache_dir / "library"

    @property
    def rules(self) -> list[SkillRule]:
        """Current loaded rules."""
        return self._rules

    @property
    def manifest(self) -> SkillManifest | None:
        """Current loaded manifest."""
        return self._manifest

    # ==================== Initialization ====================

    def _init_skills(self) -> None:
        """Initialize skills with layered priority: User > Cache > Built-in."""
        # 1. Try User Config (Hard Fail - user explicitly configured this)
        if self._user_rules_path:
            if not self._user_rules_path.exists():
                raise FileNotFoundError(f"Custom skills path not found: {self._user_rules_path}")
            self._load_rules(self._user_rules_path)
            logger.info(f"Loaded user custom skills from {self._user_rules_path}")
            return

        # 2. Try Cache (Soft Fail - fall back gracefully)
        if self._should_use_cache():
            try:
                self._load_rules(self._cache_rules_path)
                logger.debug(f"Loaded cached skills from {self._cache_rules_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached skills: {e}. Falling back to built-in.")

        # 3. Built-in (Final fallback)
        try:
            self._load_rules(self._builtin_rules_path)
            # logger.debug(f"Loaded built-in skills from {self._builtin_rules_path}")
        except Exception as e:
            logger.error(f"Failed to load built-in skills: {e}")
            self._rules = []

    def _should_use_cache(self) -> bool:
        """Check if cached skills should be used."""
        return bool(
            self._config and self._config.enable_skills_update and self._cache_rules_path.exists()
        )

    def _load_rules(self, path: Path) -> None:
        """Load rules from a YAML file and set the corresponding library path."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._manifest = SkillManifest(**data)
        self._rules = self._manifest.rules

        # Set library path relative to rules file
        # Convention: rules.yaml is in .../skills/rules.yaml, templates in .../skills/library/
        self._current_library_path = path.parent / "library"

        # Clear template cache when rules change
        self._template_cache.clear()

    # ==================== Skill Matching ====================

    def get_skill(self, challenge_text: str, job_type: ChallengeTypeEnum | None = None) -> str:
        """
        Match a skill based on challenge text and job type.

        Args:
            challenge_text: The challenge prompt text to match against.
            job_type: Optional challenge type for more precise matching.

        Returns:
            The markdown content of the matched template, or a fallback string.
        """
        if not challenge_text:
            return self._fallback_prompt(job_type)

        matched_rule = self._find_matching_rule(challenge_text, job_type)

        if matched_rule:
            return self._load_template_content(matched_rule.template)

        return self._fallback_prompt(job_type)

    def _find_matching_rule(
        self, text: str, job_type: ChallengeTypeEnum | None
    ) -> SkillRule | None:
        """
        Find the first rule that matches the given text and job type.

        Uses generator expression with next() for early termination.
        """
        text_lower = text.lower()
        job_type_value = job_type.value if job_type else None

        def matches(rule: SkillRule) -> bool:
            # Job type filter: if rule specifies job_type, it must match
            if rule.job_type:
                if not job_type_value or rule.job_type != job_type_value:
                    return False
            # Trigger matching using pre-computed lowercase triggers
            return rule.matches_text(text_lower)

        return next((rule for rule in self._rules if matches(rule)), None)

    def _load_template_content(self, filename: str) -> str:
        """
        Load markdown template content with caching.

        Uses an instance-level cache to avoid repeated disk I/O.
        """
        # Check cache first
        if filename in self._template_cache:
            return self._template_cache[filename]

        try:
            file_path = self._current_library_path / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8").strip()
                self._template_cache[filename] = content
                return content

            logger.warning(f"Template file not found: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error loading template {filename}: {e}")
            return ""

    @staticmethod
    def _fallback_prompt(job_type: ChallengeTypeEnum | None) -> str:
        """Generate a fallback prompt when no matching rule is found."""
        if job_type:
            return f"JobType: {job_type.value}"
        return ""

    # ==================== Remote Update ====================

    async def update_skills(self) -> None:
        """
        Pull skill updates from GitHub Raw.

        This should be called as a background task. Downloads are sequential
        with a Rich progress bar for visual feedback.
        """
        if not self._config or not self._config.enable_skills_update:
            return

        repo = self._config.skills_update_repo
        branch = self._config.skills_update_branch

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 1. Download and parse manifest
                manifest_url = SkillManifest.get_download_url(repo, branch)
                remote_manifest = await self._download_manifest(client, manifest_url)

                if not remote_manifest:
                    return

                # 2. Prepare cache directories
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_library_path.mkdir(parents=True, exist_ok=True)

                # 3. Save rules.yaml
                async with httpx.AsyncClient(timeout=10.0) as manifest_client:
                    resp = await manifest_client.get(manifest_url)
                    resp.raise_for_status()
                    self._cache_rules_path.write_text(resp.text, encoding="utf-8")

                # 4. Download templates with progress bar
                base_url = remote_manifest.get_library_base_url(repo, branch)
                await self._download_templates(client, remote_manifest.rules, base_url)

                # 5. Clear template cache and log success
                self._template_cache.clear()
                logger.info(f"Skills updated to version {remote_manifest.version}")

        except Exception as e:
            logger.warning(f"Failed to update skills: {e}")

    @staticmethod
    async def _download_manifest(client: httpx.AsyncClient, url: str) -> SkillManifest | None:
        """Download and parse the remote manifest."""
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = yaml.safe_load(resp.text)
            return SkillManifest(**data)
        except Exception as e:
            logger.warning(f"Failed to download manifest: {e}")
            return None

    async def _download_templates(
        self, client: httpx.AsyncClient, rules: list[SkillRule], base_url: str
    ) -> None:
        """Download template files sequentially with Rich progress bar."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Downloading skill templates...", total=len(rules))

            for rule in rules:
                template_url = f"{base_url}/{rule.template}"
                template_dest = self._cache_library_path / rule.template

                try:
                    resp = await client.get(template_url, timeout=5.0)
                    resp.raise_for_status()
                    template_dest.write_text(resp.text, encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to download template {rule.template}: {e}")

                progress.update(task, advance=1, description=f"[cyan]Downloaded {rule.template}")
