from pydantic import BaseModel, Field

from hcaptcha_challenger.models import JobTypeLiteral


class SkillRule(BaseModel):
    """Represents a single skill matching rule."""

    triggers: list[str] = Field(...)
    job_type: JobTypeLiteral | None = Field(default=None)
    template: str = Field(...)

    # Pre-computed lowercase triggers for faster matching
    _triggers_lower: list[str] | None = None

    def model_post_init(self, __context) -> None:
        """Pre-compute lowercase triggers after model initialization."""
        object.__setattr__(self, "_triggers_lower", [t.lower() for t in self.triggers])

    def matches_text(self, text_lower: str) -> bool:
        """Check if all triggers match the given lowercase text (AND logic)."""
        triggers = self._triggers_lower or [t.lower() for t in self.triggers]
        return all(trigger in text_lower for trigger in triggers)


class SkillManifest(BaseModel):
    """Represents the skill manifest containing version and rules."""

    version: str = Field(...)
    base_url: str | None = Field(default=None)
    rules: list[SkillRule] = Field(...)

    @staticmethod
    def get_download_url(repo: str, branch: str = "main") -> str:
        """Construct the raw GitHub URL for this manifest."""
        return f"https://raw.githubusercontent.com/{repo}/{branch}/src/hcaptcha_challenger/skills/rules.yaml"

    def get_library_base_url(self, repo: str, branch: str = "main") -> str:
        """Get the base URL for downloading template files."""
        return (
            self.base_url
            or f"https://raw.githubusercontent.com/{repo}/{branch}/src/hcaptcha_challenger/skills/library"
        )
