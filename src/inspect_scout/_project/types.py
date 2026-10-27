"""Project configuration types."""

from pydantic import BaseModel, ConfigDict, Field

from inspect_scout._scanjob_config import ScanJobConfig
from inspect_scout._transcript.types import MessageType


class FindingsConfig(BaseModel):
    """Project defaults for finding scanners."""

    model_config = ConfigDict(extra="forbid")

    model: str = "anthropic/claude-haiku-5-5"
    """Model used by finding scanners."""

    context_window: int = Field(default=100_000, gt=0)
    """Shared request token budget, including a fixed prompt reserve."""

    messages: list[MessageType] = Field(
        default_factory=lambda: list[MessageType](["assistant", "user"]), min_length=1
    )
    """Message roles to read; assistant tool-call arguments are retained."""


class ProjectConfig(ScanJobConfig):
    """Scout project configuration from scout.yaml.

    Extends ScanJobConfig to represent project-level defaults. All fields
    from ScanJobConfig are available as project defaults.
    """

    findings: FindingsConfig = Field(default_factory=FindingsConfig)
    """Finding scanner defaults."""

    nonce: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]{8,}$")
    """Persistent transcript marker code shared by project finding scanners."""

    trust_content: bool | None = Field(default=None)
    """Whether `scout view` may render model output richly (markdown, syntax highlighting, ANSI colors, media, links).

    `False` shows all model output as plain text, whatever each transcript's own setting. `None` (the default) and `True` defer to each transcript. Configures the viewer only; scans ignore it.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
