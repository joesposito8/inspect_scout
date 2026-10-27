"""Finding specifications stored alongside a Scout project."""

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ruamel.yaml import YAML
from shortuuid import uuid

from inspect_scout._project._project import read_project, write_project_config
from inspect_scout._project.types import ProjectConfig
from inspect_scout._scanjob import safe_scanner_name


class FindingSpec(BaseModel):
    """A standing question whose evidence is derived from project scans."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    labels: list[str] | None = None
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_sha: str = ""

    @field_validator("slug")
    @classmethod
    def valid_slug(cls, value: str) -> str:
        if value in {"all", "none", "bundle"}:
            raise ValueError(f"Finding slug '{value}' is reserved")
        return value

    @field_validator("labels")
    @classmethod
    def valid_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (
            not value
            or any(not label.strip() for label in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("Labels must be nonempty, distinct strings")
        return value

    @model_validator(mode="after")
    def hash_prompt(self) -> Self:
        if safe_scanner_name(self.scanner_name) != self.scanner_name:
            raise ValueError(
                "Finding slugs must be at most 47 characters to preserve their scanner names"
            )
        payload = json.dumps(
            [self.prompt, self.labels], ensure_ascii=False, separators=(",", ":")
        )
        self.prompt_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self

    @property
    def scanner_name(self) -> str:
        """Name under which this finding runs in a scan job."""
        return f"finding_{self.slug.replace('-', '_')}"


def finding_path(slug: str) -> Path:
    """Resolve a validated finding slug within the current project."""
    FindingSpec(slug=slug, title=slug, prompt="validate")
    return Path.cwd() / "findings" / slug / "finding.yaml"


def read_finding(slug: str) -> FindingSpec:
    """Read a finding from the current project."""
    path = finding_path(slug)
    spec = FindingSpec.model_validate(YAML(typ="safe").load(path))
    if spec.slug != slug:
        raise ValueError(f"Finding slug in {path} does not match '{slug}'")
    return spec


def list_findings() -> list[FindingSpec]:
    """Read registered findings in slug order."""
    return [
        read_finding(path.parent.name)
        for path in sorted((Path.cwd() / "findings").glob("*/finding.yaml"))
    ]


FindingSelection = Literal["all", "none"] | Sequence[str] | str | None
"""Findings for one scan: `"all"`, slugs as a list or a comma-separated string, or
`None`/`"none"` for no findings."""


def select_findings(selection: FindingSelection = None) -> list[FindingSpec]:
    """Resolve which registered findings a scan includes; a scan names its findings."""
    if selection is None or selection == "none":
        return []
    registered = {spec.slug: spec for spec in list_findings()}
    if selection == "all":
        return list(registered.values())
    slugs = selection.split(",") if isinstance(selection, str) else selection
    for slug in slugs:
        if slug not in registered:
            raise ValueError(f"Unknown finding '{slug}'")
    return [registered[slug] for slug in dict.fromkeys(slugs)]


def ensure_project_nonce() -> ProjectConfig:
    """Initialize a persistent nonce, preserving the project's other settings."""
    project = read_project()
    if project.nonce is None:
        nonce = uuid()
        write_project_config(ProjectConfig(nonce=nonce))
        project.nonce = nonce
    return project


def register_finding(
    slug: str,
    prompt: str,
    *,
    title: str | None = None,
    labels: list[str] | None = None,
) -> FindingSpec:
    """Create or update a finding, retaining its creation time."""
    path = finding_path(slug)
    previous = read_finding(slug) if path.exists() else None
    spec = FindingSpec(
        slug=slug,
        title=title
        or (previous.title if previous else slug.replace("-", " ").capitalize()),
        prompt=prompt,
        labels=labels,
        created=previous.created if previous else datetime.now(timezone.utc),
    )
    ensure_project_nonce()
    path.parent.mkdir(parents=True, exist_ok=True)
    YAML().dump(spec.model_dump(mode="json"), path)
    return spec
