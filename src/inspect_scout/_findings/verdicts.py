"""Append-only human and model decisions about finding exhibits."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .spec import FindingSpec, finding_path

VerdictState = Literal["confirmed", "rejected", "proposed"]

# A verdict older versions offered; rows carrying it are skipped on load.
RETIRED_VERDICT = "unsure"


class VerdictRow(BaseModel):
    """One decision; file order breaks ties within the same authority."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1)
    verdict: VerdictState
    label: str | None = None
    decided_by: str = Field(pattern=r"^(human|judge):.+$")
    reason: str
    source: Literal["manual", "judge", "recheck"]
    """`recheck` marks judge rows written by older versions."""
    added: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def valid_authority(self) -> Self:
        if self.verdict not in self.allowed_verdicts():
            raise ValueError(f"{self.decided_by} cannot write {self.verdict}")
        return self

    def allowed_verdicts(self) -> set[VerdictState]:
        """Decisions this row's author may record."""
        # Judge rows from older versions proposed rather than confirmed.
        return (
            {"confirmed", "rejected"}
            if self.decided_by.startswith("human:")
            else {"confirmed", "rejected", "proposed"}
        )


class JudgeVerdictRow(VerdictRow):
    """A judge row from older versions, which also stored the judge's excerpt."""

    quote: str = Field(max_length=200)
    refused: Literal[False] = False
    """Always written by older versions; their refusals carried the retired verdict."""


class JudgeRun(BaseModel):
    """Counts from one judge run; refusals leave no verdict row, so they live here."""

    model_config = ConfigDict(extra="forbid")

    model: str
    only: Literal["proposed", "all"]
    candidates: int
    confirmed: int
    rejected: int
    refused: int
    failed: int
    """Candidates without a usable answer (failed, unparsed or invalid label)."""
    input_tokens: int
    input_tokens_cache_write: int
    input_tokens_cache_read: int
    output_tokens: int
    cost: float | None
    """USD for the paid calls; `None` for a model without a known price."""
    added: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _verdicts_path(finding: FindingSpec) -> Path:
    return finding_path(finding.slug).with_name("verdicts.jsonl")


def _judge_runs_path(finding: FindingSpec) -> Path:
    return finding_path(finding.slug).with_name("judge_runs.jsonl")


def read_verdicts(finding: FindingSpec) -> list[VerdictRow]:
    """Read decisions in append order, reporting malformed rows with their location."""
    path = _verdicts_path(finding)
    if not path.exists():
        return []
    rows: list[VerdictRow] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                if isinstance(value, dict) and value.get("verdict") == RETIRED_VERDICT:
                    continue
                row_type = (
                    JudgeVerdictRow
                    if isinstance(value, dict) and "quote" in value
                    else VerdictRow
                )
                rows.append(row_type.model_validate(value))
            except ValueError as ex:
                raise ValueError(f"Invalid verdict in {path}:{number}: {ex}") from ex
    return rows


def retired_verdicts_warning(finding: FindingSpec) -> str | None:
    """Name the ledger and count its rows with a retired verdict, if any."""
    path = _verdicts_path(finding)
    if not path.exists():
        return None
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            count += isinstance(value, dict) and value.get("verdict") == RETIRED_VERDICT
    if not count:
        return None
    return f"Warning: ignoring {count} '{RETIRED_VERDICT}' rows (a retired verdict) in {path}"


def append_verdict(finding: FindingSpec, row: VerdictRow) -> None:
    """Append a decision, validating its label against the current finding."""
    if row.label is not None and (
        finding.labels is None or row.label not in finding.labels
    ):
        raise ValueError(f"Unknown label {row.label!r} for finding {finding.slug}")
    with _verdicts_path(finding).open("a", encoding="utf-8") as stream:
        stream.write(row.model_dump_json() + "\n")


def append_judge_run(finding: FindingSpec, run: JudgeRun) -> None:
    """Record a judge run's counts beside the ledger."""
    with _judge_runs_path(finding).open("a", encoding="utf-8") as stream:
        stream.write(run.model_dump_json() + "\n")


def last_judge_run(finding: FindingSpec) -> JudgeRun | None:
    """The most recent judge run for this finding, if any."""
    path = _judge_runs_path(finding)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return JudgeRun.model_validate_json(lines[-1]) if lines else None


def effective_verdicts(rows: list[VerdictRow]) -> dict[str, VerdictRow]:
    """Select the latest human decision, otherwise the latest judge decision."""
    selected: dict[str, VerdictRow] = {}
    for row in rows:
        previous = selected.get(row.message_id)
        if (
            previous is None
            or row.decided_by.startswith("human:")
            or not previous.decided_by.startswith("human:")
        ):
            selected[row.message_id] = row
    return selected
