"""Agreement between a scan's citations and human message decisions."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from inspect_scout._recorder.recorder import Status

from .verdicts import VerdictRow

if TYPE_CHECKING:
    from .derive import Exhibit


class Agreement(BaseModel):
    """Message-level counts restricted to a scan's transcript population."""

    scan_id: str
    found_confirmed: int
    total_confirmed: int
    cited_rejected: int
    total_rejected: int

    @property
    def line(self) -> str:
        """Human-readable agreement, without treating unreviewed evidence as gold."""
        return (
            f"found {self.found_confirmed} of {self.total_confirmed} human-confirmed; "
            f"cited {self.cited_rejected} of {self.total_rejected} human-rejected"
        )


class JudgeAgreement(BaseModel):
    """How the judge's decisions fared once a human reviewed them."""

    kept: int = 0
    reviewed: int = 0
    """Messages with a judge decision followed by a human decision."""

    @property
    def overturned(self) -> int:
        return self.reviewed - self.kept

    @property
    def line(self) -> str:
        """Human-readable agreement: whether to trust the judge on this finding."""
        if not self.reviewed:
            return "judge agreement: no human-reviewed judge decisions yet"
        return (
            f"judge agreement: kept {self.kept} of {self.reviewed} "
            "human-reviewed judge decisions"
        )


def judge_agreement(rows: list[VerdictRow]) -> JudgeAgreement:
    """Compare each message's latest judge decision with the human decision after it.

    Rows are in append order. A later human row on the same message replaces an
    earlier one, so the human's current view is what the judge is measured
    against; a relabel keeps the judge's verdict and counts as kept.
    """
    judged: dict[str, VerdictRow] = {}
    reviewed: dict[str, tuple[VerdictRow, VerdictRow]] = {}
    for row in rows:
        if row.decided_by.startswith("judge:"):
            # Judge rows from older versions proposed; that is not a decision.
            if row.verdict in ("confirmed", "rejected"):
                judged[row.message_id] = row
        elif row.message_id in judged:
            reviewed[row.message_id] = (judged[row.message_id], row)
    return JudgeAgreement(
        kept=sum(judge.verdict == human.verdict for judge, human in reviewed.values()),
        reviewed=len(reviewed),
    )


def scan_agreement(
    status: Status, exhibits: list["Exhibit"], decisions: dict[str, VerdictRow]
) -> Agreement:
    """Compare distinct cited messages with effective human verdicts in coverage."""
    covered = (
        set(status.spec.transcripts.transcript_ids)
        if status.spec.transcripts
        else set()
    )
    confirmed: set[tuple[str, str]] = set()
    rejected: set[tuple[str, str]] = set()
    cited: set[tuple[str, str]] = set()
    for exhibit in exhibits:
        if exhibit.transcript_id not in covered:
            continue
        key = (exhibit.transcript_id, exhibit.message_id)
        if any(s.scan_id == status.spec.scan_id for s in exhibit.sightings):
            cited.add(key)
        row = decisions.get(exhibit.message_id)
        if row is not None and row.decided_by.startswith("human:"):
            if row.verdict == "confirmed":
                confirmed.add(key)
            elif row.verdict == "rejected":
                rejected.add(key)
    return Agreement(
        scan_id=status.spec.scan_id,
        found_confirmed=len(cited & confirmed),
        total_confirmed=len(confirmed),
        cited_rejected=len(cited & rejected),
        total_rejected=len(rejected),
    )
