"""Project findings, human review, and cited message context."""

import getpass
import re
from collections import Counter, defaultdict
from threading import Lock
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query
from inspect_ai._util._async import run_coroutine
from inspect_ai.model import ChatMessage
from pydantic import BaseModel, ConfigDict, Field
from rich.text import Text

from inspect_scout._findings.answer import AnswerCache, answer_headline, refresh_answer
from inspect_scout._findings.derive import (
    Exhibit,
    FindingView,
    derive_finding,
    exhibit_transcript,
)
from inspect_scout._findings.judge import (
    decide_exhibits,
    record_judge_run,
    reviewed_examples,
    select_candidates,
)
from inspect_scout._findings.links import view_url
from inspect_scout._findings.spec import (
    FindingSpec,
    finding_path,
    list_findings,
    read_finding,
)
from inspect_scout._findings.verdicts import JudgeRun, VerdictRow, append_verdict
from inspect_scout._project._project import read_project
from inspect_scout._project.types import ProjectConfig
from inspect_scout._scanner.extract import message_as_str

from ._server_common import InspectPydanticJSONResponse

FindingSlug = Annotated[str, Path(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]


class FindingStatus(BaseModel):
    """One status row for a standing question."""

    slug: str
    title: str
    states: dict[str, int]
    sightings: int
    coverage: dict[str, int]
    agreement: str
    judge_refused: int | None
    """Refusals in the finding's last judge run; `None` before any run."""
    judge_agreement: str
    """How many human-reviewed judge decisions the humans kept."""
    headline: str


class FindingDetail(FindingView):
    """Derived evidence with its materialized answer and navigation links."""

    status: FindingStatus
    answer_markdown: str
    answer_stale: bool
    """The synthesis predates the current evidence and awaits regeneration."""
    message_links: dict[str, str]
    citation_messages: dict[str, str]
    display_quotes: dict[str, str]


class FindingContextMessage(BaseModel):
    """An original message and the text used for citation verification."""

    message: ChatMessage
    index: int
    rendered: str


class FindingContext(BaseModel):
    """A bounded window around a located exhibit."""

    message_id: str
    transcript_id: str
    messages: list[FindingContextMessage]
    trust_content: bool | None


class FindingVerdictRequest(BaseModel):
    """A human decision; identity and append time are assigned by the server."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1)
    verdict: Literal["confirmed", "rejected"]
    label: str | None = None
    reason: str = "Reviewed in Scout View"


class FindingReviewResponse(BaseModel):
    """A saved decision, even when subsequent answer regeneration fails."""

    verdict: VerdictRow
    answer_error: str | None = None


class FindingJudgeRequest(BaseModel):
    """Options for judging a finding's pending exhibits."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    """Judge model; defaults to the project's findings model."""
    limit: int | None = Field(default=None, ge=1)
    """Judge at most this many exhibits, in transcript order."""


class FindingJudgeResponse(BaseModel):
    """A recorded judge run, even when subsequent answer regeneration fails."""

    run: JudgeRun
    skipped_unlocated: int
    answer_error: str | None = None


def _status(view: FindingView) -> FindingStatus:
    states = Counter(e.state for e in view.exhibits)
    coverage = Counter(view.coverage.get(view.finding.prompt_sha, {}).values())
    return FindingStatus(
        slug=view.finding.slug,
        title=view.finding.title,
        states={s: states[s] for s in ("proposed", "confirmed", "rejected")},
        sightings=sum(len(e.sightings) for e in view.exhibits),
        coverage={s: coverage[s] for s in ("judged", "partial", "missing")},
        agreement=view.agreements[-1].line if view.agreements else "no agreement yet",
        judge_refused=view.last_judge_run.refused if view.last_judge_run else None,
        judge_agreement=view.judge_agreement.line,
        headline=answer_headline(view.finding.slug),
    )


def _spec(slug: str) -> FindingSpec:
    try:
        return read_finding(slug)
    except FileNotFoundError as ex:
        raise HTTPException(404, "Finding not found") from ex
    except ValueError as ex:
        raise HTTPException(422, str(ex)) from ex


def _exhibit(view: FindingView, message_id: str) -> Exhibit:
    for exhibit in view.exhibits:
        if exhibit.message_id == message_id:
            return exhibit
    raise HTTPException(404, "Exhibit not found")


def _display_quote(quote: str, nonce: str | None) -> str:
    if nonce is None:
        return quote
    marker = re.escape(nonce)
    quote = re.sub(
        rf"(?:SYSTEM|USER|ASSISTANT|TOOL|Tool Call|Tool Use)\s*\({marker}\):?\s*",
        "",
        quote,
    )
    quote = re.sub(
        rf"</?(?:thinking(?:_summary|_redacted)?|prefill)\s+{marker}(?:\s*/?>|$)",
        "",
        quote,
    )
    quote = re.sub(rf"\[(?:BEGIN|END) '[^']+' \({marker}\)\]\s*", "", quote)
    tail = re.search(
        r"(?:SYSTEM|USER|ASSISTANT|TOOL|Tool Call|Tool Use)\s*\(([A-Za-z0-9]{8,})$",
        quote,
    )
    if tail and nonce.startswith(tail[1]):
        quote = quote[: tail.start()]
    return quote.strip()


def _detail(finding: FindingSpec, project: ProjectConfig) -> FindingDetail:
    view = derive_finding(finding, project)
    path = finding_path(finding.slug).with_name("answer.md")
    answer = path.read_text(encoding="utf-8") if path.exists() else ""
    citations = {}
    stale = False
    cache_match = re.search(r"<!-- findings-cache (.+) -->", answer)
    if cache_match:
        cache = AnswerCache.model_validate_json(cache_match[1])
        stale = cache.stale
        citations = {
            f"e{i}": message_id for i, message_id in enumerate(cache.evidence_ids, 1)
        }
    # Cache metadata is not report content; saved reports may name another viewer.
    answer = re.sub(r"\n<!-- findings-cache .+ -->\s*$", "", answer)
    answer = re.sub(r"\]\(https?://[^\s)]+/#/transcripts/", "](#/transcripts/", answer)
    links = {}
    for exhibit in view.exhibits:
        location = exhibit.transcripts_location or project.transcripts
        if exhibit.verified_ref and location:
            links[exhibit.message_id] = view_url(
                location, exhibit.transcript_id, exhibit.verified_ref, base_url=""
            ).removeprefix("/")
    return FindingDetail(
        **view.model_dump(),
        status=_status(view),
        answer_markdown=answer,
        answer_stale=stale,
        message_links=links,
        citation_messages=citations,
        display_quotes={
            e.message_id: _display_quote(e.quote, project.nonce) for e in view.exhibits
        },
    )


def create_findings_router() -> APIRouter:
    """Create project-scoped routes using the existing findings ledger."""
    router = APIRouter(
        tags=["findings"], default_response_class=InspectPydanticJSONResponse
    )
    locks: defaultdict[str, Lock] = defaultdict(Lock)

    # The ledger and report perform synchronous I/O. Plain endpoints keep that
    # work in FastAPI's threadpool; a finding's report cannot race its next edit.
    @router.get("/findings")
    def findings() -> list[FindingStatus]:
        project = read_project()
        rows = []
        for finding in list_findings():
            with locks[finding.slug]:
                rows.append(_status(derive_finding(finding, project)))
        return rows

    @router.get("/findings/{slug}")
    def finding(slug: FindingSlug) -> FindingDetail:
        with locks[slug]:
            return _detail(_spec(slug), read_project())

    @router.get("/findings/{slug}/context/{message_id}")
    def context(
        slug: FindingSlug,
        message_id: str,
        k: Annotated[int, Query(ge=0, le=100)] = 6,
    ) -> FindingContext:
        with locks[slug]:
            spec = _spec(slug)
            project = read_project()
            exhibit = _exhibit(derive_finding(spec, project), message_id)
            if exhibit.verified_ref is None:
                raise HTTPException(404, "Exhibit has no located message")
            try:
                transcript = run_coroutine(exhibit_transcript(exhibit, project))
            except FileNotFoundError as ex:
                raise HTTPException(404, str(ex)) from ex
            index = next(
                (
                    i
                    for i, message in enumerate(transcript.messages)
                    if message.id == exhibit.verified_ref
                ),
                None,
            )
            if index is None:
                raise HTTPException(404, "Message not found in transcript")
            start = max(0, index - k)
            return FindingContext(
                message_id=exhibit.verified_ref,
                transcript_id=transcript.transcript_id,
                trust_content=transcript.trust_content,
                messages=[
                    FindingContextMessage(
                        message=message,
                        index=i + 1,
                        rendered=message_as_str(message, nonce=project.nonce) or "",
                    )
                    for i, message in enumerate(
                        transcript.messages[start : index + k + 1], start
                    )
                ],
            )

    @router.post("/findings/{slug}/verdicts", status_code=201)
    def verdict(
        slug: FindingSlug, body: FindingVerdictRequest
    ) -> FindingReviewResponse:
        with locks[slug]:
            spec = _spec(slug)
            project = read_project()
            exhibit = _exhibit(derive_finding(spec, project), body.message_id)
            if exhibit.verified_ref is None and body.verdict == "confirmed":
                raise HTTPException(
                    422,
                    "An unlocated exhibit cannot be confirmed: its quote was not "
                    "found in any message. Reject it.",
                )
            row = VerdictRow(
                message_id=body.message_id,
                verdict=body.verdict,
                label=body.label if body.verdict == "confirmed" else None,
                decided_by=f"human:{getpass.getuser()}",
                reason=body.reason,
                source="manual",
            )
            try:
                append_verdict(spec, row)
            except ValueError as ex:
                raise HTTPException(422, str(ex)) from ex
            try:
                # Reviewers record many decisions in a row; synthesis (a paid
                # model call) waits for an explicit regenerate.
                run_coroutine(refresh_answer(spec, project, synthesize=False))
            except Exception as ex:
                # The append succeeded. Returning a failed mutation would invite
                # retries that duplicate the human decision in an append-only log.
                return FindingReviewResponse(
                    verdict=row, answer_error=Text.from_markup(str(ex)).plain
                )
            return FindingReviewResponse(verdict=row)

    @router.post("/findings/{slug}/judge")
    def judge(slug: FindingSlug, body: FindingJudgeRequest) -> FindingJudgeResponse:
        spec = _spec(slug)
        project = read_project()
        with locks[slug]:
            view = derive_finding(spec, project)
        candidates, unlocated = select_candidates(view, "proposed", body.limit)
        # Model calls run outside the lock so the page keeps loading meanwhile; a
        # human decision recorded in between still outranks the judge's.
        try:
            outcome = run_coroutine(
                decide_exhibits(
                    spec, project, candidates, reviewed_examples(view), model=body.model
                )
            )
        except Exception as ex:
            raise HTTPException(500, Text.from_markup(str(ex)).plain) from ex
        with locks[slug]:
            run = record_judge_run(spec, outcome, "proposed")
            try:
                run_coroutine(refresh_answer(spec, project))
            except Exception as ex:
                # The run is recorded; a retry would pay for the same calls again.
                return FindingJudgeResponse(
                    run=run,
                    skipped_unlocated=unlocated,
                    answer_error=Text.from_markup(str(ex)).plain,
                )
        return FindingJudgeResponse(run=run, skipped_unlocated=unlocated)

    @router.post("/findings/{slug}/answer")
    def answer(slug: FindingSlug) -> FindingDetail:
        with locks[slug]:
            spec = _spec(slug)
            project = read_project()
            try:
                run_coroutine(refresh_answer(spec, project, force=True))
            except Exception as ex:
                raise HTTPException(500, Text.from_markup(str(ex)).plain) from ex
            return _detail(spec, project)

    return router
