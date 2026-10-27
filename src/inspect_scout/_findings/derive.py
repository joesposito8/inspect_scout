"""Exhibits and coverage derived from immutable scan results."""

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from threading import Lock, get_ident
from typing import Literal, TypeVar

import pandas as pd
from inspect_ai._util._async import run_coroutine
from inspect_ai._util.file import dirname
from pydantic import BaseModel, Field, ValidationError

from inspect_scout._llm_scanner.rendering import ScannerRendering
from inspect_scout._project._project import read_project
from inspect_scout._project.types import ProjectConfig
from inspect_scout._recorder.recorder import Status
from inspect_scout._scanlist import scan_list_async
from inspect_scout._scanner.result import Result
from inspect_scout._scanresults import scan_results_df_async
from inspect_scout._scanspec import ScannerSpec
from inspect_scout._transcript.factory import (
    transcripts_from,
    transcripts_from_snapshot,
)
from inspect_scout._transcript.transcripts import TranscriptsReader
from inspect_scout._transcript.types import (
    Transcript,
    TranscriptContent,
    TranscriptInfo,
)

from .agreement import Agreement, JudgeAgreement, judge_agreement, scan_agreement
from .spec import FindingSpec
from .verdicts import JudgeRun, effective_verdicts, last_judge_run, read_verdicts
from .verify import QuoteVerifier, VerifiedQuote, excerpt

CoverageStatus = Literal["judged", "partial", "missing"]


class Sighting(BaseModel):
    """One scanner's observation of a message in one scan."""

    scan_id: str
    scanner: str
    model: str | None
    label: str | None
    result_uuid: str
    model_refs: list[str]
    verification: VerifiedQuote


class Exhibit(BaseModel):
    """Evidence grouped by original message identity."""

    message_id: str
    transcript_id: str
    task_id: str | None = None
    transcripts_location: str | None = None
    model_ref: str | None
    verified_ref: str | None
    verification: Literal["verified", "relocated", "unlocated"]
    message_index: int | None
    message_label: str | None
    quote: str
    reason: str
    label: str | None
    state: Literal["proposed", "confirmed", "rejected"] = "proposed"
    sightings: list[Sighting] = Field(default_factory=list)
    decided_by: str | None = None
    verdict_reason: str | None = None

    @property
    def corroboration(self) -> int:
        """Number of distinct scans citing this message."""
        return len({sighting.scan_id for sighting in self.sightings})


class FindingView(BaseModel):
    """Materialized evidence and prompt-version coverage for a finding."""

    finding: FindingSpec
    exhibits: list[Exhibit] = Field(default_factory=list)
    coverage: dict[str, dict[str, CoverageStatus]] = Field(default_factory=dict)
    scans: list[str] = Field(default_factory=list)
    agreements: list[Agreement] = Field(default_factory=list)
    last_judge_run: JudgeRun | None = None
    judge_agreement: JudgeAgreement = Field(default_factory=JudgeAgreement)

    @property
    def evidence(self) -> list[Exhibit]:
        """Confirmed exhibits: the evidence the finding's answer is built from."""
        return [e for e in self.exhibits if e.state == "confirmed"]

    @property
    def awaiting_review(self) -> list[Exhibit]:
        """Proposed exhibits that nobody has confirmed or rejected yet."""
        return [e for e in self.exhibits if e.state == "proposed"]


class _Contribution(BaseModel):
    exhibits: list[Exhibit] = Field(default_factory=list)
    coverage: dict[str, dict[str, CoverageStatus]] = Field(default_factory=dict)
    # Identities a located exhibit would have had while its quote was unlocated,
    # so verdicts written then still apply.
    aliases: dict[str, str] = Field(default_factory=dict)


class _Stored(BaseModel):
    version: int = Field(default_factory=lambda: DERIVATION_VERSION)


class _StoredScan(_Stored):
    contributions: dict[str, _Contribution]


StoredT = TypeVar("StoredT", bound=_Stored)


# Bump when the derivation of an unchanged scan would change, so persisted
# results from older code are recomputed rather than reused.
DERIVATION_VERSION = 2

# Only completed scans enter this cache; active/resumed scans are read again.
_scan_cache: dict[str, dict[str, _Contribution]] = {}

# Rendering and normalizing a long transcript dominates derivation, and every
# scanner, scan and finding verifies quotes against the same few transcripts.
_VERIFIER_CACHE_SIZE = 16
_verifiers: OrderedDict[tuple[str, str, str], QuoteVerifier] = OrderedDict()
_verifiers_lock = Lock()


def _unlocated_id(
    slug: str, transcript_id: str, quote: str, model_ref: str | None
) -> str:
    payload = [slug, transcript_id, " ".join(quote.split()).casefold(), model_ref]
    return "unlocated:" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()


async def _verifier(
    reader: TranscriptsReader,
    info: TranscriptInfo,
    location: str | None,
    rendering: ScannerRendering,
) -> QuoteVerifier:
    key = (
        location or info.source_uri or "",
        info.transcript_id,
        rendering.model_dump_json(),
    )
    with _verifiers_lock:
        if key in _verifiers:
            _verifiers.move_to_end(key)
            return _verifiers[key]
    transcript = await reader.read(info, TranscriptContent(messages="all"))
    verifier = QuoteVerifier(transcript, rendering)
    await verifier.initialize()
    with _verifiers_lock:
        _verifiers[key] = verifier
        while len(_verifiers) > _VERIFIER_CACHE_SIZE:
            _verifiers.popitem(last=False)
    return verifier


def _stored_path(name: str) -> Path:
    return Path.cwd() / "findings" / ".derived" / name


def _load_stored(model: type[StoredT], name: str) -> StoredT | None:
    path = _stored_path(name)
    if not path.exists():
        return None
    try:
        stored = model.model_validate_json(path.read_bytes())
    except (ValidationError, OSError):
        return None
    return stored if stored.version == DERIVATION_VERSION else None


def _save_stored(name: str, value: _Stored) -> None:
    path = _stored_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent derivations of the same scan write identical content; replacing
    # atomically keeps a reader from seeing a partial file.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.tmp")
    temporary.write_text(value.model_dump_json(), encoding="utf-8")
    os.replace(temporary, path)


async def exhibit_transcript(
    exhibit: Exhibit,
    project: ProjectConfig,
    *,
    events: bool = False,
) -> Transcript:
    """Read the original transcript, including scan snapshots of other sources."""
    from inspect_scout._query import Column

    sources = []
    if project.transcripts is not None:
        sources.append(transcripts_from(project.transcripts))
    for status in await scan_list_async(project.scans or "./scans"):
        if (
            status.spec.transcripts
            and exhibit.transcript_id in status.spec.transcripts.transcript_ids
        ):
            sources.append(await transcripts_from_snapshot(status.spec.transcripts))
    for source in sources:
        async with source.where(
            Column("transcript_id") == exhibit.transcript_id
        ).reader() as reader:
            async for info in reader.index():
                return await reader.read(
                    info,
                    TranscriptContent(
                        messages="all", events=["model"] if events else None
                    ),
                )
    raise FileNotFoundError(
        f"Transcript {exhibit.transcript_id} for message {exhibit.message_id} is unavailable"
    )


def _items(result: Result) -> Iterator[Result]:
    if result.type == "resultset" and isinstance(result.value, list):
        for value in result.value:
            yield from _items(Result.model_validate(value))
    else:
        yield result


def _scanner_findings(name: str, spec: ScannerSpec) -> dict[str, str]:
    if name == "finding_bundle":
        # Recorded specs keep fields later versions dropped, so only these two are read.
        return {
            finding["slug"]: finding["prompt_sha"]
            for finding in spec.params.get("findings", [])
        }
    if name.startswith("finding_") and isinstance(spec.params.get("prompt_sha"), str):
        return {
            name.removeprefix("finding_").replace("_", "-"): spec.params["prompt_sha"]
        }
    return {}


async def _derive_scan(status: Status) -> dict[str, _Contribution]:
    stored_name = f"{status.spec.scan_id}.findings.json"
    if status.complete:
        if status.spec.scan_id in _scan_cache:
            return _scan_cache[status.spec.scan_id]
        stored = _load_stored(_StoredScan, stored_name)
        if stored is not None:
            _scan_cache[status.spec.scan_id] = stored.contributions
            return stored.contributions
    contributions: dict[str, _Contribution] = {}
    snapshot = status.spec.transcripts
    if snapshot is None:
        return contributions
    scanners = {
        name: (spec, _scanner_findings(name, spec))
        for name, spec in status.spec.scanners.items()
        if _scanner_findings(name, spec)
    }
    if not scanners:
        return contributions
    transcripts = await transcripts_from_snapshot(snapshot)
    async with transcripts.reader() as reader:
        infos = {info.transcript_id: info async for info in reader.index()}
        for name, (spec, findings) in scanners.items():
            rendering = ScannerRendering.model_validate(
                spec.params.get("rendering") or {}
            )
            model_value = spec.params.get("model")
            model = (
                model_value
                if isinstance(model_value, str)
                else model_value.get("model")
                if isinstance(model_value, dict)
                else status.spec.model.model
                if status.spec.model
                else None
            )
            for slug, sha in findings.items():
                contributions.setdefault(slug, _Contribution()).coverage[sha] = {
                    transcript_id: "missing"
                    for transcript_id in snapshot.transcript_ids
                }
            try:
                results = await scan_results_df_async(
                    status.location, scanner=name, rows="transcripts"
                )
                if name not in results.scanners:
                    continue
                frame = results.scanners[name]
            except FileNotFoundError:
                # Active scans publish their spec before their first result file.
                if status.complete:
                    raise
                continue
            for _, row in frame.iterrows():
                transcript_id = str(row["transcript_id"])
                if pd.notna(row.get("scan_error")):
                    coverage: CoverageStatus = "partial"
                    items: list[Result] = []
                elif row.get("value_type") == "resultset" and isinstance(
                    row["value"], str
                ):
                    root = Result(value=json.loads(row["value"]), type="resultset")
                    items = list(_items(root))
                    coverage = (
                        "partial"
                        if any(item.value is None for item in items)
                        else "judged"
                    )
                else:
                    items = []
                    coverage = "partial"
                for slug, sha in findings.items():
                    contributions[slug].coverage[sha][transcript_id] = coverage
                if not items or transcript_id not in infos:
                    continue
                verifier = await _verifier(
                    reader, infos[transcript_id], snapshot.location, rendering
                )
                transcript = verifier.transcript
                for item in items:
                    if not isinstance(item.value, dict):
                        continue
                    quote = item.value.get("quote")
                    if not isinstance(quote, str):
                        continue
                    slug_value = (
                        item.value.get("finding")
                        if name == "finding_bundle"
                        else next(iter(findings))
                    )
                    if not isinstance(slug_value, str) or slug_value not in findings:
                        continue
                    verification = verifier.verify(quote, item.references)
                    result_uuid = item.uuid or str(row["uuid"])
                    sighting = Sighting(
                        scan_id=status.spec.scan_id,
                        scanner=name,
                        model=model,
                        label=item.label,
                        result_uuid=result_uuid,
                        model_refs=[
                            ref.id for ref in item.references if ref.type == "message"
                        ],
                        verification=verification,
                    )
                    unlocated = _unlocated_id(
                        slug_value, transcript_id, quote, verification.model_ref
                    )
                    if verification.verified_ref is not None:
                        contributions[slug_value].aliases[unlocated] = (
                            verification.verified_ref
                        )
                    contributions[slug_value].exhibits.append(
                        Exhibit(
                            message_id=verification.verified_ref or unlocated,
                            transcript_id=transcript_id,
                            task_id=transcript.task_id,
                            transcripts_location=snapshot.location
                            or (
                                dirname(transcript.source_uri)
                                if transcript.source_uri
                                else None
                            ),
                            model_ref=verification.model_ref,
                            verified_ref=verification.verified_ref,
                            verification=verification.status,
                            message_index=verification.message_index,
                            message_label=verification.message_label,
                            quote=excerpt(quote, rendering.nonce),
                            reason=item.explanation or "",
                            label=item.label,
                            sightings=[sighting],
                        )
                    )
    if status.complete:
        _scan_cache[status.spec.scan_id] = contributions
        _save_stored(stored_name, _StoredScan(contributions=contributions))
    return contributions


async def derive_finding_async(
    finding: FindingSpec, project: ProjectConfig | None = None
) -> FindingView:
    """Derive a finding from the scans currently present in its project.

    Args:
        finding: Registered standing question.
        project: Project configuration; defaults to the current project.

    Returns:
        Located exhibits, unlocated results, sightings, and coverage by prompt hash.
    """
    project = project or read_project()
    view = FindingView(finding=finding)
    all_ids: set[str] = set()
    if project.transcripts is not None:
        async with transcripts_from(project.transcripts).reader() as reader:
            all_ids = {info.transcript_id async for info in reader.index()}
    statuses = await scan_list_async(project.scans or "./scans")
    grouped: dict[tuple[str, str], Exhibit] = {}
    aliases: dict[str, str] = {}
    ranks = {"missing": 0, "partial": 1, "judged": 2}
    for status in sorted(
        statuses, key=lambda value: (value.spec.timestamp, value.spec.scan_id)
    ):
        contribution = (await _derive_scan(status)).get(finding.slug)
        if contribution is None:
            continue
        view.scans.append(status.spec.scan_id)
        for sha, coverage in contribution.coverage.items():
            combined = view.coverage.setdefault(sha, {})
            for transcript_id, state in coverage.items():
                all_ids.add(transcript_id)
                if ranks[state] >= ranks[combined.get(transcript_id, "missing")]:
                    combined[transcript_id] = state
        aliases.update(contribution.aliases)
        for exhibit in contribution.exhibits:
            for sighting in exhibit.sightings:
                aliases[f"unlocated:{sighting.result_uuid}"] = exhibit.message_id
            key = (exhibit.transcript_id, exhibit.message_id)
            if key not in grouped:
                grouped[key] = exhibit.model_copy(deep=True)
            else:
                existing = grouped[key]
                seen = {
                    (s.scan_id, s.scanner, s.result_uuid) for s in existing.sightings
                }
                existing.sightings.extend(
                    s
                    for s in exhibit.sightings
                    if (s.scan_id, s.scanner, s.result_uuid) not in seen
                )
    view.coverage.setdefault(finding.prompt_sha, {})
    for coverage in view.coverage.values():
        for transcript_id in sorted(all_ids):
            coverage.setdefault(transcript_id, "missing")
    # Older ledgers used per-result UUIDs. Resolve them before precedence so the
    # latest human row still wins when several historical copies collapse.
    ledger = [
        row.model_copy(
            update={"message_id": aliases.get(row.message_id, row.message_id)}
        )
        for row in read_verdicts(finding)
    ]
    decisions = effective_verdicts(ledger)
    view.judge_agreement = judge_agreement(ledger)
    manual_ids = {
        message_id
        for message_id, row in decisions.items()
        if row.source == "manual"
        and not message_id.startswith("unlocated:")
        and not any(key[1] == message_id for key in grouped)
    }
    if manual_ids and project.transcripts is not None:
        rendering = ScannerRendering(
            messages=project.findings.messages, nonce=project.nonce
        )
        async with transcripts_from(project.transcripts).reader() as reader:
            async for info in reader.index():
                verifier = await _verifier(reader, info, project.transcripts, rendering)
                for message_id in manual_ids.intersection(verifier.texts):
                    grouped[(info.transcript_id, message_id)] = Exhibit(
                        message_id=message_id,
                        transcript_id=info.transcript_id,
                        task_id=info.task_id,
                        transcripts_location=project.transcripts,
                        model_ref=None,
                        verified_ref=message_id,
                        verification="verified",
                        message_index=verifier.indices[message_id],
                        message_label=verifier.labels[message_id],
                        quote=verifier.excerpt(message_id),
                        reason=decisions[message_id].reason[:1000],
                        label=None,
                    )
    view.exhibits = list(grouped.values())
    for exhibit in view.exhibits:
        if row := decisions.get(exhibit.message_id):
            exhibit.state = row.verdict
            exhibit.label = row.label
            exhibit.decided_by = row.decided_by
            exhibit.verdict_reason = row.reason
    view.last_judge_run = last_judge_run(finding)
    view.agreements = [
        scan_agreement(status, view.exhibits, decisions)
        for status in statuses
        if status.spec.scan_id in view.scans
    ]
    return view


def derive_finding(
    finding: FindingSpec, project: ProjectConfig | None = None
) -> FindingView:
    """Synchronously derive a finding from the current project's scans."""
    return run_coroutine(derive_finding_async(finding, project))
