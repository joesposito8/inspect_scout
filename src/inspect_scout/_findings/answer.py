"""Computed finding reports and synthesis cached within answer.md."""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from inspect_ai.event import ModelEvent
from inspect_ai.model import Model
from pydantic import BaseModel, Field

from inspect_scout._project.types import ProjectConfig
from inspect_scout._transcript.factory import transcripts_from
from inspect_scout._transcript.types import Transcript, TranscriptInfo

from .derive import FindingView, derive_finding_async, exhibit_transcript
from .links import view_url
from .spec import FindingSpec, finding_path
from .synthesis import SYNTHESIS_PROMPT, check_synthesis, generate_synthesis


class AnswerCache(BaseModel):
    """Regeneration metadata stored with the report rather than in another ledger."""

    sha: str
    model: str
    synthesis: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    synthesis_ids: list[str] = Field(default_factory=list)
    """Message ids that the synthesis's [E<n>] citations refer to, in order."""
    stale: bool = False
    """The synthesis predates the current evidence and awaits regeneration."""


def markdown_text(value: str) -> str:
    """Render transcript-derived text literally in generated Markdown."""
    return re.sub(r"([\\`*_{}\[\]()<>#+!|])", r"\\\1", value).replace("\n", " ")


def _synthesis_markdown(text: str, ids: set[str]) -> str:
    lines = []
    for line in text.splitlines():
        escaped = markdown_text(line)
        for eid in ids:
            escaped = escaped.replace(f"\\[{eid}\\]", f"[\\[{eid}\\]](#{eid.lower()})")
        lines.append(escaped)
    return "\n\n".join(lines)


def _remap_citations(
    text: str, previous_ids: list[str], current: dict[str, str]
) -> str:
    """Point a kept synthesis's [E<n>] citations at the same messages' current ids."""
    current_ids = {message_id: eid for eid, message_id in current.items()}

    def replace(match: re.Match[str]) -> str:
        index = int(match[1]) - 1
        if not 0 <= index < len(previous_ids):
            # An invented citation stays visible to the citation check.
            return match[0]
        eid = current_ids.get(previous_ids[index])
        return f"[{eid}]" if eid else "[no longer evidence]"

    return re.sub(r"\[E(\d+)\]", replace, text)


def answer_headline(slug: str) -> str:
    """Read the materialized headline without invoking a model."""
    path = finding_path(slug).with_name("answer.md")
    if not path.exists():
        return "answer pending"
    for line in path.read_text(encoding="utf-8").splitlines():
        if "ANSWER:" in line and not line.startswith("<!--"):
            return line.strip()
    return "computed report available"


async def refresh_answer(
    finding: FindingSpec,
    project: ProjectConfig,
    *,
    model: str | Model | None = None,
    no_synthesis: bool = False,
    force: bool = False,
    synthesize: bool = True,
    view: FindingView | None = None,
) -> Path:
    """Refresh computed facts on every call and synthesize only when evidence changes.

    With `synthesize=False` (single review decisions), changed evidence keeps the
    previous synthesis, marked stale, until a later refresh regenerates it.
    """
    view = view or await derive_finding_async(finding, project)
    evidence = sorted(
        view.evidence,
        key=lambda e: (
            e.task_id or e.transcript_id,
            e.transcript_id,
            e.message_index or 10**12,
            e.message_id,
        ),
    )
    exhibits = {f"E{i}": e for i, e in enumerate(evidence, 1)}
    infos: dict[str, TranscriptInfo] = {}
    if project.transcripts:
        async with transcripts_from(project.transcripts).reader() as reader:
            infos = {info.transcript_id: info async for info in reader.index()}
    transcripts: dict[str, Transcript] = {}
    timestamps: dict[str, datetime] = {}
    for exhibit in evidence:
        if exhibit.verified_ref is None or exhibit.transcript_id in transcripts:
            continue
        transcript = await exhibit_transcript(exhibit, project, events=True)
        transcripts[exhibit.transcript_id] = transcript
        infos.setdefault(exhibit.transcript_id, transcript)
        for event in transcript.events:
            if isinstance(event, ModelEvent):
                for choice in event.output.choices:
                    if choice.message.id:
                        timestamps[choice.message.id] = event.timestamp
    coverage = view.coverage.get(finding.prompt_sha, {})
    states = Counter(e.state for e in view.exhibits)
    labels = Counter(e.label or "unlabeled" for e in evidence)
    source_counts = Counter(
        (infos[e.transcript_id].source_id or e.transcript_id)
        if e.transcript_id in infos
        else e.transcript_id
        for e in evidence
    )
    awaiting = view.awaiting_review
    facts = {
        "evidence exhibits": len(evidence),
        "confirmed exhibits": states["confirmed"],
        "awaiting review: proposed exhibits": states["proposed"],
        "excluded: rejected exhibits": states["rejected"],
        "covered transcripts": sum(v != "missing" for v in coverage.values()),
        "fully scanned transcripts": sum(v == "judged" for v in coverage.values()),
        "total transcripts": len(coverage),
        **{f"label {label}": count for label, count in sorted(labels.items())},
        **{
            f"source {source}": count for source, count in sorted(source_counts.items())
        },
    }
    agents = {tid: (info.task_id or tid) for tid, info in infos.items()}
    for exhibit in view.exhibits:
        agents.setdefault(
            exhibit.transcript_id, exhibit.task_id or exhibit.transcript_id
        )
    per_agent = Counter(e.transcript_id for e in evidence)
    per_agent_awaiting = Counter(e.transcript_id for e in awaiting)
    facts.update(
        {f"agent {agents.get(tid, tid)}": per_agent[tid] for tid in sorted(coverage)}
    )
    lines = [
        f"# {markdown_text(finding.title)}",
        "",
        markdown_text(finding.prompt),
        "",
        "## Computed",
        "",
        "Exists: "
        + (
            "yes, among confirmed exhibits"
            if evidence
            else f"not established: no exhibits are confirmed yet ({len(awaiting)} proposed await review)"
            if awaiting
            else "none found within coverage"
        )
        + ".",
        "",
    ]
    dated = [
        (timestamps[e.message_id], eid)
        for eid, e in exhibits.items()
        if e.message_id in timestamps
    ]
    if dated:
        timestamp, eid = min(dated)
        lines.append(
            f"Earliest timestamped evidence: {timestamp.isoformat()} [{eid}](#{eid.lower()}); an upper bound given coverage."
        )
    else:
        lines.append(
            "Earliest: no matching model-event timestamp; use each agent's message-index fallback below."
        )
    if any(e.message_id not in timestamps for e in evidence):
        lines.append(
            "Some evidence has no timestamp; message indices are comparable only within an agent, so earlier untimed evidence cannot be ruled out."
        )
    lines += [
        "",
        "Evidence exhibits = confirmed exhibits. Proposed exhibits await review; rejected exhibits are excluded.",
        "",
        *[f"- {markdown_text(key)}: {value}" for key, value in facts.items()],
        "",
        "## Agents",
        "",
        "| Agent | Confirmed | Proposed | Coverage | Earliest |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for tid in sorted(
        set(agents) | set(coverage), key=lambda tid: (agents.get(tid, tid), tid)
    ):
        members = [(eid, e) for eid, e in exhibits.items() if e.transcript_id == tid]
        times = [
            (timestamps[e.message_id], eid)
            for eid, e in members
            if e.message_id in timestamps
        ]
        indexed = [
            (e.message_index, eid) for eid, e in members if e.message_index is not None
        ]
        earliest = (
            min(times)[0].isoformat()
            if times
            else f"#{min(indexed)[0]} (index fallback)"
            if indexed
            else "—"
        )
        state = coverage.get(tid, "missing")
        lines.append(
            f"| {markdown_text(agents.get(tid, tid))} | {per_agent[tid]} | {per_agent_awaiting[tid]} | {'not scanned' if state == 'missing' else state} | {earliest} |"
        )
    lines += [
        "",
        "## Coverage and review",
        "",
        "Coverage is scoped to the current finding prompt. Partial coverage means some scan segments of that transcript were refused, unparsed or failed.",
        f"Review: {states['confirmed']} confirmed; {states['proposed']} proposed; {states['rejected']} rejected"
        + (
            f"; {view.last_judge_run.refused} refused by judge."
            if view.last_judge_run
            else "."
        ),
        "",
        f"{view.judge_agreement.line.capitalize()}.",
    ]
    if view.agreements:
        lines += [
            "",
            f"Latest agreement ({view.agreements[-1].scan_id}): {view.agreements[-1].line}.",
        ]
    if awaiting:
        lines += ["", "Awaiting review:", ""]
        for e in awaiting:
            location = e.transcripts_location or project.transcripts
            link = (
                f" [Open message]({view_url(location, e.transcript_id, e.verified_ref)})"
                if e.verified_ref and location
                else ""
            )
            lines.append(
                f"- {markdown_text(e.task_id or e.transcript_id)} #{e.message_index or '?'}: {markdown_text(e.label or 'unlabeled')}; {markdown_text(e.message_id)}; {markdown_text(e.quote)}{link}"
            )
    excluded = [e for e in view.exhibits if e.state == "rejected"]
    if excluded:
        lines += ["", "Excluded from the evidence set:", ""]
        lines += [
            f"- {markdown_text(e.task_id or e.transcript_id)} #{e.message_index or '?'}: {e.state}; {markdown_text(e.message_id)}; {markdown_text(e.verdict_reason or e.reason)}"
            for e in excluded
        ]
    lines += ["", "## Exhibits", ""]
    previous_agent = None
    for eid, exhibit in exhibits.items():
        agent = exhibit.task_id or exhibit.transcript_id
        if agent != previous_agent:
            lines += [f"### {markdown_text(agent)}", ""]
            previous_agent = agent
        lines += [
            f'<a id="{eid.lower()}"></a>',
            f"#### {eid}",
            "",
            f"[{exhibit.message_label or '?'}] | #{exhibit.message_index or '?'} | {markdown_text(exhibit.label or 'unlabeled')} | {exhibit.state} | {exhibit.verification}",
            "",
            f"> {markdown_text(exhibit.quote)}",
            "",
            markdown_text(exhibit.reason),
            "",
            f"Decision: {markdown_text(exhibit.decided_by or 'finding scanner')}; {markdown_text(exhibit.verdict_reason or '')}",
            f"Sightings: {len(exhibit.sightings)} in {exhibit.corroboration} distinct scans.",
        ]
        location = exhibit.transcripts_location or project.transcripts
        if exhibit.verified_ref and location:
            lines += [
                f"[Open message]({view_url(location, exhibit.transcript_id, exhibit.verified_ref)})"
            ]
        lines += [""]
    computed = "\n".join(lines)
    stable = [
        {
            "transcript": e.transcript_id,
            "message": e.message_id,
            "quote": e.quote,
            "label": e.label,
            "state": e.state,
        }
        for e in sorted(evidence, key=lambda e: (e.transcript_id, e.message_id))
    ]
    # The synthesis instructions are part of the key, so revising them
    # regenerates answers on their next refresh.
    digest = hashlib.sha256(
        json.dumps(
            [finding.prompt_sha, SYNTHESIS_PROMPT, stable], sort_keys=True
        ).encode()
    ).hexdigest()
    selected_model = model or project.findings.model
    model_name = str(selected_model)
    path = finding_path(finding.slug).with_name("answer.md")
    previous = None
    if path.exists():
        match = re.search(
            r"<!-- findings-cache (.+) -->", path.read_text(encoding="utf-8")
        )
        if match:
            try:
                previous = AnswerCache.model_validate_json(match[1])
            except ValueError:
                previous = None
    cache = AnswerCache(
        sha=digest, model=model_name, evidence_ids=[e.message_id for e in evidence]
    )
    reuse = (
        previous is not None
        and previous.sha == digest
        and previous.model == model_name
        and not force
    )
    if previous and (reuse or (not synthesize and previous.synthesis)):
        cache.synthesis = previous.synthesis
        # Caches written before synthesis_ids existed numbered by evidence_ids.
        cache.synthesis_ids = previous.synthesis_ids or previous.evidence_ids
        if not reuse:
            # Keep the previous digest so the next synthesizing refresh replaces it.
            cache.sha, cache.model, cache.stale = previous.sha, previous.model, True
    if no_synthesis:
        cache.synthesis = ""
    elif not evidence:
        cache.synthesis = (
            f"ANSWER: No confirmed exhibits yet; {len(awaiting)} proposed exhibits await review."
            if awaiting
            else "ANSWER: None found within coverage."
        )
        cache.stale = False
    # Only located exhibits can be cited: an unlocated quote matches no message.
    located = {eid: e for eid, e in exhibits.items() if e.verified_ref is not None}
    # A failed generation must leave the last complete report and its citation
    # map intact. Publish the replacement only after synthesis succeeds.
    if synthesize and not no_synthesis and not cache.synthesis:
        reasons = {
            reason: f"R{i}"
            for i, reason in enumerate(
                dict.fromkeys(e.reason for e in evidence if len(e.reason) > 1000), 1
            )
        }
        exhibit_text = "\n".join(
            f"[{eid}] agent {e.task_id or e.transcript_id} #{e.message_index} ({e.label or 'unlabeled'}, {e.state}) {json.dumps(e.quote)} — source reason: {reasons.get(e.reason, e.reason)}; decision reason: {e.verdict_reason or 'none'}"
            for eid, e in located.items()
        )
        prompt = (
            "<facts>\nEvidence exhibits = confirmed exhibits. Proposed exhibits await review; rejected exhibits are excluded from evidence.\n"
            f"Evidence exhibits whose quote was not found in any message, and which are therefore not listed: {len(exhibits) - len(located)}.\n"
            f"{json.dumps(facts, ensure_ascii=False)}\nNamed facts:\n"
            + "\n".join(f"{key}: {value}" for key, value in facts.items())
            + "\n</facts>\n<source_reasons>\n"
            + json.dumps({ref: reason for reason, ref in reasons.items()})
            + f"\n</source_reasons>\n<exhibits>\n{exhibit_text}\n</exhibits>\n<finding>\n{finding.prompt}\n</finding>\nAnswer using only these exhibits, cite exhibit ids, name each cited exhibit's agent, and use the given named facts."
        )
        cache.synthesis = await generate_synthesis(prompt, selected_model)
        cache.synthesis_ids = cache.evidence_ids
    checked = check_synthesis(
        _remap_citations(
            cache.synthesis,
            cache.synthesis_ids,
            {eid: e.message_id for eid, e in exhibits.items()},
        ),
        set(located),
        facts,
        list(agents.values()) + list(labels) + list(source_counts),
        unlocated_ids=set(exhibits) - set(located),
        exhibit_texts={
            eid: " ".join([e.quote, e.reason, e.verdict_reason or ""])
            for eid, e in exhibits.items()
        },
    )
    cache.references = sorted(
        {
            ref
            for eid in checked.cited_ids
            if (ref := exhibits[eid].verified_ref) is not None
        }
    )
    metadata = cache.model_dump_json().replace("<", "\\u003c").replace(">", "\\u003e")
    # The notice sits outside the synthesis section so it is never read as model text.
    notice = ""
    if no_synthesis:
        content = "Synthesis disabled; computed evidence above."
    elif not cache.synthesis:
        content = "Synthesis pending: regenerate the answer to summarize the evidence."
    else:
        content = _synthesis_markdown(checked.text, set(exhibits))
        if cache.stale:
            notice = (
                "> **Out of date:** review decisions have changed the evidence since "
                "the synthesis below was written. Regenerate the answer to update it.\n\n"
            )
    path.write_text(
        f"{computed}\n\n{notice}## Synthesis\n\n{content}\n\n<!-- findings-cache {metadata} -->\n",
        encoding="utf-8",
    )
    return path
