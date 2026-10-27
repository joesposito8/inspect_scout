import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_scout import (
    Reference,
    Result,
    Scanner,
    ScannerRendering,
    Transcript,
    scan_complete,
    scanner,
)
from inspect_scout._findings.answer import refresh_answer
from inspect_scout._findings.derive import Exhibit, derive_finding, exhibit_transcript
from inspect_scout._findings.scanner import finding_scanner
from inspect_scout._findings.spec import FindingSpec, register_finding
from inspect_scout._findings.synthesis import check_synthesis
from inspect_scout._findings.verdicts import VerdictRow, append_verdict
from inspect_scout._project._project import write_project_config
from inspect_scout._project.types import ProjectConfig

from .test_derive import run_finding, run_scan, setup_project
from .test_judge import FORGED, scan_all
from .test_scanner import Capture


class SynthesisCapture:
    def __init__(self) -> None:
        self.requests: list[list[ChatMessage]] = []
        self.text = "The agent admitted falsification [E1]. evidence exhibits: 1.\nANSWER: Supported [E1]."

    def __call__(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.requests.append(messages)
        return ModelOutput.from_content("mockllm/model", self.text)


def confirm(spec: FindingSpec, message_id: str) -> None:
    """Record a human confirmation, the only way an exhibit becomes evidence."""
    append_verdict(
        spec,
        VerdictRow(
            message_id=message_id,
            verdict="confirmed",
            decided_by="human:reviewer",
            reason="Confirmed in review",
            source="manual",
        ),
    )


def test_answer_is_built_from_confirmed_exhibits_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    run_finding(
        project, spec, [FORGED, {"quote": "I rejected cheating", "reason": "[M3]"}]
    )
    capture = SynthesisCapture()
    model = get_model("mockllm/confirmed-only", custom_outputs=capture, memoize=False)
    path = asyncio.run(refresh_answer(spec, project, model=model))
    text = path.read_text()
    assert not capture.requests
    assert (
        "ANSWER: No confirmed exhibits yet; 2 proposed exhibits await review." in text
    )
    assert "Awaiting review:" in text and "#### E1" not in text
    confirm(spec, "t1-3")
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 1
    exhibits = capture.requests[0][-1].text.split("<exhibits>")[1]
    assert "I rejected cheating" in exhibits and "I forged PASS" not in exhibits
    assert "- confirmed exhibits: 1" in path.read_text()


def test_rotated_nonce_preserves_evidence_and_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    run_finding(project, spec, [FORGED])
    confirm(spec, "t1-2")
    # Start from an empty synthesis cache rather than the scan's own refresh.
    (tmp_path / "findings/gamed/answer.md").unlink()
    # A human decision on a message the scanner did not cite adds a manual exhibit.
    append_verdict(
        spec,
        VerdictRow(
            message_id="t1-3",
            verdict="confirmed",
            decided_by="human:reviewer",
            reason="Verified manually",
            source="manual",
        ),
    )
    capture = SynthesisCapture()
    model = get_model("mockllm/nonce-regression", custom_outputs=capture, memoize=False)
    before = derive_finding(spec, project)
    assert len(before.evidence) == 2
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 1
    project.nonce = "RotatedNonce456"
    write_project_config(project)
    after = derive_finding(spec, project)
    assert [e.quote for e in after.evidence] == [e.quote for e in before.evidence]
    assert all(len(e.quote) <= 200 and "Nonce" not in e.quote for e in after.exhibits)
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 1
    # Explanatory prose is not evidence; decisions still invalidate the cache.
    append_verdict(
        spec,
        VerdictRow(
            message_id="t1-3",
            verdict="confirmed",
            decided_by="human:reviewer",
            reason="Expanded review explanation",
            source="manual",
        ),
    )
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 1
    append_verdict(
        spec,
        VerdictRow(
            message_id="t1-3",
            verdict="rejected",
            decided_by="human:reviewer",
            reason="Correction",
            source="manual",
        ),
    )
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 2


def test_answer_refresh_and_unchanged_evidence_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    scanner_model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [dict(quote="I forged PASS", reason="[M2] admits it", label=None)]
        ),
        memoize=False,
    )
    scanners: dict[str, Scanner[Transcript]] = {
        spec.scanner_name: finding_scanner(
            spec.prompt,
            None,
            spec.prompt_sha,
            rendering=ScannerRendering(nonce=project.nonce),
            model=scanner_model,
        )
    }
    run_scan(project, scanners, scanner_model)
    path = tmp_path / "findings/gamed/answer.md"
    assert path.exists()  # The scan completion path materializes it automatically.
    confirm(spec, "t1-2")
    capture = SynthesisCapture()
    project.findings.model = "mockllm/synthesis"
    write_project_config(project)
    model = get_model("mockllm/synthesis", custom_outputs=capture, memoize=False)
    asyncio.run(refresh_answer(spec, project, model=model, force=True))
    first = path.read_text()
    assert "[\\[E1\\]](#e1)" in first
    assert "| t2 | 0 | 0 | not scanned |" in first
    assert "#2 (index fallback)" in first
    assert "unverified numeric" not in first
    asyncio.run(refresh_answer(spec, project, model=model))
    run_scan(project, scanners, scanner_model)
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 1
    assert "Sightings: 2 in 2 distinct scans" in path.read_text()
    match = re.search(r"<!-- findings-cache (.+) -->", path.read_text())
    assert match is not None
    metadata = json.loads(match[1])
    assert set(metadata["evidence_ids"]) == {
        e.message_id for e in derive_finding(spec, project).evidence
    }
    assert metadata["references"] == ["t1-2"]
    asyncio.run(refresh_answer(spec, project, no_synthesis=True))
    assert "Synthesis disabled" in path.read_text()
    assert len(capture.requests) == 1
    invalid = get_model(
        "mockllm/invalid-citation",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/invalid-citation",
                "The agent admits falsification [E9].\nANSWER: Supported [E9].",
            )
        ],
        memoize=False,
    )
    asyncio.run(refresh_answer(spec, project, model=invalid, force=True))
    assert "unknown citation E9" in path.read_text()


@pytest.mark.parametrize(
    "text,flagged",
    [
        ("Evidence [E9].", True),
        ("evidence exhibits: 9 [E1].", True),
        ("rejected exhibits: 1 [E1].", True),
        ("evidence exhibits: 1 [E1].", False),
        ("The agent admits falsifying the result [E1].", False),
    ],
)
def test_synthesis_checks_citations_and_named_counts(text: str, flagged: bool) -> None:
    checked = check_synthesis(
        text, {"E1"}, {"evidence exhibits": 1, "rejected exhibits": 0}, []
    )
    assert ("[CHECK:" in checked.text) == flagged


def test_empty_answer_needs_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("empty", "Find absent evidence")
    capture = SynthesisCapture()
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
    path = asyncio.run(refresh_answer(spec, project, model=model))
    assert not capture.requests
    assert "None found within coverage" in path.read_text()


def test_timestamp_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [dict(quote="I forged PASS", reason="[M2] admits it", label=None)]
        ),
        memoize=False,
    )
    run_scan(
        project,
        {
            spec.scanner_name: finding_scanner(
                spec.prompt,
                None,
                spec.prompt_sha,
                rendering=ScannerRendering(nonce=project.nonce),
                model=model,
            )
        },
        model,
    )
    confirm(spec, "t1-2")
    timestamp = datetime(2026, 9, 17, 10, 37, tzinfo=timezone.utc)

    async def with_events(
        exhibit: Exhibit, project: ProjectConfig, *, events: bool = False
    ) -> Transcript:
        transcript = await exhibit_transcript(exhibit, project, events=events)
        output = ModelOutput.from_content("mockllm/model", "I forged PASS")
        output.message.id = "t1-2"
        transcript.events = [
            ModelEvent(
                model="mockllm/model",
                input=[],
                tools=[],
                tool_choice="none",
                config=GenerateConfig(),
                output=output,
                timestamp=timestamp,
            )
        ]
        return transcript

    monkeypatch.setattr(
        "inspect_scout._findings.answer.exhibit_transcript", with_events
    )
    path = asyncio.run(refresh_answer(spec, project, no_synthesis=True))
    assert f"Earliest timestamped evidence: {timestamp.isoformat()}" in path.read_text()


@scanner(messages="all")
def half_failed_finding(prompt_sha: str) -> Scanner[Transcript]:
    async def scan(transcript: Transcript) -> Result:
        if transcript.transcript_id == "t2":
            raise ValueError("Forced finding failure")
        item = Result(
            value={"quote": "I forged PASS"},
            explanation="[M2] admits it",
            references=[Reference(type="message", id="t1-2", cite="[M2]")],
        )
        return Result(value=[item.model_dump(mode="json")], type="resultset")

    return scan


def test_scan_with_errors_refreshes_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    status = scan_all(
        project, {spec.scanner_name: half_failed_finding(spec.prompt_sha)}
    )
    assert not status.complete
    path = tmp_path / "findings/gamed/answer.md"
    text = path.read_text()
    assert "- awaiting review: proposed exhibits: 1" in text
    assert "| t2 | 0 | 0 | partial |" in text
    path.unlink()
    scan_complete(status.location, display="none")
    assert path.exists()


@pytest.mark.parametrize(
    "text,flagged",
    [
        # Correct sentences of the kinds the check used to flag.
        ('agent-alpha said "all 5 workers across 3 queues" fail [E1].', False),
        ("agent-alpha ran `head -20` on the log file [E1].", False),
        ("One agent planned the probe [E1].", False),
        ("agent-alpha read a 250-byte limit as a safeguard [E1].", False),
        ("Excluded exhibits: rejected 55 [E1].", False),
        ("agent-alpha made 37 status writes [E1].", True),
        ("Evidence [E2].", True),
    ],
)
def test_synthesis_numbers_need_facts_or_cited_text(text: str, flagged: bool) -> None:
    checked = check_synthesis(
        text,
        {"E1"},
        {"evidence exhibits": 1, "excluded: rejected exhibits": 55},
        ["agent-alpha"],
        unlocated_ids={"E2"},
        exhibit_texts={"E1": "reads the 250 byte limit"},
    )
    assert ("[CHECK:" in checked.text) == flagged


def test_synthesis_check_keeps_paragraphs() -> None:
    text = (
        'agent-alpha wrote "PLEASE REVIEW... I review yours" [E1].\nANSWER: Yes [E1].'
    )
    checked = check_synthesis(text, {"E1"}, {}, ["agent-alpha"])
    assert checked.text == text


def test_synthesis_input_names_agents_and_omits_unlocated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    scanner_model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [
                dict(quote="I forged PASS", reason="[M2] admits it", label=None),
                dict(quote="Never said this", reason="[M3] claims it", label=None),
            ]
        ),
        memoize=False,
    )
    finding = finding_scanner(
        spec.prompt,
        None,
        spec.prompt_sha,
        rendering=ScannerRendering(nonce=project.nonce),
        model=scanner_model,
    )
    scan_all(project, {spec.scanner_name: finding})
    # The ledger accepts a confirmed unlocated identity (older reviews wrote them),
    # so synthesis must still leave it out.
    for exhibit in derive_finding(spec, project).exhibits:
        confirm(spec, exhibit.message_id)
    view = derive_finding(spec, project)
    assert {e.verification for e in view.evidence} == {"verified", "unlocated"}
    prompts: list[str] = []

    async def capture_prompt(prompt: str, model: object) -> str:
        prompts.append(prompt)
        # E2 is t1's unlocated exhibit (ordered after its located one).
        return (
            "t1 and t2 forged PASS [E1][E3]. t1 said more [E2].\nANSWER: Both [E1][E3]."
        )

    monkeypatch.setattr(
        "inspect_scout._findings.answer.generate_synthesis", capture_prompt
    )
    path = asyncio.run(refresh_answer(spec, project, force=True, view=view))
    exhibits = prompts[0].split("<exhibits>")[1].split("</exhibits>")[0]
    assert "agent t1 #2" in exhibits and "agent t2 #2" in exhibits
    assert "Never said this" not in exhibits
    # Agent facts follow a canonical order, whatever order coverage was built in.
    shuffled = view.model_copy(deep=True)
    shuffled.coverage[spec.prompt_sha] = dict(
        reversed(shuffled.coverage[spec.prompt_sha].items())
    )
    asyncio.run(refresh_answer(spec, project, force=True, view=shuffled))
    assert prompts[1] == prompts[0]
    assert "unlocated exhibit E2" in path.read_text()


def test_verdict_refresh_keeps_synthesis_until_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    scanner_model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [
                dict(quote="I forged PASS", reason="[M2] admits it", label=None),
                dict(quote="I rejected cheating", reason="[M3] declines", label=None),
            ]
        ),
        memoize=False,
    )
    run_scan(
        project,
        {
            spec.scanner_name: finding_scanner(
                spec.prompt,
                None,
                spec.prompt_sha,
                rendering=ScannerRendering(nonce=project.nonce),
                model=scanner_model,
            )
        },
        scanner_model,
    )
    confirm(spec, "t1-2")
    confirm(spec, "t1-3")
    capture = SynthesisCapture()
    model = get_model("mockllm/synthesis", custom_outputs=capture, memoize=False)
    capture_text = "t1 declined cheating [E2].\nANSWER: Declined [E2]."
    capture.text = capture_text
    path = asyncio.run(refresh_answer(spec, project, model=model, force=True))
    assert len(capture.requests) == 1
    append_verdict(
        spec,
        VerdictRow(
            message_id="t1-2",
            verdict="rejected",
            decided_by="human:reviewer",
            reason="Not evidence",
            source="manual",
        ),
    )
    asyncio.run(refresh_answer(spec, project, model=model, synthesize=False))
    assert len(capture.requests) == 1
    before, synthesis = path.read_text().split("## Synthesis")
    # The notice is outside the synthesis section, so it cannot read as model text.
    assert "> **Out of date:**" in before and "Out of date" not in synthesis
    # The kept prose now cites t1-3 as E1, its number in the current evidence.
    assert "[\\[E1\\]](#e1)" in synthesis and "#e2" not in synthesis
    asyncio.run(refresh_answer(spec, project, model=model))
    assert len(capture.requests) == 2
    assert "Out of date" not in path.read_text()
