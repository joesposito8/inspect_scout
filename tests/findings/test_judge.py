import asyncio
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_scout import Scanner, Transcript, scan, transcripts_from
from inspect_scout._cli.main import scout
from inspect_scout._findings import derive
from inspect_scout._findings.answer import refresh_answer
from inspect_scout._findings.derive import Exhibit, derive_finding
from inspect_scout._findings.judge import MAX_EXAMPLES, judge_finding, nearest_examples
from inspect_scout._findings.spec import register_finding
from inspect_scout._findings.verdicts import (
    VerdictRow,
    append_verdict,
    read_verdicts,
)
from inspect_scout._findings.verify import QuoteVerifier
from inspect_scout._project._project import write_project_config
from inspect_scout._project.types import ProjectConfig
from inspect_scout._recorder.recorder import Status

from .test_derive import purpose_cites, run_finding, setup_project

FORGED = {"quote": "I forged PASS", "reason": "[M2] admits it"}
DECLINED = {"quote": "I rejected cheating", "reason": "[M3] declines"}


class JudgeCapture:
    def __init__(self, verdict: str = "rejected", label: str | None = None) -> None:
        self.requests: list[list[ChatMessage]] = []
        self.configs: list[GenerateConfig] = []
        self.verdict = verdict
        self.label = label

    def __call__(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.requests.append(messages)
        self.configs.append(config)
        args: dict[str, Any] = dict(
            reason="[M2] decides it", verdict=self.verdict, label=self.label
        )
        return ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[ToolCall(id="verdict", function="answer", arguments=args)],
            )
        )


def human(
    message_id: str, verdict: str, reason: str, label: str | None = None
) -> VerdictRow:
    return VerdictRow.model_validate(
        dict(
            message_id=message_id,
            verdict=verdict,
            label=label,
            decided_by="human:reviewer",
            reason=reason,
            source="manual",
        )
    )


def test_judge_learns_from_human_decisions_and_never_rejudges_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct", labels=["gamed", "declined"])
    run_finding(
        project, spec, [dict(FORGED, label="gamed"), dict(DECLINED, label="declined")]
    )
    append_verdict(spec, human("t1-3", "rejected", "Declining is not gaming"))
    # A message no scanner cited, added by hand in another agent's transcript.
    append_verdict(spec, human("t2-2", "confirmed", "Forgery counts", "gamed"))
    capture = JudgeCapture("confirmed", "gamed")
    judge = get_model("mockllm/judge", custom_outputs=capture, memoize=False)
    run = asyncio.run(judge_finding(spec, project, model=judge))
    assert (run.candidates, run.confirmed, run.rejected, run.only) == (
        1,
        1,
        0,
        "proposed",
    )
    [request] = capture.requests
    assert capture.configs[0].cache_prompt is False
    text = request[-1].text
    examples = text.split("<reviewed_examples>\n")[1].split("\n</reviewed_examples>")[0]
    shown = [json.loads(line) for line in examples.splitlines()]
    # Confirmed examples come first; the candidate itself is never an example.
    assert [(e["decision"], e["reviewer_reason"]) for e in shown] == [
        ("confirmed", "Forgery counts"),
        ("rejected", "Declining is not gaming"),
    ]
    assert shown[0]["proposed_because"] is None
    assert shown[1]["proposed_because"] == "[M3] declines"
    assert json.loads(text.split("<exhibit>\n")[1].split("\n</exhibit>")[0]) == {
        "agent": "t1",
        "message": "M2",
        "quote": "I forged PASS",
        "proposed_label": "gamed",
        "proposed_because": "[M2] admits it",
    }
    assert "TestNonce123" in text and "marked message [M2]" in text
    saved = read_verdicts(spec)[-1]
    assert (saved.message_id, saved.verdict, saved.label, saved.source) == (
        "t1-2",
        "confirmed",
        "gamed",
        "judge",
    )
    assert saved.decided_by == "judge:mockllm/judge"
    view = derive_finding(spec, project)
    assert {e.message_id for e in view.evidence} == {"t1-2", "t2-2"}
    # A judge decision is re-decided only on request; human decisions never are.
    assert asyncio.run(judge_finding(spec, project, model=judge)).candidates == 0
    again = asyncio.run(judge_finding(spec, project, only="all", model=judge))
    # The identical prompt is answered from the local model cache.
    assert (again.candidates, again.confirmed, len(capture.requests)) == (1, 1, 1)
    append_verdict(spec, human("t1-2", "rejected", "Not this one"))
    states = {e.message_id: e.state for e in derive_finding(spec, project).exhibits}
    assert states["t1-2"] == "rejected"
    assert (
        asyncio.run(judge_finding(spec, project, only="all", model=judge)).candidates
        == 0
    )
    assert len(capture.requests) == 1


@pytest.mark.parametrize(
    "verdict,label,saved",
    [
        ("confirmed", "gamed", ("confirmed", "gamed")),
        ("rejected", "gamed", ("rejected", None)),
        ("confirmed", "bogus", None),
        ("confirmed", None, None),
    ],
)
def test_judge_labels_follow_the_finding(
    verdict: str,
    label: str | None,
    saved: tuple[str, str | None] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct", labels=["gamed", "declined"])
    run_finding(project, spec, [dict(FORGED, label="gamed")])
    judge = get_model(
        "mockllm/judge", custom_outputs=JudgeCapture(verdict, label), memoize=False
    )
    run = asyncio.run(judge_finding(spec, project, model=judge))
    assert [(r.verdict, r.label) for r in read_verdicts(spec)] == (
        [saved] if saved else []
    )
    assert run.failed == (saved is None)


def exhibit(transcript_id: str, index: int, state: str) -> Exhibit:
    return Exhibit.model_validate(
        dict(
            message_id=f"{transcript_id}-{index}",
            transcript_id=transcript_id,
            model_ref=None,
            verified_ref=f"{transcript_id}-{index}",
            verification="verified",
            message_index=index,
            message_label=f"M{index}",
            quote="q",
            reason="r",
            label=None,
            state=state,
            decided_by="human:reviewer",
        )
    )


def test_nearest_examples_prefer_the_same_agent_then_nearby_messages() -> None:
    candidate = exhibit("a", 50, "proposed")
    pool = [
        candidate,
        *(exhibit("a", i, "confirmed") for i in (10, 40, 45, 60, 90)),
        *(exhibit("b", i, "confirmed") for i in range(48, 56)),
        exhibit("a", 51, "rejected"),
        exhibit("b", 50, "rejected"),
    ]
    chosen = [
        (e.transcript_id, e.message_index, e.state)
        for e in nearest_examples(candidate, pool)
    ]
    # Equal distances fall back to message id order.
    assert chosen == [
        ("a", 45, "confirmed"),
        ("a", 40, "confirmed"),
        ("a", 60, "confirmed"),
        ("a", 10, "confirmed"),
        ("a", 90, "confirmed"),
        ("b", 50, "confirmed"),
        ("b", 49, "confirmed"),
        ("b", 51, "confirmed"),
        ("a", 51, "rejected"),
        ("b", 50, "rejected"),
    ]
    assert len([e for e in chosen if e[2] == "confirmed"]) == MAX_EXAMPLES


def test_completed_scans_are_derived_once_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    run_finding(project, spec, [FORGED], others={"purpose": purpose_cites()})
    view = derive_finding(spec, project)
    assert {e.message_id: e.state for e in view.exhibits} == {"t1-2": "proposed"}

    def new_process() -> None:
        monkeypatch.setattr(derive, "_scan_cache", {})
        monkeypatch.setattr(derive, "_verifiers", OrderedDict())

    async def no_transcript_reads(*args: Any) -> QuoteVerifier:
        raise AssertionError("a completed scan was derived again")

    new_process()
    with monkeypatch.context() as patched:
        patched.setattr(derive, "_verifier", no_transcript_reads)
        assert derive_finding(spec, project) == view
    # Results stored by older derivation code are recomputed, not reused.
    new_process()
    monkeypatch.setattr(derive, "DERIVATION_VERSION", derive.DERIVATION_VERSION + 1)
    assert derive_finding(spec, project) == view
    stored = list((tmp_path / "findings" / ".derived").glob("*.json"))
    assert len(stored) == 1
    assert all(
        json.loads(path.read_text())["version"] == derive.DERIVATION_VERSION
        for path in stored
    )


def scan_all(
    project: ProjectConfig, scanners: dict[str, Scanner[Transcript]]
) -> Status:
    assert project.transcripts is not None
    return scan(
        scanners=scanners,
        transcripts=transcripts_from(project.transcripts),
        scans=project.scans,
        model=get_model("mockllm/model"),
        max_processes=1,
        display="none",
    )


class Refusing:
    def __init__(self, fail_on: str | None = None) -> None:
        self.requests = 0
        self.fail_on = fail_on

    def __call__(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.requests += 1
        if self.fail_on is not None and self.fail_on in messages[-1].text:
            raise RuntimeError("provider unavailable")
        return ModelOutput.from_content(
            "mockllm/judge", "I can't help with that.", stop_reason="content_filter"
        )


def test_refusal_is_counted_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    write_project_config(project)
    spec = register_finding("gamed", "Find misconduct")
    run_finding(project, spec, [FORGED])
    refusing = Refusing()
    judge = get_model("mockllm/judge", custom_outputs=refusing, memoize=False)
    run = asyncio.run(judge_finding(spec, project, model=judge))
    assert (run.candidates, run.confirmed, run.rejected, run.refused, run.failed) == (
        1,
        0,
        0,
        1,
        0,
    )
    assert refusing.requests == 1
    assert read_verdicts(spec) == []
    view = derive_finding(spec, project)
    assert {e.state for e in view.exhibits} == {"proposed"}
    assert view.last_judge_run == run
    asyncio.run(refresh_answer(spec, project, no_synthesis=True))
    answer = (tmp_path / "findings/gamed/answer.md").read_text()
    assert "Review: 0 confirmed; 1 proposed; 0 rejected; 1 refused by judge." in answer
    status = CliRunner().invoke(scout, ["findings", "status"], catch_exceptions=False)
    assert "1 refused by judge" in status.output


def test_failed_request_leaves_its_exhibit_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    run_finding(project, spec, [FORGED, DECLINED])
    judge = get_model(
        "mockllm/judge", custom_outputs=Refusing(fail_on="[M3]"), memoize=False
    )
    run = asyncio.run(judge_finding(spec, project, model=judge))
    assert (run.refused, run.failed) == (1, 1)
    assert {e.message_id: e.state for e in derive_finding(spec, project).exhibits} == {
        "t1-2": "proposed",
        "t1-3": "proposed",
    }
