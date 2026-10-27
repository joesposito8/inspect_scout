import asyncio
from pathlib import Path

import pytest
from click.testing import CliRunner
from inspect_ai.model import get_model
from inspect_scout import ScannerRendering
from inspect_scout._cli.main import scout
from inspect_scout._findings.agreement import judge_agreement
from inspect_scout._findings.answer import refresh_answer
from inspect_scout._findings.derive import derive_finding
from inspect_scout._findings.scanner import finding_scanner
from inspect_scout._findings.spec import register_finding
from inspect_scout._findings.verdicts import VerdictRow, append_verdict

from .test_derive import run_scan, setup_project
from .test_scanner import Capture


def test_agreement_counts_messages_within_scan_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    for message_id, verdict in [
        ("t1-1", "confirmed"),
        ("t1-2", "confirmed"),
        ("t1-3", "rejected"),
        ("t2-2", "confirmed"),
    ]:
        append_verdict(
            spec,
            VerdictRow.model_validate(
                dict(
                    message_id=message_id,
                    verdict=verdict,
                    decided_by="human:reviewer",
                    reason="Reviewed",
                    source="manual",
                )
            ),
        )
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [
                dict(quote="I forged PASS", reason="[M2] admits it", label=None),
                dict(quote="I forged PASS", reason="[M2] duplicate", label=None),
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
                model=model,
            )
        },
        model,
    )
    agreement = derive_finding(spec, project).agreements[-1]
    assert agreement.line == "found 1 of 2 human-confirmed; cited 1 of 1 human-rejected"


def row(message_id: str, actor: str, verdict: str) -> VerdictRow:
    return VerdictRow.model_validate(
        dict(
            message_id=message_id,
            verdict=verdict,
            label=None,
            decided_by=actor,
            reason="Decided",
            source="manual" if actor.startswith("human:") else "judge",
        )
    )


@pytest.mark.parametrize(
    "rows,kept,reviewed",
    [
        ([], 0, 0),
        # Unreviewed judge decisions do not count either way.
        ([("m", "judge:j", "confirmed")], 0, 0),
        ([("m", "judge:j", "confirmed"), ("m", "human:h", "confirmed")], 1, 1),
        ([("m", "judge:j", "rejected"), ("m", "human:h", "confirmed")], 0, 1),
        # A human decision before the judge's is not a review of it.
        ([("m", "human:h", "rejected"), ("m", "judge:j", "rejected")], 0, 0),
        # The human's latest decision after the judge's is the one compared.
        (
            [
                ("m", "judge:j", "rejected"),
                ("m", "human:h", "confirmed"),
                ("m", "human:h", "rejected"),
            ],
            1,
            1,
        ),
        # A re-judged message is measured on its latest judge decision.
        (
            [
                ("m", "judge:j", "rejected"),
                ("m", "judge:j", "confirmed"),
                ("m", "human:h", "confirmed"),
            ],
            1,
            1,
        ),
        # Judge rows from older versions proposed; that is not a decision.
        ([("m", "judge:j", "proposed"), ("m", "human:h", "rejected")], 0, 0),
        (
            [
                ("a", "judge:j", "confirmed"),
                ("b", "judge:j", "rejected"),
                ("a", "human:h", "confirmed"),
                ("b", "human:h", "confirmed"),
            ],
            1,
            2,
        ),
    ],
)
def test_judge_agreement_measures_judge_decisions_humans_reviewed(
    rows: list[tuple[str, str, str]], kept: int, reviewed: int
) -> None:
    agreement = judge_agreement([row(*r) for r in rows])
    assert (agreement.kept, agreement.reviewed) == (kept, reviewed)
    assert agreement.overturned == reviewed - kept


def test_judge_agreement_is_reported_in_status_and_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    for message_id, actor, verdict in [
        ("t1-2", "judge:j", "confirmed"),
        ("t1-3", "judge:j", "confirmed"),
        ("t1-2", "human:h", "confirmed"),
        ("t1-3", "human:h", "rejected"),
    ]:
        append_verdict(spec, row(message_id, actor, verdict))
    line = "judge agreement: kept 1 of 2 human-reviewed judge decisions"
    assert derive_finding(spec, project).judge_agreement.line == line
    status = CliRunner().invoke(scout, ["findings", "status"], catch_exceptions=False)
    assert line in status.output
    asyncio.run(refresh_answer(spec, project, no_synthesis=True))
    answer = (tmp_path / "findings/gamed/answer.md").read_text()
    assert "Judge agreement: kept 1 of 2 human-reviewed judge decisions." in answer
