from pathlib import Path

import pytest
from click.testing import CliRunner
from inspect_ai.model import get_model
from inspect_scout import ScannerRendering
from inspect_scout._cli.main import scout
from inspect_scout._findings.derive import derive_finding
from inspect_scout._findings.scanner import finding_scanner
from inspect_scout._findings.spec import register_finding
from inspect_scout._findings.verdicts import read_verdicts
from inspect_scout._project._project import write_project_config

from .test_derive import run_scan, setup_project
from .test_scanner import Capture


def test_review_cli_appends_and_relabels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    write_project_config(project)
    spec = register_finding("gamed", "Find misconduct", labels=["gamed", "declined"])
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [
                dict(quote="I forged PASS", reason="[M2] admits it", label="gamed"),
                dict(
                    quote="I rejected cheating",
                    reason="[M3] declines",
                    label="declined",
                ),
            ]
        ),
        memoize=False,
    )
    run_scan(
        project,
        {
            spec.scanner_name: finding_scanner(
                spec.prompt,
                spec.labels,
                spec.prompt_sha,
                rendering=ScannerRendering(nonce=project.nonce),
                model=model,
            )
        },
        model,
    )
    runner = CliRunner()
    reviewed = runner.invoke(
        scout,
        ["findings", "review", "gamed"],
        input="r\nl declined\n",
        catch_exceptions=False,
    )
    assert reviewed.exit_code == 0
    assert "[M2] | #2" in reviewed.output
    assert "?tab=transcript-messages&message=t1-2" in reviewed.output
    assert [r.verdict for r in read_verdicts(spec)] == ["rejected", "confirmed"]
    shown = runner.invoke(scout, ["findings", "show", "gamed"], catch_exceptions=False)
    assert (
        "rejected | unlabeled" in shown.output
        and "confirmed | declined" in shown.output
    )
    assert read_verdicts(spec)[0].label is None
    assert not runner.invoke(scout, ["findings", "review", "gamed"], input="q\n").output
    reviewed = runner.invoke(
        scout,
        ["findings", "review", "gamed", "--all"],
        input="r\nr\n",
        catch_exceptions=False,
    )
    assert reviewed.exit_code == 0
    assert {e.state for e in derive_finding(spec, project).exhibits} == {"rejected"}
    assert read_verdicts(spec)[-1].label is None
    status = runner.invoke(scout, ["findings", "status"], catch_exceptions=False)
    assert len(status.output.splitlines()) == 1
    assert "2 rejected" in status.output and "0 confirmed" in status.output
    assert "human-confirmed" in status.output and "coverage" in status.output


def test_review_offers_no_confirmation_for_unlocated_exhibits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [dict(quote="Not in this transcript", reason="[M2] claims it", label=None)]
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
    reviewed = CliRunner().invoke(
        scout, ["findings", "review", "gamed"], input="c\nr\n", catch_exceptions=False
    )
    assert reviewed.exit_code == 0
    assert "Use r, s, q." in reviewed.output
    assert [r.verdict for r in read_verdicts(spec)] == ["rejected"]
