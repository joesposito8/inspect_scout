from pathlib import Path
from typing import Literal

import pytest
from click.testing import CliRunner
from inspect_ai.model import Model, get_model
from inspect_scout import Scanner, ScannerRendering, Transcript
from inspect_scout._cli.main import scout
from inspect_scout._findings import judge as judge_module
from inspect_scout._findings.derive import derive_finding
from inspect_scout._findings.scanner import finding_scanner
from inspect_scout._findings.spec import read_finding, register_finding
from inspect_scout._findings.verdicts import VerdictRow, append_verdict, read_verdicts

from .test_derive import run_finding, run_scan, setup_project
from .test_judge import DECLINED, FORGED, JudgeCapture
from .test_scanner import Capture


@pytest.mark.parametrize("verdict", ["confirmed", "rejected"])
def test_unlocated_identity_and_legacy_verdict_survive_rescan(
    verdict: Literal["confirmed", "rejected"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [dict(quote="Absent quotation", reason="[M2] claims this", label=None)]
        ),
        memoize=False,
    )
    scanners: dict[str, Scanner[Transcript]] = {
        spec.scanner_name: finding_scanner(
            spec.prompt,
            None,
            spec.prompt_sha,
            rendering=ScannerRendering(nonce=project.nonce),
            model=model,
        )
    }
    run_scan(project, scanners, model)
    first = derive_finding(spec, project).exhibits[0]
    append_verdict(
        spec,
        VerdictRow(
            message_id=f"unlocated:{first.sightings[0].result_uuid}",
            verdict=verdict,
            decided_by="human:reviewer",
            reason="Reviewed legacy result",
            source="manual",
        ),
    )
    assert derive_finding(spec, project).exhibits[0].state == verdict
    before = (tmp_path / "findings/gamed/verdicts.jsonl").read_bytes()
    run_scan(project, scanners, model)
    view = derive_finding(spec, project)
    assert len(view.exhibits) == 1
    assert view.exhibits[0].message_id == first.message_id
    assert view.exhibits[0].state == verdict
    assert view.exhibits[0].corroboration == 2
    assert len({s.result_uuid for s in view.exhibits[0].sightings}) == 2
    assert view.agreements[0].line == view.agreements[1].line
    assert (tmp_path / "findings/gamed/verdicts.jsonl").read_bytes() == before
    changed: Literal["confirmed", "rejected"] = (
        "rejected" if verdict == "confirmed" else "confirmed"
    )
    append_verdict(
        spec,
        VerdictRow(
            message_id=first.message_id,
            verdict=changed,
            decided_by="human:reviewer",
            reason="New decision",
            source="manual",
        ),
    )
    assert derive_finding(spec, project).exhibits[0].state == changed


def test_judge_cli_limits_records_and_refreshes_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    run_finding(
        project, register_finding("gamed", "Find misconduct"), [FORGED, DECLINED]
    )
    answer = tmp_path / "findings/gamed/answer.md"
    answer.unlink()
    confirming = get_model(
        "mockllm/judge", custom_outputs=JudgeCapture("confirmed"), memoize=False
    )

    def judge_model(model: str | Model) -> Model:
        return confirming

    monkeypatch.setattr(judge_module, "get_model", judge_model)
    result = CliRunner().invoke(
        scout, ["findings", "judge", "gamed", "--limit", "1"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "gamed: judging 1 exhibits with mockllm/model (0 skipped: unlocated)" in (
        result.output
    )
    assert "1 confirmed, 0 rejected, 0 refused by judge, 0 failed" in result.output
    assert [
        (r.message_id, r.verdict) for r in read_verdicts(read_finding("gamed"))
    ] == [("t1-2", "confirmed")]
    # Judge-confirmed exhibits are evidence, like human-confirmed ones.
    report = answer.read_text()
    assert "- evidence exhibits: 1" in report
    assert "- awaiting review: proposed exhibits: 1" in report
    assert CliRunner().invoke(scout, ["findings", "recheck", "gamed"]).exit_code != 0
