import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from inspect_ai.model import get_model
from inspect_scout import ScannerRendering
from inspect_scout._cli.main import scout
from inspect_scout._findings.derive import derive_finding
from inspect_scout._findings.scanner import finding_scanner
from inspect_scout._findings.spec import register_finding
from inspect_scout._findings.verdicts import (
    RETIRED_VERDICT,
    VerdictRow,
    append_verdict,
    effective_verdicts,
    read_verdicts,
)
from inspect_scout._project._project import write_project_config

from .test_derive import run_scan, setup_project
from .test_scanner import Capture


def test_decisions_survive_rescans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct", labels=["gamed", "declined"])
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [{"quote": "I forged PASS", "reason": "[M2] admits it", "label": "gamed"}]
        ),
        memoize=False,
    )
    scanner = finding_scanner(
        spec.prompt,
        spec.labels,
        spec.prompt_sha,
        rendering=ScannerRendering(nonce=project.nonce),
        model=model,
    )
    run_scan(project, {spec.scanner_name: scanner}, model)
    assert derive_finding(spec, project).exhibits[0].state == "proposed"
    cases = [
        ("judge:a", "rejected", None, "rejected"),
        ("judge:a", "confirmed", "gamed", "confirmed"),
        ("judge:b", "proposed", "gamed", "proposed"),
        ("human:reviewer", "rejected", None, "rejected"),
        ("judge:c", "proposed", "gamed", "rejected"),
        ("judge:d", "confirmed", "gamed", "rejected"),
        ("human:reviewer", "confirmed", "declined", "confirmed"),
        # A human decision outranks every judge row, earlier or later.
        ("judge:e", "rejected", None, "confirmed"),
    ]
    for actor, state, label, expected in cases:
        row = VerdictRow.model_validate(
            dict(
                message_id="t1-2",
                verdict=state,
                label=label,
                decided_by=actor,
                reason="Reviewed",
                source="manual" if actor.startswith("human:") else "recheck",
            )
        )
        append_verdict(spec, row)
        run_scan(project, {spec.scanner_name: scanner}, model)
        view = derive_finding(spec, project)
        assert view.exhibits[0].state == expected
        assert bool(view.evidence) == (expected == "confirmed")
    assert view.exhibits[0].label == "declined"
    assert len(read_verdicts(spec)) == len(cases)


def test_manual_message_without_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    append_verdict(
        spec,
        VerdictRow(
            message_id="t2-3",
            verdict="confirmed",
            decided_by="human:reviewer",
            reason="Manual evidence",
            source="manual",
        ),
    )
    view = derive_finding(spec, project)
    assert [(e.message_id, e.state, e.sightings) for e in view.evidence] == [
        ("t2-3", "confirmed", [])
    ]
    assert view.evidence[0].message_label == "M3"


def test_judge_rows_from_older_versions_keep_their_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")

    # Rows exactly as the older recheck command serialized them.
    def recheck_row(message_id: str, verdict: str) -> dict[str, object]:
        return dict(
            message_id=message_id,
            verdict=verdict,
            label=None,
            decided_by="judge:anthropic/claude-haiku-5-5",
            reason="[M2] checked",
            source="recheck",
            added="2026-10-20T12:00:00Z",
            quote="I forged PASS",
            refused=False,
        )

    rows = [
        recheck_row("t1-2", "proposed"),
        recheck_row("t1-2", "rejected"),
        dict(
            message_id="t1-3",
            verdict="confirmed",
            label=None,
            decided_by="human:reviewer",
            reason="Confirmed during review",
            source="manual",
            added="2026-10-20T11:00:00Z",
        ),
        recheck_row("t1-3", "rejected"),
    ]
    ledger = tmp_path / "findings/gamed/verdicts.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    effective = effective_verdicts(read_verdicts(spec))
    assert {key: (row.verdict, row.decided_by) for key, row in effective.items()} == {
        "t1-2": ("rejected", "judge:anthropic/claude-haiku-5-5"),
        "t1-3": ("confirmed", "human:reviewer"),
    }


def test_retired_verdict_rows_are_ignored_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    write_project_config(project)
    spec = register_finding("gamed", "Find misconduct")
    ledger = tmp_path / "findings/gamed/verdicts.jsonl"
    rows = [
        dict(
            message_id="t2-3",
            verdict="confirmed",
            decided_by="human:reviewer",
            reason="Manual evidence",
            source="manual",
        ),
        # Older versions let humans and refusing judges write this verdict.
        dict(
            message_id="t2-3",
            verdict=RETIRED_VERDICT,
            decided_by="human:reviewer",
            reason="Not sure",
            source="manual",
        ),
        dict(
            message_id="t2-3",
            verdict=RETIRED_VERDICT,
            refused=True,
            quote="",
            decided_by="judge:old",
            reason="The judge refused",
            source="recheck",
        ),
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert [row.verdict for row in read_verdicts(spec)] == ["confirmed"]
    assert derive_finding(spec, project).exhibits[0].state == "confirmed"
    status = CliRunner().invoke(scout, ["findings", "status"], catch_exceptions=False)
    assert status.exit_code == 0
    assert (
        f"ignoring 2 '{RETIRED_VERDICT}' rows (a retired verdict) in {ledger}"
        in status.output
    )
