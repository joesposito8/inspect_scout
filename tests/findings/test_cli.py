from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from inspect_scout import Result, Scanner, Transcript, scan, scanner, transcripts_from
from inspect_scout._cli.main import scout
from inspect_scout._findings.spec import FindingSelection, register_finding

SCANNER = str(Path(__file__).parents[1] / "cli" / "broken_scanner.py")
LOGS = str(Path(__file__).parents[2] / "examples" / "scanner" / "logs")


@pytest.fixture(autouse=True)
def isolated_cli_options() -> Iterator[None]:
    with patch("inspect_scout._cli.scan.process_common_options"):
        yield


@scanner(messages="all")
def noop() -> Scanner[Transcript]:
    async def scan(transcript: Transcript) -> Result:
        return Result(value=None)

    return scan


def two_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two registered findings; a scan runs only the ones it names."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scout.yaml").write_text(
        f"transcripts: {LOGS}\nfindings:\n  model: mockllm/model\n"
    )
    register_finding("alpha", "Find A")
    register_finding("beta", "Find B")


@pytest.mark.parametrize(
    "findings,scanners,bundle,names",
    [
        (None, {"noop": noop()}, False, ["noop"]),
        ("beta", {}, False, ["finding_beta"]),
        (["alpha", "beta"], {}, False, ["finding_alpha", "finding_beta"]),
        ("all", {}, False, ["finding_alpha", "finding_beta"]),
        ("none", {"noop": noop()}, False, ["noop"]),
        ("beta", {"noop": noop()}, True, ["noop", "finding_bundle"]),
        # A finding scanner the job already defines is not added again.
        ("alpha", {"finding_alpha": noop()}, False, ["finding_alpha"]),
    ],
)
def test_scan_selects_findings(
    findings: FindingSelection,
    scanners: dict[str, Scanner[Transcript]],
    bundle: bool,
    names: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    two_findings(tmp_path, monkeypatch)
    status = scan(
        scanners,
        transcripts=transcripts_from(LOGS).where("task_id = 'nonexistent'"),
        findings=findings,
        findings_bundle=bundle,
        display="none",
        dry_run=True,
    )
    assert list(status.spec.scanners) == names
    if "finding_alpha" in names and not scanners:
        params = status.spec.scanners["finding_alpha"].params
        assert params["prompt_sha"]
        assert params["rendering"]["nonce"]
    if "finding_alpha" in scanners:
        assert status.spec.scanners["finding_alpha"].name.endswith("noop")


@pytest.mark.parametrize(
    "args,findings,bundle",
    [
        ([SCANNER], [], False),
        (["--findings", "beta"], ["beta"], False),
        ([SCANNER, "--findings", "none"], [], False),
        (["--findings-bundle", "--findings", "all", SCANNER], ["alpha", "beta"], True),
    ],
)
def test_cli_passes_selection_and_project_filter(
    args: list[str],
    findings: list[str],
    bundle: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    two_findings(tmp_path, monkeypatch)
    with patch("inspect_scout._cli.scan.scan") as run:
        result = CliRunner().invoke(
            scout,
            [
                "scan",
                *args,
                "-F",
                "task_id = 'nonexistent'",
                "--display",
                "none",
                "--dry-run",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    kwargs: Mapping[str, Any] = run.call_args.kwargs
    assert kwargs["findings"] == findings
    assert kwargs["findings_bundle"] is bundle
    assert len(kwargs["transcripts"]._query.where) == 1


def test_registration_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert (
        runner.invoke(
            scout, ["findings", "register", "alpha", "--prompt", "Find A"]
        ).exit_code
        == 0
    )
    listed = runner.invoke(scout, ["findings", "list"]).output
    assert listed.startswith("alpha\tAlpha\t") and listed.count("\t") == 2
    assert "Find A" in runner.invoke(scout, ["findings", "show", "alpha"]).output
    for removed in (
        ["findings", "ride-along", "alpha"],
        ["findings", "register", "beta", "--prompt", "Find B", "--ride-along"],
    ):
        assert runner.invoke(scout, removed).exit_code != 0
    unknown = runner.invoke(scout, ["scan", "--findings", "unknown"])
    assert unknown.exit_code != 0 and "Unknown finding" in unknown.output
    # Without FILE, a scan needs named findings: none run by default.
    for args in (["scan"], ["scan", "--findings", "none"]):
        missing = runner.invoke(scout, args)
        assert missing.exit_code != 0 and "Missing argument 'FILE'" in missing.output
