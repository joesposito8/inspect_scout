import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_scout import (
    Reference,
    Result,
    ScannerRendering,
    Transcript,
    scan,
    scan_results_df,
    scanner,
    transcripts_db,
    transcripts_from,
)
from inspect_scout._findings import derive
from inspect_scout._findings.derive import _unlocated_id, derive_finding
from inspect_scout._findings.scanner import finding_bundle_scanner, finding_scanner
from inspect_scout._findings.spec import FindingSpec, register_finding
from inspect_scout._findings.verdicts import VerdictRow, append_verdict
from inspect_scout._project._project import write_project_config
from inspect_scout._project.types import FindingsConfig, ProjectConfig
from inspect_scout._scanner.scanner import Scanner, scanner_create

from .test_scanner import Capture


def setup_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectConfig:
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "transcripts")

    async def insert() -> None:
        async with transcripts_db(db_path) as db:
            await db.insert(
                [
                    Transcript(
                        transcript_id=tid,
                        source_type="test",
                        source_id=tid,
                        source_uri=f"test://{tid}",
                        task_id=tid,
                        messages=[
                            ChatMessageUser(id=f"{tid}-1", content="Task instructions"),
                            ChatMessageAssistant(
                                id=f"{tid}-2", content="I forged PASS"
                            ),
                            ChatMessageAssistant(
                                id=f"{tid}-3", content="I rejected cheating"
                            ),
                        ],
                    )
                    for tid in ("t1", "t2")
                ]
            )

    asyncio.run(insert())
    project = ProjectConfig(
        transcripts=db_path,
        scans=str(tmp_path / "scans"),
        nonce="TestNonce123",
        findings=FindingsConfig(model="mockllm/model"),
    )
    write_project_config(project)
    return project


def run_scan(
    project: ProjectConfig, scanners: dict[str, Scanner[Transcript]], model: Model
) -> str:
    assert project.transcripts is not None
    status = scan(
        scanners=scanners,
        transcripts=transcripts_from(project.transcripts).where("transcript_id = 't1'"),
        scans=project.scans,
        model=model,
        model_config=GenerateConfig(cache=True),
        max_processes=1,
        display="none",
    )
    return status.location


def test_rerun_and_scan_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct", labels=["gamed"])
    capture = Capture(
        [
            {"quote": "I forged PASS", "reason": "[M2] admits it", "label": "gamed"},
            {"quote": "I forged PASS", "reason": "[M2] repeats it", "label": "gamed"},
            {
                "quote": "I rejected cheating",
                "reason": "[M1] rejects it",
                "label": "gamed",
            },
            {
                "quote": "Not in this transcript",
                "reason": "[M2] claims it",
                "label": "gamed",
            },
        ]
    )
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
    factory = finding_scanner(
        spec.prompt,
        spec.labels,
        spec.prompt_sha,
        rendering=ScannerRendering(nonce=project.nonce),
        model=model,
    )
    first_location = run_scan(project, {spec.scanner_name: factory}, model)
    first = derive_finding(spec, project)
    assert len(first.exhibits) == 3
    assert [e.verification for e in first.exhibits] == [
        "verified",
        "relocated",
        "unlocated",
    ]
    assert first.exhibits[1].model_ref == "t1-1"
    assert first.exhibits[1].verified_ref == "t1-3"
    assert first.exhibits[1].message_index == 3
    assert first.exhibits[1].message_label == "M3"
    assert first.coverage[spec.prompt_sha] == {"t1": "judged", "t2": "missing"}
    assert [len(e.sightings) for e in first.exhibits] == [2, 1, 1]
    second_location = run_scan(project, {spec.scanner_name: factory}, model)
    second = derive_finding(spec, project)
    assert len(capture.requests) == 1
    assert len(second.exhibits) == 3
    located = [e for e in second.exhibits if e.verified_ref is not None]
    assert {e.message_id for e in located} == {"t1-2", "t1-3"}
    assert all(e.corroboration == 2 for e in located)
    assert [len(e.sightings) for e in located] == [4, 2]
    assert (
        len([e for e in second.exhibits if e.message_id.startswith("unlocated:")]) == 1
    )
    # Deriving again must not mutate the cached contribution's sighting lists.
    assert derive_finding(spec, project) == second
    results = scan_results_df(second_location, rows="transcripts")
    assert all(
        not json.loads(usage)
        for usage in results.scanners[spec.scanner_name].scan_model_usage
    )
    recorded = results.spec.scanners[spec.scanner_name]
    assert recorded.params["prompt_sha"] == spec.prompt_sha
    assert recorded.params["rendering"]["nonce"] == project.nonce
    scanner_create(recorded.name, recorded.params)
    shutil.rmtree(second_location)
    assert derive_finding(spec, project) == first
    assert Path(first_location).exists()
    changed = FindingSpec(slug=spec.slug, title=spec.title, prompt="Changed question")
    view = derive_finding(changed, project)
    assert view.coverage[spec.prompt_sha]["t1"] == "judged"
    assert view.coverage[changed.prompt_sha]["t1"] == "missing"


@pytest.mark.parametrize(
    "invalid_label,expected", [(False, "judged"), (True, "partial")]
)
def test_empty_and_partial_coverage(
    invalid_label: bool, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("question", "Find evidence", labels=["allowed"])
    capture = Capture(
        [{"quote": "I forged PASS", "reason": "[M2] evidence", "label": "invalid"}]
        if invalid_label
        else []
    )
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
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
    view = derive_finding(spec, project)
    assert not view.exhibits
    assert view.coverage[spec.prompt_sha] == {"t1": expected, "t2": "missing"}


# Older versions recorded finding fields that later ones dropped.
@pytest.mark.parametrize(
    "recorded_extra", [{}, {"ride_along": True, "sources": ["EarlierScan"]}]
)
def test_bundle_attribution(
    recorded_extra: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    specs = [register_finding(slug, f"Find {slug}") for slug in ("gamed", "declined")]
    capture = Capture(
        [
            {
                "finding": "gamed",
                "quote": "I forged PASS",
                "reason": "[M2] evidence",
                "label": None,
            }
        ]
    )
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
    location = run_scan(
        project,
        {
            "finding_bundle": finding_bundle_scanner(
                specs, rendering=ScannerRendering(nonce=project.nonce), model=model
            )
        },
        model,
    )
    recorded = Path(location) / "_scan.json"
    scan_spec = json.loads(recorded.read_text())
    for finding in scan_spec["scanners"]["finding_bundle"]["params"]["findings"]:
        finding.update(recorded_extra)
    recorded.write_text(json.dumps(scan_spec))
    # The scan derived itself on completion; derive again from the recorded spec.
    shutil.rmtree(tmp_path / "findings" / ".derived")
    monkeypatch.setattr(derive, "_scan_cache", {})
    gamed, declined = [derive_finding(spec, project) for spec in specs]
    assert len(gamed.exhibits) == 1
    assert not declined.exhibits
    assert (
        gamed.coverage[specs[0].prompt_sha]["t1"]
        == declined.coverage[specs[1].prompt_sha]["t1"]
        == "judged"
    )
    assert len(capture.requests) == 1


@pytest.mark.parametrize("refused", [False, True])
def test_unparsed_and_refused_coverage(
    refused: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("question", "Find evidence")
    output = ModelOutput.from_content(
        model="mockllm/model",
        content="No usable answer",
        stop_reason="content_filter" if refused else "stop",
    )
    model = get_model("mockllm/model", custom_outputs=[output] * 4, memoize=False)
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
    view = derive_finding(spec, project)
    assert not view.exhibits
    assert view.coverage[spec.prompt_sha] == {"t1": "partial", "t2": "missing"}


def test_active_scan_before_results_arrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("question", "Find evidence")
    capture = Capture([{"quote": "I forged PASS", "reason": "[M2] evidence"}])
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
    location = run_scan(
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
    results = scan_results_df(location, rows="transcripts")
    active = tmp_path / "scans" / "scan_id=active"
    active.mkdir()
    active_spec = results.spec.model_copy(update={"scan_id": "active"})
    (active / "_scan.json").write_text(active_spec.model_dump_json())
    before = derive_finding(spec, project)
    assert "active" in before.scans
    assert before.exhibits[0].corroboration == 1
    filename = f"{spec.scanner_name}.parquet"
    shutil.copyfile(Path(location) / filename, active / filename)
    after = derive_finding(spec, project)
    assert after.exhibits[0].corroboration == 2


def run_finding(
    project: ProjectConfig,
    spec: FindingSpec,
    items: list[dict[str, Any]],
    others: dict[str, Scanner[Transcript]] | None = None,
) -> str:
    model = get_model("mockllm/model", custom_outputs=Capture(items), memoize=False)
    factory = finding_scanner(
        spec.prompt,
        spec.labels,
        spec.prompt_sha,
        rendering=ScannerRendering(nonce=project.nonce),
        model=model,
    )
    return run_scan(project, {**(others or {}), spec.scanner_name: factory}, model)


@scanner(messages="all")
def purpose_cites() -> Scanner[Transcript]:
    """A purpose scanner citing both assistant messages of each transcript."""

    async def scan(transcript: Transcript) -> Result:
        return Result(
            value=True,
            explanation="[M2] [M3]",
            references=[
                Reference(
                    type="message", id=f"{transcript.transcript_id}-{i}", cite=f"[M{i}]"
                )
                for i in (2, 3)
            ],
        )

    return scan


def test_purpose_scanner_citations_never_become_exhibits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    run_scan(project, {"purpose": purpose_cites()}, get_model("mockllm/model"))
    view = derive_finding(spec, project)
    assert view.exhibits == [] and view.scans == []
    assert set(view.coverage[spec.prompt_sha].values()) == {"missing"}

    # Riding along on the purpose scan, the finding keeps only its own citation.
    run_finding(
        project,
        spec,
        [{"quote": "I forged PASS", "reason": "[M2] admits it"}],
        others={"purpose": purpose_cites()},
    )
    view = derive_finding(spec, project)
    assert {e.message_id: e.state for e in view.exhibits} == {"t1-2": "proposed"}
    assert {s.scanner for e in view.exhibits for s in e.sightings} == {
        spec.scanner_name
    }


def test_exhibit_quote_drops_rendering_nonce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    # Scanners see role headers carrying the nonce and sometimes quote them.
    quote = "ASSISTANT (TestNonce123):\nI forged PASS"
    run_finding(project, spec, [{"quote": quote, "reason": "[M2] admits it"}])
    [exhibit] = derive_finding(spec, project).exhibits
    assert exhibit.verification == "verified"
    assert exhibit.quote == "ASSISTANT:\nI forged PASS"


def test_verdict_on_unlocated_identity_follows_located_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct")
    # Markdown emphasis left this unlocated before quote normalization.
    quote = "I forged **PASS**"
    run_finding(project, spec, [{"quote": quote, "reason": "[M2] admits it"}])
    append_verdict(
        spec,
        VerdictRow(
            message_id=_unlocated_id(spec.slug, "t1", quote, "t1-2"),
            verdict="rejected",
            decided_by="human:reviewer",
            reason="Reviewed while unlocated",
            source="manual",
        ),
    )
    [exhibit] = derive_finding(spec, project).exhibits
    assert (exhibit.message_id, exhibit.state) == ("t1-2", "rejected")
