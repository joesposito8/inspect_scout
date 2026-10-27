"""Findings routes exercised against real synthetic scans and ledgers."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inspect_ai.model import Model, get_model
from inspect_scout import ScannerRendering
from inspect_scout._findings import judge as judge_module
from inspect_scout._findings.answer import answer_headline
from inspect_scout._findings.derive import derive_finding
from inspect_scout._findings.scanner import finding_scanner
from inspect_scout._findings.spec import FindingSpec, read_finding, register_finding
from inspect_scout._findings.verdicts import VerdictRow, append_verdict, read_verdicts
from inspect_scout._project.types import ProjectConfig
from inspect_scout._view._api_v2 import v2_api_app
from inspect_scout._view._api_v2_findings import _display_quote

from tests.findings.test_derive import run_scan, setup_project
from tests.findings.test_judge import JudgeCapture
from tests.findings.test_scanner import Capture


@pytest.mark.parametrize(
    "quote,expected",
    [
        (
            "ASSISTANT (abcdefgh): <thinking abcdefgh>Evidence</thinking abcdefgh>",
            "Evidence",
        ),
        ("Tool Call (abcdefgh): run_command", "run_command"),
        ("[BEGIN 'command' (abcdefgh)]\nls\n[END 'command' (abcdefgh)]", "ls"),
        ("USER (abcdefgh): Evidence", "Evidence"),
        ("Evidence <thinking abcdefgh", "Evidence"),
        ("Actual <thinking> text", "Actual <thinking> text"),
    ],
)
def test_quote_display_preserves_content(quote: str, expected: str) -> None:
    assert _display_quote(quote, "abcdefgh") == expected


def test_quote_display_strips_only_matching_truncated_nonce() -> None:
    quote = "Evidence\n\nTool Call (AW44KoFAHpjE8"
    assert _display_quote(quote, "AW44KoFAHpjE8ndMpdzEnT") == "Evidence"
    assert _display_quote(quote, "anothernonce") == quote


@pytest.fixture
def finding_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "model-cache"))
    project = setup_project(tmp_path, monkeypatch)
    spec = register_finding("gamed", "Find misconduct", labels=["gamed", "declined"])
    model = get_model(
        "mockllm/model",
        custom_outputs=Capture(
            [
                dict(quote="I forged PASS", reason="[M2] admits it", label="gamed"),
                dict(quote="Absent quotation", reason="[M1] claims it", label="gamed"),
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
    with TestClient(v2_api_app()) as client:
        yield client


def test_list_detail_and_context(finding_client: TestClient) -> None:
    response = finding_client.get("/findings")
    assert response.status_code == 200
    status = response.json()[0]
    assert status["slug"] == "gamed"
    assert status["states"]["proposed"] == 2
    assert status["coverage"] == {"judged": 1, "partial": 0, "missing": 1}
    response = finding_client.get("/findings/gamed")
    assert response.status_code == 200
    detail = response.json()
    assert len(detail["exhibits"]) == 2
    assert len(detail["scans"]) == 1
    assert "## Computed" in detail["answer_markdown"]
    assert "findings-cache" not in detail["answer_markdown"]
    assert detail["message_links"]["t1-2"].startswith("#/transcripts/")
    assert detail["message_links"]["t1-2"].endswith(
        "?tab=transcript-messages&message=t1-2"
    )
    # Nothing is confirmed yet, so nothing is citable evidence.
    assert detail["citation_messages"] == {}
    assert "No confirmed exhibits yet; 2 proposed" in detail["answer_markdown"]
    finding_client.post(
        "/findings/gamed/verdicts", json={"message_id": "t1-2", "verdict": "confirmed"}
    )
    finding_client.post("/findings/gamed/answer")
    detail = finding_client.get("/findings/gamed").json()
    assert list(detail["citation_messages"].values()) == ["t1-2"]
    context = finding_client.get("/findings/gamed/context/t1-2").json()
    assert [m["index"] for m in context["messages"]] == [1, 2, 3]
    assert context["messages"][1]["message"]["content"] == "I forged PASS"
    assert "I forged PASS" in context["messages"][1]["rendered"]
    assert (
        len(finding_client.get("/findings/gamed/context/t1-2?k=0").json()["messages"])
        == 1
    )
    assert finding_client.get("/findings/gamed/context/t1-2?k=-1").status_code == 422
    assert finding_client.get("/findings/gamed/context/t1-2?k=101").status_code == 422
    assert finding_client.get("/findings/missing").status_code == 404
    assert finding_client.get("/findings/gamed/context/missing").status_code == 404
    unlocated = next(
        e["message_id"] for e in detail["exhibits"] if e["verification"] == "unlocated"
    )
    assert finding_client.get(f"/findings/gamed/context/{unlocated}").status_code == 404


def test_verdicts_append_and_override_judges(finding_client: TestClient) -> None:
    spec = read_finding("gamed")
    body = {
        "message_id": "t1-2",
        "verdict": "rejected",
        "label": "gamed",
        "reason": "Not evidence",
    }
    for expected_count in (1, 2):
        response = finding_client.post("/findings/gamed/verdicts", json=body)
        assert response.status_code == 201
        assert response.json()["answer_error"] is None
        assert len(read_verdicts(spec)) == expected_count
        assert read_verdicts(spec)[-1].label is None
        detail = finding_client.get("/findings/gamed").json()
        assert (
            next(e for e in detail["exhibits"] if e["message_id"] == "t1-2")["state"]
            == "rejected"
        )
        assert "rejected exhibits: 1" in detail["answer_markdown"]
    append_verdict(
        spec,
        VerdictRow(
            message_id="t1-2",
            verdict="proposed",
            label="gamed",
            decided_by="judge:mock",
            reason="Later judge",
            source="recheck",
        ),
    )
    assert (
        next(e for e in derive_finding(spec).exhibits if e.message_id == "t1-2").state
        == "rejected"
    )
    response = finding_client.post(
        "/findings/gamed/verdicts",
        json={"message_id": "t1-2", "verdict": "confirmed", "label": "declined"},
    )
    assert response.status_code == 201
    detail = finding_client.get("/findings/gamed").json()
    exhibit = next(e for e in detail["exhibits"] if e["message_id"] == "t1-2")
    assert (exhibit["state"], exhibit["label"]) == ("confirmed", "declined")
    before = len(read_verdicts(spec))
    for invalid in (
        {"message_id": "t1-2", "verdict": "confirmed", "label": "unknown"},
        {"message_id": "t1-2", "verdict": "proposed"},
        {"message_id": "t1-2", "verdict": "rejected", "decided_by": "judge:fake"},
    ):
        assert (
            finding_client.post("/findings/gamed/verdicts", json=invalid).status_code
            == 422
        )
    assert (
        finding_client.post(
            "/findings/gamed/verdicts",
            json={"message_id": "missing", "verdict": "rejected"},
        ).status_code
        == 404
    )
    assert len(read_verdicts(spec)) == before


def test_verdicts_defer_synthesis_until_regenerated(
    finding_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = Path("findings/gamed/answer.md")
    headline = answer_headline("gamed")
    prompts: list[str] = []

    async def fail_synthesis(prompt: str, model: object) -> str:
        prompts.append(prompt)
        raise RuntimeError("[bold]Synthesis unavailable[/bold]")

    monkeypatch.setattr(
        "inspect_scout._findings.answer.generate_synthesis", fail_synthesis
    )
    response = finding_client.post(
        "/findings/gamed/verdicts", json={"message_id": "t1-2", "verdict": "confirmed"}
    )
    assert response.status_code == 201
    assert response.json()["answer_error"] is None
    assert prompts == []
    assert len(read_verdicts(read_finding("gamed"))) == 1
    detail = finding_client.get("/findings/gamed").json()
    assert detail["status"]["states"]["confirmed"] == 1
    assert "confirmed exhibits: 1" in detail["answer_markdown"]
    assert detail["answer_stale"] is True
    assert "Out of date" not in detail["answer_markdown"].split("## Synthesis")[1]
    assert detail["status"]["headline"] == headline
    previous = answer.read_text()
    response = finding_client.post("/findings/gamed/answer")
    assert response.status_code == 500
    assert response.json()["detail"] == "Synthesis unavailable"
    assert len(prompts) == 1
    assert answer.read_text() == previous


def test_judge_decides_pending_exhibits_until_a_human_overrides(
    finding_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = get_model(
        "mockllm/judge",
        custom_outputs=JudgeCapture("confirmed", "gamed"),
        memoize=False,
    )

    def judge_model(model: str | Model) -> Model:
        return judge

    monkeypatch.setattr(judge_module, "get_model", judge_model)
    assert (
        finding_client.post("/findings/gamed/judge", json={"limit": 0}).status_code
        == 422
    )
    response = finding_client.post("/findings/gamed/judge", json={})
    assert response.status_code == 200
    result = response.json()
    assert result["skipped_unlocated"] == 1 and result["answer_error"] is None
    run = result["run"]
    assert (run["candidates"], run["confirmed"], run["refused"]) == (1, 1, 0)
    detail = finding_client.get("/findings/gamed").json()
    exhibit = next(e for e in detail["exhibits"] if e["message_id"] == "t1-2")
    assert (exhibit["state"], exhibit["decided_by"]) == (
        "confirmed",
        "judge:mockllm/judge",
    )
    assert detail["status"]["judge_refused"] == 0
    assert list(detail["citation_messages"].values()) == ["t1-2"]
    finding_client.post(
        "/findings/gamed/verdicts", json={"message_id": "t1-2", "verdict": "rejected"}
    )
    status = finding_client.get("/findings").json()[0]
    assert status["states"]["rejected"] == 1
    assert status["judge_agreement"] == (
        "judge agreement: kept 0 of 1 human-reviewed judge decisions"
    )
    again = finding_client.post("/findings/gamed/judge", json={}).json()
    assert again["run"]["candidates"] == 0


def test_unlocated_exhibit_cannot_be_confirmed(finding_client: TestClient) -> None:
    detail = finding_client.get("/findings/gamed").json()
    unlocated = next(
        e["message_id"] for e in detail["exhibits"] if e["verification"] == "unlocated"
    )
    confirm = {"message_id": unlocated, "verdict": "confirmed", "label": "gamed"}
    assert (
        finding_client.post("/findings/gamed/verdicts", json=confirm).status_code == 422
    )
    reject = {"message_id": unlocated, "verdict": "rejected"}
    assert (
        finding_client.post("/findings/gamed/verdicts", json=reject).status_code == 201
    )


def test_force_answer(
    finding_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_scout._findings.answer import refresh_answer

    forced: list[bool] = []

    async def record_answer(
        finding: FindingSpec, project: ProjectConfig, *, force: bool = False
    ) -> Path:
        forced.append(force)
        return await refresh_answer(finding, project, no_synthesis=True, force=force)

    monkeypatch.setattr(
        "inspect_scout._view._api_v2_findings.refresh_answer", record_answer
    )
    response = finding_client.post("/findings/gamed/answer")
    assert response.status_code == 200
    assert forced == [True]
    assert "Synthesis disabled" in response.json()["answer_markdown"]


def test_empty_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(v2_api_app()) as client:
        assert client.get("/findings").json() == []
