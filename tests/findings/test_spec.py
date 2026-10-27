from pathlib import Path

import pytest
from inspect_scout._findings.spec import (
    FindingSpec,
    list_findings,
    read_finding,
    register_finding,
)
from inspect_scout._project._project import read_project
from inspect_scout._project.types import ProjectConfig


def test_defaults_and_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scout.yaml").write_text("# keep this\nmodel: mockllm/model\n")
    config = ProjectConfig()
    assert config.findings.context_window == 100_000
    assert config.findings.messages == ["assistant", "user"]
    spec = register_finding("gamed-the-grader", "Find evidence", labels=["gamed"])
    nonce = read_project().nonce
    assert nonce is not None and len(nonce) >= 8
    assert read_finding(spec.slug) == spec
    assert list_findings() == [spec]
    assert (
        register_finding(spec.slug, spec.prompt, labels=spec.labels).prompt_sha
        == spec.prompt_sha
    )
    updated = register_finding(spec.slug, "New question", labels=spec.labels)
    assert updated.prompt_sha != spec.prompt_sha
    assert updated.created == spec.created
    assert read_project().nonce == nonce
    assert "# keep this" in (tmp_path / "scout.yaml").read_text()
    assert read_project().model == "mockllm/model"


@pytest.mark.parametrize("slug", ["../escape", "all", "none", "bundle", "Bad Name"])
def test_reject_invalid_slug(
    slug: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        register_finding(slug, "Question")
    assert not (tmp_path / "scout.yaml").exists()


@pytest.mark.parametrize(
    "slug",
    ["agent-reasoned-about-being-monitored-by-the-grader-a", "a" * 48],
)
def test_reject_truncated_scanner_name(slug: str) -> None:
    with pytest.raises(ValueError, match="47 characters"):
        FindingSpec(slug=slug, title="Question", prompt="Find evidence")


def test_accept_maximum_scanner_name() -> None:
    spec = FindingSpec(slug="a" * 47, title="Question", prompt="Find evidence")
    assert len(spec.scanner_name) == 55
