from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "model-cache"))
