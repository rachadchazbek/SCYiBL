"""Data checks (#3) that must run without the `data` extra installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scyibl import cli
from scyibl.taxonomy import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_missing_extra_exits_with_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str) -> object:
        return None if name == "pyarrow" else real_find_spec(name)

    monkeypatch.setattr(cli.importlib.util, "find_spec", find_spec)
    args = ["data", "prepare", "--input", "raw.csv", "--out-dir", str(tmp_path / "out")]
    assert cli.main(args) == 1
    assert capsys.readouterr().err.strip() == "Install the data extra: uv sync --extra data"
    assert not (tmp_path / "out").exists()


def test_labels_doc_matches_taxonomy() -> None:
    doc = (REPO_ROOT / "docs" / "labels.md").read_text()
    for info in CATEGORIES.values():
        assert f"## {info.label}" in doc
        assert info.definition in doc
    assert doc.count("Status: unconfirmed") == len(CATEGORIES)
