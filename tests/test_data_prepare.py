"""Dataset preparation (#3). Uses only the synthetic fixture; never the real dataset.

Needs the `data` extra (CI installs it); skipped without it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

for _module in ("pandas", "pyarrow", "iterstrat"):
    pytest.importorskip(_module, reason="needs the data extra: uv sync --extra data")

import pandas as pd  # noqa: E402

from scyibl.cli import main  # noqa: E402
from scyibl.data.prepare import (  # noqa: E402
    LABELS,
    ORIGINS,
    OUTPUT_COLUMNS,
    SPLITS,
    VERSIONED_PACKAGES,
    DataPrepError,
    prepare,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_raw.csv"
FIXTURE_ROWS = 41
KEPT_ROWS = FIXTURE_ROWS - 2  # one empty sentence, one duplicate
INVENTED_NAMES = ["Jordan Vale", "Priya Moonfield", "Tomas Quill", "Ana Brightwater"]


def _run(out_dir: Path, seed: int = 42) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    stats = prepare(FIXTURE, out_dir, seed=seed)
    parts = {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in SPLITS}
    return stats, parts


def _write_variant(tmp_path: Path, edit: str) -> Path:
    raw = pd.read_csv(FIXTURE)
    if edit == "bad_label":
        raw.loc[3, "PERMA"] = 2
    elif edit == "missing_column":
        raw = raw.drop(columns=["Coping"])
    path = tmp_path / "variant.csv"
    raw.to_csv(path, index=False)
    return path


def _fixture_texts() -> list[str]:
    """Normalised fixture sentences and names, for checking they never leak."""
    raw = pd.read_csv(FIXTURE)
    texts = [" ".join(str(s).split()) for s in raw["Sentence"].fillna("")]
    return [t for t in texts if t] + INVENTED_NAMES


def test_fixture_covers_the_cases_under_test() -> None:
    raw = pd.read_csv(FIXTURE)
    assert len(raw) == FIXTURE_ROWS
    assert int((raw["Sentence"].fillna("").str.strip() == "").sum()) == 1
    assert (raw[LABELS].sum() > 0).all()
    assert int((raw[LABELS].sum(axis=1) == 0).sum()) == 1
    assert set(INVENTED_NAMES) <= set(raw["Source"])


def test_outputs_contain_no_source_or_names(tmp_path: Path) -> None:
    _, parts = _run(tmp_path)
    for part in parts.values():
        assert list(part.columns) == OUTPUT_COLUMNS
        assert set(part["origin"]) <= set(ORIGINS)
        text = part.to_csv(index=False)
        for name in INVENTED_NAMES:
            assert name not in text
    stats_text = (tmp_path / "stats.json").read_text()
    assert "Source" not in stats_text
    for name in INVENTED_NAMES:
        assert name not in stats_text
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "stats.json",
        "test.parquet",
        "train.parquet",
        "validation.parquet",
    ]


def test_origin_mapping(tmp_path: Path) -> None:
    stats, _ = _run(tmp_path)
    report = stats["input"]
    assert isinstance(report, dict)
    # " Web " is normalised to "web"; the duplicate row (a name) is dropped before counting.
    assert report["origin"] == {"business": 5, "ptsd": 5, "web": 10, "participant": 19}


def test_stats_are_aggregate_only(tmp_path: Path) -> None:
    _run(tmp_path)
    stats_text = (tmp_path / "stats.json").read_text()
    assert not any(text in stats_text for text in _fixture_texts())

    def leaves(node: object, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], object]]:
        if isinstance(node, dict):
            return [leaf for key, value in node.items() for leaf in leaves(value, (*path, key))]
        return [(path, node)]

    for path, value in leaves(json.loads(stats_text)):
        if path[0] == "hashes":
            assert isinstance(value, str)
            assert re.fullmatch(r"[0-9a-f]{64}", value), path
        elif path[0] == "versions":
            assert isinstance(value, str)
            assert re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", value), path
        else:
            assert isinstance(value, int | float), path


def test_hashes_follow_the_seed(tmp_path: Path) -> None:
    stats_a, parts = _run(tmp_path / "a")
    stats_b, _ = _run(tmp_path / "b")
    stats_c, _ = _run(tmp_path / "c", seed=7)
    assert stats_a["hashes"] == stats_b["hashes"]
    assert stats_a["hashes"] != stats_c["hashes"]
    expected = {
        name: hashlib.sha256("\n".join(sorted(part["sentence"])).encode("utf-8")).hexdigest()
        for name, part in parts.items()
    }
    assert stats_a["hashes"] == expected


def test_versions_recorded(tmp_path: Path) -> None:
    stats, _ = _run(tmp_path)
    versions = stats["versions"]
    assert isinstance(versions, dict)
    assert list(versions) == list(VERSIONED_PACKAGES)
    assert versions["pandas"] == pd.__version__


def test_empty_sentences_dropped(tmp_path: Path) -> None:
    stats, parts = _run(tmp_path)
    report = stats["input"]
    assert isinstance(report, dict)
    assert report["empty_sentences_dropped"] == 1
    assert "empty_sentences" not in report
    assert all(part["sentence"].str.len().gt(0).all() for part in parts.values())


def test_duplicates_dropped_and_whitespace_normalised(tmp_path: Path) -> None:
    stats, parts = _run(tmp_path)
    report = stats["input"]
    assert isinstance(report, dict)
    assert report["rows_read"] == FIXTURE_ROWS
    assert report["duplicates_dropped"] == 1
    assert report["duplicate_groups_with_conflicting_labels"] == 0
    assert report["rows_kept"] == KEPT_ROWS
    assert report["rows_without_label"] == 1  # kept, only reported
    sentences = pd.concat(parts.values())["sentence"]
    assert sentences.is_unique
    assert all(s == " ".join(s.split()) for s in sentences)


def test_split_ratios(tmp_path: Path) -> None:
    _, parts = _run(tmp_path)
    total = sum(len(part) for part in parts.values())
    assert total == KEPT_ROWS
    for name, ratio in SPLITS.items():
        assert abs(len(parts[name]) - ratio * total) <= 1


def test_every_category_has_positives_in_every_split(tmp_path: Path) -> None:
    stats, parts = _run(tmp_path)
    splits = stats["splits"]
    assert isinstance(splits, dict)
    for name, part in parts.items():
        for label in LABELS:
            assert part[label].sum() > 0, (name, label)
            assert splits[name]["positives"][label] == int(part[label].sum())


def test_same_seed_is_deterministic(tmp_path: Path) -> None:
    stats_a, parts_a = _run(tmp_path / "a")
    stats_b, parts_b = _run(tmp_path / "b")
    assert stats_a == stats_b
    for name in SPLITS:
        pd.testing.assert_frame_equal(parts_a[name], parts_b[name])


def test_refuses_to_write_inside_this_repo() -> None:
    target = REPO_ROOT / "outputs" / "should-not-exist"
    with pytest.raises(DataPrepError, match="inside a git repository"):
        prepare(FIXTURE, target)
    assert not target.exists()


def test_refuses_any_git_working_tree(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    with pytest.raises(DataPrepError, match="inside a git repository"):
        prepare(FIXTURE, tmp_path / "nested" / "out")
    assert not (tmp_path / "nested").exists()


def test_rejects_non_binary_labels_without_leaking_rows(tmp_path: Path) -> None:
    path = _write_variant(tmp_path, "bad_label")
    with pytest.raises(DataPrepError) as excinfo:
        prepare(path, tmp_path / "out")
    assert "'PERMA': 1 value(s) are not 0/1" in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_rejects_missing_columns(tmp_path: Path) -> None:
    path = _write_variant(tmp_path, "missing_column")
    with pytest.raises(DataPrepError, match="Coping"):
        prepare(path, tmp_path / "out")


def test_cli_prints_aggregate_stats_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"
    assert main(["data", "prepare", "--input", str(FIXTURE), "--out-dir", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert f'"rows_kept": {KEPT_ROWS}' in printed
    for text in _fixture_texts():
        assert text not in printed


def test_cli_refusal_exits_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    target = REPO_ROOT / "outputs" / "should-not-exist"
    assert main(["data", "prepare", "--input", str(FIXTURE), "--out-dir", str(target)]) == 1
    assert "inside a git repository" in capsys.readouterr().err
