"""Anonymise, clean and split the labelled dataset into frozen train/validation/test sets (#3).

Privacy: the input holds participant names and mental-health text. Nothing in this module may
print, log or raise with row content; errors and ``stats.json`` are aggregate only.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from scyibl.taxonomy import CATEGORIES

LABELS: list[str] = [info.label for info in CATEGORIES.values()]
SOURCE_COLUMN = "Source"
SENTENCE_COLUMN = "Sentence"
KNOWN_ORIGINS = frozenset({"web", "business", "ptsd"})
"""Coarse values of ``Source``. Anything else (e.g. a person's name) becomes ``participant``."""
PARTICIPANT = "participant"
ORIGINS = (*sorted(KNOWN_ORIGINS), PARTICIPANT)
OUTPUT_COLUMNS: list[str] = ["sentence", "origin", *LABELS]
SPLITS: dict[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_SEED = 42
VERSIONED_PACKAGES = ("pandas", "numpy", "iterative-stratification", "scyibl")


class DataPrepError(ValueError):
    """Invalid input or output location. Messages never contain row content."""


def find_git_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def check_out_dir(out_dir: Path) -> Path:
    """Resolve ``out_dir`` and refuse any location inside a git working tree."""
    resolved = out_dir.expanduser().resolve()
    root = find_git_root(resolved)
    if root is not None:
        raise DataPrepError(
            f"refusing to write inside a git repository ({root}); "
            "use a private directory outside it (see #16)"
        )
    return resolved


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def to_origin(source: pd.Series) -> pd.Series:
    coarse = source.fillna("").astype(str).str.strip().str.lower()
    return coarse.where(coarse.isin(KNOWN_ORIGINS), PARTICIPANT)


def _validated_labels(raw: pd.DataFrame) -> pd.DataFrame:
    labels = {}
    for label in LABELS:
        values = pd.to_numeric(raw[label], errors="coerce")
        invalid = int((~values.isin([0, 1])).sum())
        if invalid:
            raise DataPrepError(f"column {label!r}: {invalid} value(s) are not 0/1")
        labels[label] = values.astype("int8")
    return pd.DataFrame(labels, index=raw.index)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop index/Source columns, map origin, normalise text, drop empty and duplicate rows."""
    missing = [c for c in (SOURCE_COLUMN, SENTENCE_COLUMN, *LABELS) if c not in raw.columns]
    if missing:
        raise DataPrepError(f"missing column(s): {missing}")

    df = pd.DataFrame(
        {
            "sentence": raw[SENTENCE_COLUMN].fillna("").astype(str).map(normalise_whitespace),
            "origin": to_origin(raw[SOURCE_COLUMN]),
        }
    )
    df = df.join(_validated_labels(raw))

    empty = df["sentence"] == ""
    df = df[~empty]
    duplicated = df["sentence"].duplicated(keep="first")
    repeated = df[df["sentence"].duplicated(keep=False)]
    conflicting = int((repeated.groupby("sentence")[LABELS].nunique() > 1).any(axis=1).sum())
    df = df[~duplicated].reset_index(drop=True)[OUTPUT_COLUMNS]

    report = {
        "rows_read": len(raw),
        "empty_sentences_dropped": int(empty.sum()),
        "duplicates_dropped": int(duplicated.sum()),
        "duplicate_groups_with_conflicting_labels": conflicting,
        "rows_kept": len(df),
        "rows_without_label": _rows_without_label(df),
        "origin": _origin_counts(df),
    }
    return df, report


def split(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    """Iterative stratified multi-label split (Sechidis et al., 2011) into SPLITS ratios."""
    y = df[LABELS].to_numpy()
    rows = np.arange(len(df))

    test_cut = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=SPLITS["test"], random_state=seed
    )
    rest, test = next(test_cut.split(rows.reshape(-1, 1), y))
    val_share = SPLITS["validation"] / (SPLITS["train"] + SPLITS["validation"])
    val_cut = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    train, validation = next(val_cut.split(rest.reshape(-1, 1), y[rest]))

    indices = {"train": rest[train], "validation": rest[validation], "test": test}
    parts = {name: df.iloc[np.sort(ix)].reset_index(drop=True) for name, ix in indices.items()}

    empty = {
        name: [label for label in LABELS if int(part[label].sum()) == 0]
        for name, part in parts.items()
    }
    empty = {name: labels for name, labels in empty.items() if labels}
    if empty:
        raise DataPrepError(f"categories without positives in a split: {empty}")
    return parts


def split_stats(parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        name: {
            "rows": len(part),
            "rows_without_label": _rows_without_label(part),
            "positives": {label: int(part[label].sum()) for label in LABELS},
            "origin": _origin_counts(part),
        }
        for name, part in parts.items()
    }


def split_hashes(parts: dict[str, pd.DataFrame]) -> dict[str, str]:
    """SHA-256 of each split's sentences, sorted and joined with newlines (UTF-8)."""
    return {
        name: hashlib.sha256("\n".join(sorted(part["sentence"])).encode("utf-8")).hexdigest()
        for name, part in parts.items()
    }


def prepare(input_path: Path, out_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Run the whole pipeline and write ``<split>.parquet`` and ``stats.json`` to ``out_dir``."""
    target = check_out_dir(out_dir)
    raw = pd.read_csv(input_path, dtype={SOURCE_COLUMN: str, SENTENCE_COLUMN: str})
    df, report = clean(raw)
    parts = split(df, seed)

    stats = {
        "seed": seed,
        "ratios": SPLITS,
        "input": report,
        "splits": split_stats(parts),
        "hashes": split_hashes(parts),
        "versions": {package: version(package) for package in VERSIONED_PACKAGES},
    }
    target.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(target / f"{name}.parquet", index=False)
    (target / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _rows_without_label(df: pd.DataFrame) -> int:
    return int((df[LABELS].sum(axis=1) == 0).sum())


def _origin_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["origin"].value_counts()
    return {origin: int(counts.get(origin, 0)) for origin in ORIGINS}
