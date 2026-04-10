"""Internal validation and utility functions."""

from __future__ import annotations

import pandas as pd


def validate_series(s: pd.Series, name: str) -> pd.Series:
    """Validate that *s* is a non-empty pandas Series with DatetimeIndex."""
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(s.index).__name__}")
    if len(s) == 0:
        raise ValueError(f"{name} is empty")
    if s.isna().all():
        raise ValueError(f"{name} is all NaN")
    return s


def validate_freq(s: pd.Series, freq: str | None = None) -> str:
    """Return the frequency of *s*, inferring if needed."""
    if freq is not None:
        return freq
    inferred = pd.infer_freq(s.index)
    if inferred is None:
        raise ValueError(
            "Could not infer freq from series index. Pass freq= explicitly."
        )
    return inferred


def validate_same_index(a: pd.Series, b: pd.Series) -> None:
    """Raise if *a* and *b* do not share the same index."""
    if not a.index.equals(b.index):
        raise ValueError(
            f"Series index mismatch: {a.index[[0, -1]]} vs {b.index[[0, -1]]}"
        )
