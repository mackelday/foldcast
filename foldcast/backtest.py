"""Temporal cross-validation for forecast backtesting."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from foldcast._types import BacktestResult, FoldResult
from foldcast._utils import validate_freq, validate_series


def expanding_window(
    data: pd.Series,
    model_fn: Callable[[pd.Series, int], pd.Series],
    horizon: int,
    step: int = 1,
    min_train_size: int | None = None,
    embargo: int = 0,
    freq: str | None = None,
) -> BacktestResult:
    """Run expanding-window backtest.

    Args:
        data: Time series with DatetimeIndex.
        model_fn: Callable(train_series, horizon) -> forecast_series.
        horizon: Number of periods to forecast at each fold.
        step: Number of periods to advance the cutoff between folds.
        min_train_size: Minimum training observations before first fold.
            Defaults to 2 * horizon.
        embargo: Number of periods to skip between training end and forecast start.
        freq: Frequency string. Inferred from data if not provided.

    Returns:
        BacktestResult with all fold results.
    """
    validate_series(data, "data")
    validate_freq(data, freq)
    if min_train_size is None:
        min_train_size = 2 * horizon

    folds = _generate_folds(
        data=data,
        model_fn=model_fn,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        embargo=embargo,
        window_size=None,
    )
    return BacktestResult(folds=folds, horizon=horizon, strategy="expanding")


def sliding_window(
    data: pd.Series,
    model_fn: Callable[[pd.Series, int], pd.Series],
    horizon: int,
    step: int = 1,
    window_size: int = 30,
    embargo: int = 0,
    freq: str | None = None,
) -> BacktestResult:
    """Run sliding-window backtest.

    Args:
        data: Time series with DatetimeIndex.
        model_fn: Callable(train_series, horizon) -> forecast_series.
        horizon: Number of periods to forecast at each fold.
        step: Number of periods to advance the cutoff between folds.
        window_size: Fixed number of training observations per fold.
        embargo: Number of periods to skip between training end and forecast start.
        freq: Frequency string. Inferred from data if not provided.

    Returns:
        BacktestResult with all fold results.
    """
    validate_series(data, "data")
    validate_freq(data, freq)

    folds = _generate_folds(
        data=data,
        model_fn=model_fn,
        horizon=horizon,
        step=step,
        min_train_size=window_size,
        embargo=embargo,
        window_size=window_size,
    )
    return BacktestResult(folds=folds, horizon=horizon, strategy="sliding")


def _generate_folds(
    data: pd.Series,
    model_fn: Callable[[pd.Series, int], pd.Series],
    horizon: int,
    step: int,
    min_train_size: int,
    embargo: int,
    window_size: int | None,
) -> list[FoldResult]:
    """Core fold generation logic shared by expanding and sliding window."""
    n = len(data)
    folds: list[FoldResult] = []
    fold_id = 0
    cutoff_idx = min_train_size - 1

    while cutoff_idx + embargo + horizon < n:
        if window_size is not None:
            train_start = max(0, cutoff_idx - window_size + 1)
            train = data.iloc[train_start : cutoff_idx + 1]
        else:
            train = data.iloc[: cutoff_idx + 1]

        test_start = cutoff_idx + 1 + embargo
        test_end = test_start + horizon
        if test_end > n:
            break

        actuals = data.iloc[test_start:test_end]
        forecasts = model_fn(train, horizon)

        if not forecasts.index.equals(actuals.index):
            forecasts = pd.Series(
                forecasts.values[:horizon], index=actuals.index, name=forecasts.name
            )

        folds.append(
            FoldResult(
                fold_id=fold_id,
                cutoff=data.index[cutoff_idx],
                forecasts=forecasts,
                actuals=actuals,
            )
        )

        fold_id += 1
        cutoff_idx += step

    return folds
