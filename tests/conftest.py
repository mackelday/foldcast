"""Shared test fixtures for foldcast test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def daily_index() -> pd.DatetimeIndex:
    """200 days of daily dates."""
    return pd.date_range("2020-01-01", periods=200, freq="D")


@pytest.fixture()
def monthly_index() -> pd.DatetimeIndex:
    """60 months of monthly dates."""
    return pd.date_range("2018-01-01", periods=60, freq="MS")


@pytest.fixture()
def random_series(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Random walk time series, 200 daily observations."""
    rng = np.random.default_rng(42)
    values = 100 + np.cumsum(rng.normal(0, 1, len(daily_index)))
    return pd.Series(values, index=daily_index, name="value")


@pytest.fixture()
def monthly_series(monthly_index: pd.DatetimeIndex) -> pd.Series:
    """Monthly time series with trend and seasonality, 60 observations."""
    rng = np.random.default_rng(42)
    t = np.arange(len(monthly_index), dtype=float)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 2, len(monthly_index))
    return pd.Series(trend + seasonal + noise, index=monthly_index, name="value")


@pytest.fixture()
def naive_model():
    """Naive forecast model: repeats last observed value for all horizons."""

    def _model(train: pd.Series, horizon: int) -> pd.Series:
        last_val = train.iloc[-1]
        future_index = pd.date_range(
            train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
        )
        return pd.Series(last_val, index=future_index, name=train.name)

    return _model


@pytest.fixture()
def seasonal_naive_model():
    """Seasonal naive: repeats the last seasonal cycle."""

    def _model(train: pd.Series, horizon: int, season_length: int = 12) -> pd.Series:
        last_cycle = train.iloc[-season_length:]
        reps = (horizon // season_length) + 1
        forecasts = np.tile(last_cycle.values, reps)[:horizon]
        future_index = pd.date_range(
            train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
        )
        return pd.Series(forecasts, index=future_index, name=train.name)

    return _model
