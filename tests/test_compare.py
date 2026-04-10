"""Tests for foldcast.compare."""

from __future__ import annotations

import numpy as np
import pandas as pd

from foldcast._types import CombineResult, DMResult, MCSResult
from foldcast.compare import (
    combine_forecasts,
    diebold_mariano,
    model_confidence_set,
    rank_table,
)


class TestDieboldMariano:
    def test_identical_forecasts(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        forecast = actual + 0.1
        dm = diebold_mariano(forecast, forecast, actual)
        assert isinstance(dm, DMResult)
        assert abs(dm.statistic) < 1e-9
        assert dm.conclusion == "fail_to_reject"

    def test_different_forecasts(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0, 1, 200)
        good = actual + rng.normal(0, 0.1, 200)
        bad = actual + rng.normal(0, 2.0, 200)
        dm = diebold_mariano(good, bad, actual)
        assert isinstance(dm, DMResult)
        assert dm.p_value < 0.05

    def test_horizon_adjustment(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0, 1, 200)
        fa = actual + rng.normal(0, 0.5, 200)
        fb = actual + rng.normal(0, 1.5, 200)
        dm = diebold_mariano(fa, fb, actual, horizon=3)
        assert isinstance(dm, DMResult)


class TestModelConfidenceSet:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        actual = rng.normal(0, 1, n)
        forecasts = {
            "good": actual + rng.normal(0, 0.1, n),
            "medium": actual + rng.normal(0, 0.5, n),
            "bad": actual + rng.normal(0, 2.0, n),
        }
        mcs = model_confidence_set(forecasts, actual, alpha=0.10)
        assert isinstance(mcs, MCSResult)
        assert "good" in mcs.included
        assert len(mcs.included) + len(mcs.excluded) == 3

    def test_all_equal(self):
        rng = np.random.default_rng(42)
        n = 100
        actual = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.5, n)
        forecasts = {
            "a": actual + noise,
            "b": actual + noise,
        }
        mcs = model_confidence_set(forecasts, actual, alpha=0.10)
        assert len(mcs.included) == 2


class TestCombineForecasts:
    def test_equal_weights(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        f1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        f2 = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        actual = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], index=idx)
        result = combine_forecasts({"a": f1, "b": f2}, actual, method="equal")
        assert isinstance(result, CombineResult)
        np.testing.assert_allclose(result.combined.values, [1.5, 2.5, 3.5, 4.5, 5.5])
        assert result.weights == {"a": 0.5, "b": 0.5}

    def test_inverse_mse(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        f_good = actual + 0.1
        f_bad = actual + 1.0
        result = combine_forecasts(
            {"good": f_good, "bad": f_bad}, actual, method="inverse_mse"
        )
        assert result.weights["good"] > result.weights["bad"]


class TestRankTable:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        actual = rng.normal(0, 1, n)
        forecasts = {
            "model_a": actual + rng.normal(0, 0.1, n),
            "model_b": actual + rng.normal(0, 0.5, n),
        }
        table = rank_table(forecasts, actual, metrics=["mae", "rmse"])
        assert isinstance(table, pd.DataFrame)
        assert "mae" in table.columns
        assert "rmse" in table.columns
        assert len(table) == 2
