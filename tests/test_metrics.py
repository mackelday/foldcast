"""Tests for foldcast.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from foldcast.metrics import (
    crps_gaussian,
    mae,
    mape,
    mase,
    mdae,
    mdape,
    rmse,
    rmsse,
    smape,
    winkler_score,
)


@pytest.fixture()
def actuals():
    return np.array([3.0, -0.5, 2.0, 7.0])


@pytest.fixture()
def forecasts():
    return np.array([2.5, 0.0, 2.0, 8.0])


class TestScaleDependent:
    def test_mae(self, actuals, forecasts):
        result = mae(actuals, forecasts)
        assert abs(result - 0.5) < 1e-9

    def test_rmse(self, actuals, forecasts):
        result = rmse(actuals, forecasts)
        expected = np.sqrt(np.mean((actuals - forecasts) ** 2))
        assert abs(result - expected) < 1e-9

    def test_mdae(self, actuals, forecasts):
        result = mdae(actuals, forecasts)
        expected = np.median(np.abs(actuals - forecasts))
        assert abs(result - expected) < 1e-9


class TestPercentage:
    def test_mape(self, actuals, forecasts):
        result = mape(actuals, forecasts)
        assert result > 0

    def test_mape_zero_actual_raises(self):
        with pytest.raises(ValueError, match="zero"):
            mape(np.array([0.0, 1.0]), np.array([1.0, 1.0]))

    def test_smape(self, actuals, forecasts):
        result = smape(actuals, forecasts)
        assert 0 <= result <= 200

    def test_mdape(self, actuals, forecasts):
        result = mdape(actuals, forecasts)
        assert result > 0


class TestScaled:
    def test_mase(self):
        insample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        forecast = np.array([5.5, 7.5])
        result = mase(actual, forecast, insample, season=1)
        assert abs(result - 0.5) < 1e-9

    def test_mase_seasonal(self):
        insample = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0])
        actual = np.array([7.0, 8.0])
        forecast = np.array([6.5, 8.5])
        result = mase(actual, forecast, insample, season=2)
        assert result > 0

    def test_rmsse(self):
        insample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        forecast = np.array([5.5, 7.5])
        result = rmsse(actual, forecast, insample, season=1)
        assert result > 0


class TestDistributional:
    def test_crps_gaussian(self):
        result = crps_gaussian(
            actual=np.array([0.0]),
            mu=np.array([0.0]),
            sigma=np.array([1.0]),
        )
        assert result > 0
        result_wide = crps_gaussian(
            actual=np.array([0.0]),
            mu=np.array([0.0]),
            sigma=np.array([10.0]),
        )
        assert result_wide > result

    def test_winkler_score(self):
        actual = np.array([5.0, 10.0, 15.0])
        lower = np.array([4.0, 8.0, 14.0])
        upper = np.array([6.0, 12.0, 16.0])
        result = winkler_score(actual, lower, upper, alpha=0.10)
        assert result > 0

    def test_winkler_penalty_for_miss(self):
        actual = np.array([20.0])
        lower = np.array([4.0])
        upper = np.array([6.0])
        score_miss = winkler_score(actual, lower, upper, alpha=0.10)
        actual_in = np.array([5.0])
        score_hit = winkler_score(actual_in, lower, upper, alpha=0.10)
        assert score_miss > score_hit


class TestEdgeCases:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0
        assert rmse(y, y) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            mae(np.array([1.0, 2.0]), np.array([1.0]))
