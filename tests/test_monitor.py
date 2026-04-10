"""Tests for foldcast.monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foldcast._types import BiasResult, CoverageResult, DriftResult, ResidualReport
from foldcast.monitor import check_coverage, check_residuals, detect_bias, detect_drift


class TestCheckResiduals:
    def test_white_noise(self):
        rng = np.random.default_rng(42)
        residuals = pd.Series(rng.normal(0, 1, 500))
        report = check_residuals(residuals)
        assert isinstance(report, ResidualReport)
        assert report.ljung_box_p_value > 0.01

    def test_autocorrelated_residuals(self):
        rng = np.random.default_rng(42)
        n = 500
        residuals = np.zeros(n)
        residuals[0] = rng.normal()
        for i in range(1, n):
            residuals[i] = 0.8 * residuals[i - 1] + rng.normal(0, 0.5)
        report = check_residuals(pd.Series(residuals))
        assert report.ljung_box_p_value < 0.05

    def test_summary_string(self):
        rng = np.random.default_rng(42)
        report = check_residuals(pd.Series(rng.normal(0, 1, 200)))
        s = report.summary()
        assert "Ljung-Box" in s
        assert "Jarque-Bera" in s


class TestDetectDrift:
    def test_no_drift(self):
        rng = np.random.default_rng(42)
        historical = rng.normal(0, 1, 500)
        recent = rng.normal(0, 1, 50)
        result = detect_drift(historical, recent, method="psi")
        assert isinstance(result, DriftResult)
        assert result.alert is False

    def test_drift_detected(self):
        rng = np.random.default_rng(42)
        historical = rng.normal(0, 1, 500)
        recent = rng.normal(3, 1, 50)
        result = detect_drift(historical, recent, method="psi", threshold=0.2)
        assert result.alert is True
        assert result.statistic > 0.2

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            detect_drift(np.array([1.0]), np.array([1.0]), method="unknown")


class TestCheckCoverage:
    def test_good_coverage(self):
        rng = np.random.default_rng(42)
        n = 200
        actuals = rng.normal(0, 1, n)
        lower = actuals - 2.0
        upper = actuals + 2.0
        result = check_coverage(actuals, lower, upper, nominal=0.95)
        assert isinstance(result, CoverageResult)
        assert result.empirical == 1.0
        assert result.alert is False

    def test_poor_coverage(self):
        rng = np.random.default_rng(42)
        n = 200
        actuals = rng.normal(0, 1, n)
        lower = actuals - 0.01
        upper = actuals + 0.01
        actuals_shifted = actuals + 5.0
        result = check_coverage(actuals_shifted, lower, upper, nominal=0.95)
        assert result.empirical < 0.95
        assert result.alert is True


class TestDetectBias:
    def test_unbiased(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 200)
        result = detect_bias(residuals, threshold=0.5)
        assert isinstance(result, BiasResult)
        assert abs(result.mean_bias) < 0.5

    def test_biased(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(2.0, 0.5, 200)
        result = detect_bias(residuals, threshold=0.5)
        assert result.alert is True
        assert result.mean_bias > 0.5
