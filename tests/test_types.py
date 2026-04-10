"""Tests for foldcast result dataclasses."""

from __future__ import annotations

import pandas as pd
import pytest

from foldcast._types import (
    BacktestResult,
    BiasResult,
    CoherenceResult,
    CombineResult,
    CoverageResult,
    DMResult,
    DriftResult,
    FoldResult,
    MCSResult,
    ResidualReport,
)


class TestFoldResult:
    def test_creation(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        fold = FoldResult(
            fold_id=0,
            cutoff=pd.Timestamp("2020-01-01"),
            forecasts=pd.Series([1.0, 2.0, 3.0], index=idx),
            actuals=pd.Series([1.1, 2.1, 2.9], index=idx),
        )
        assert fold.fold_id == 0
        assert len(fold.forecasts) == 3

    def test_immutable(self):
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
        fold = FoldResult(
            fold_id=0,
            cutoff=pd.Timestamp("2020-01-01"),
            forecasts=pd.Series([1.0, 2.0], index=idx),
            actuals=pd.Series([1.0, 2.0], index=idx),
        )
        with pytest.raises(AttributeError):
            fold.fold_id = 1


class TestBacktestResult:
    def test_to_dataframe(self):
        idx1 = pd.date_range("2020-01-01", periods=2, freq="D")
        idx2 = pd.date_range("2020-01-03", periods=2, freq="D")
        folds = [
            FoldResult(
                0,
                pd.Timestamp("2020-01-01"),
                pd.Series([1.0, 2.0], index=idx1),
                pd.Series([1.1, 2.1], index=idx1),
            ),
            FoldResult(
                1,
                pd.Timestamp("2020-01-03"),
                pd.Series([3.0, 4.0], index=idx2),
                pd.Series([3.1, 3.9], index=idx2),
            ),
        ]
        result = BacktestResult(folds=folds, horizon=2, strategy="expanding")
        df = result.to_dataframe()
        assert "forecast" in df.columns
        assert "actual" in df.columns
        assert "fold_id" in df.columns
        assert len(df) == 4

    def test_summary(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        folds = [
            FoldResult(
                0,
                pd.Timestamp("2020-01-01"),
                pd.Series([1.0, 2.0, 3.0], index=idx),
                pd.Series([1.0, 2.0, 3.0], index=idx),
            ),
        ]
        result = BacktestResult(folds=folds, horizon=3, strategy="expanding")
        s = result.summary()
        assert isinstance(s, str)
        assert "expanding" in s


class TestDMResult:
    def test_creation(self):
        dm = DMResult(statistic=2.1, p_value=0.03, conclusion="reject")
        assert dm.statistic == 2.1
        assert dm.p_value == 0.03

    def test_repr(self):
        dm = DMResult(statistic=2.1, p_value=0.03, conclusion="reject")
        assert "2.1" in repr(dm)


class TestMCSResult:
    def test_creation(self):
        mcs = MCSResult(
            included=["arima", "ets"],
            excluded=["prophet"],
            p_values={"arima": 1.0, "ets": 0.45, "prophet": 0.02},
            elimination_order=["prophet"],
            alpha=0.10,
        )
        assert "arima" in mcs.included
        assert "prophet" in mcs.excluded


class TestCombineResult:
    def test_creation(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        cr = CombineResult(
            combined=pd.Series([1.0, 2.0, 3.0], index=idx),
            weights={"a": 0.6, "b": 0.4},
            method="inverse_mse",
        )
        assert len(cr.combined) == 3
        assert abs(sum(cr.weights.values()) - 1.0) < 1e-9


class TestDriftResult:
    def test_creation(self):
        dr = DriftResult(statistic=0.25, threshold=0.2, alert=True, method="psi")
        assert dr.alert is True


class TestCoverageResult:
    def test_creation(self):
        cr = CoverageResult(
            nominal=0.95,
            empirical=0.91,
            n_observations=100,
            n_covered=91,
            alert=True,
        )
        assert cr.alert is True


class TestResidualReport:
    def test_creation(self):
        rr = ResidualReport(
            ljung_box_statistic=12.3,
            ljung_box_p_value=0.04,
            jarque_bera_statistic=5.1,
            jarque_bera_p_value=0.08,
            arch_statistic=3.2,
            arch_p_value=0.36,
            mean_residual=0.01,
            std_residual=1.02,
        )
        assert rr.ljung_box_p_value == 0.04

    def test_summary(self):
        rr = ResidualReport(
            ljung_box_statistic=12.3,
            ljung_box_p_value=0.04,
            jarque_bera_statistic=5.1,
            jarque_bera_p_value=0.08,
            arch_statistic=3.2,
            arch_p_value=0.36,
            mean_residual=0.01,
            std_residual=1.02,
        )
        s = rr.summary()
        assert "Ljung-Box" in s


class TestBiasResult:
    def test_creation(self):
        br = BiasResult(
            cumulative_bias=2.5,
            mean_bias=0.05,
            alert=False,
            threshold=1.0,
        )
        assert br.alert is False


class TestCoherenceResult:
    def test_creation(self):
        cr = CoherenceResult(
            is_coherent=False,
            max_violation=0.5,
            incoherent_nodes=["region_A"],
        )
        assert cr.is_coherent is False
