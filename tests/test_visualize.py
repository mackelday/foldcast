"""Tests for foldcast.visualize."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foldcast._types import BacktestResult, FoldResult  # noqa: E402
from foldcast.visualize import (  # noqa: E402
    plot_backtest,
    plot_hierarchy_heatmap,
    plot_model_comparison,
    plot_residual_diagnostics,
)


def _sample_backtest_result():
    idx1 = pd.date_range("2020-01-01", periods=3, freq="MS")
    idx2 = pd.date_range("2020-04-01", periods=3, freq="MS")
    folds = [
        FoldResult(
            0,
            pd.Timestamp("2020-01-01"),
            pd.Series([100, 101, 102.0], index=idx1),
            pd.Series([100.5, 101.5, 101.8], index=idx1),
        ),
        FoldResult(
            1,
            pd.Timestamp("2020-04-01"),
            pd.Series([103, 104, 105.0], index=idx2),
            pd.Series([102.8, 104.2, 105.1], index=idx2),
        ),
    ]
    return BacktestResult(folds=folds, horizon=3, strategy="expanding")


class TestPlotBacktest:
    def test_returns_figure(self):
        fig = plot_backtest(_sample_backtest_result())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_full_series(self, monthly_series):
        fig = plot_backtest(_sample_backtest_result(), full_series=monthly_series)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotModelComparison:
    def test_returns_figure(self):
        scores = pd.DataFrame(
            {"mae": [1.0, 2.0, 3.0], "rmse": [1.5, 2.5, 3.5]},
            index=["model_a", "model_b", "model_c"],
        )
        fig = plot_model_comparison(scores)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResidualDiagnostics:
    def test_returns_figure(self):
        rng = np.random.default_rng(42)
        residuals = pd.Series(rng.normal(0, 1, 100))
        fig = plot_residual_diagnostics(residuals)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotHierarchyHeatmap:
    def test_returns_figure(self):
        df = pd.DataFrame(
            {
                "level": ["total", "region", "region", "city", "city", "city", "city"],
                "node": ["total", "East", "West", "NYC", "BOS", "LAX", "SEA"],
                "mae": [5.0, 3.0, 4.0, 2.0, 2.5, 3.0, 3.5],
                "rmse": [6.0, 3.5, 4.5, 2.5, 3.0, 3.5, 4.0],
            }
        )
        fig = plot_hierarchy_heatmap(df, metric="mae")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
