"""Tests for foldcast public API."""

from __future__ import annotations

import foldcast
from foldcast import backtest, compare, hierarchy, metrics, monitor, visualize


class TestPublicAPI:
    def test_version(self):
        assert hasattr(foldcast, "__version__")
        assert foldcast.__version__ == "0.1.0"

    def test_submodules_importable(self):
        assert hasattr(backtest, "expanding_window")
        assert hasattr(backtest, "sliding_window")
        assert hasattr(compare, "diebold_mariano")
        assert hasattr(compare, "model_confidence_set")
        assert hasattr(compare, "combine_forecasts")
        assert hasattr(compare, "rank_table")
        assert hasattr(monitor, "check_residuals")
        assert hasattr(monitor, "detect_drift")
        assert hasattr(monitor, "check_coverage")
        assert hasattr(monitor, "detect_bias")
        assert hasattr(hierarchy, "HierarchyTree")
        assert hasattr(hierarchy, "evaluate_levels")
        assert hasattr(hierarchy, "check_coherence")
        assert hasattr(metrics, "mae")
        assert hasattr(metrics, "mase")
        assert hasattr(metrics, "crps_gaussian")
        assert hasattr(visualize, "plot_backtest")
