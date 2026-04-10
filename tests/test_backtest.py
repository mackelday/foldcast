"""Tests for foldcast.backtest."""

from __future__ import annotations

import pandas as pd
import pytest

from foldcast._types import BacktestResult
from foldcast.backtest import expanding_window, sliding_window


class TestExpandingWindow:
    def test_basic(self, monthly_series, naive_model):
        result = expanding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            min_train_size=24,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy == "expanding"
        assert result.horizon == 3
        assert len(result.folds) > 0

    def test_fold_count(self, monthly_series, naive_model):
        # 60 obs, min_train=24, horizon=3, step=3
        # cutoffs at indices 23,26,29,...,56 => (56-23)//3 + 1 = 12 folds
        result = expanding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            min_train_size=24,
        )
        assert len(result.folds) == 12

    def test_folds_are_non_overlapping_with_train(self, monthly_series, naive_model):
        result = expanding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            min_train_size=24,
        )
        for fold in result.folds:
            assert fold.forecasts.index[0] > fold.cutoff

    def test_embargo(self, monthly_series, naive_model):
        result = expanding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            min_train_size=24,
            embargo=2,
        )
        assert len(result.folds) > 0

    def test_to_dataframe(self, monthly_series, naive_model):
        result = expanding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            min_train_size=24,
        )
        df = result.to_dataframe()
        assert "forecast" in df.columns
        assert "actual" in df.columns
        assert len(df) == len(result.folds) * 3

    def test_rejects_bad_input(self, naive_model):
        s = pd.Series([1, 2, 3], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            expanding_window(data=s, model_fn=naive_model, horizon=1)


class TestSlidingWindow:
    def test_basic(self, monthly_series, naive_model):
        result = sliding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            window_size=24,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy == "sliding"
        assert len(result.folds) > 0

    def test_fixed_train_size(self, monthly_series, naive_model):
        result = sliding_window(
            data=monthly_series,
            model_fn=naive_model,
            horizon=3,
            step=3,
            window_size=24,
        )
        for fold in result.folds:
            train = monthly_series[monthly_series.index <= fold.cutoff]
            assert len(train) >= 24
