"""Tests for foldcast internal utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foldcast._utils import validate_freq, validate_same_index, validate_series


class TestValidateSeries:
    def test_valid_series(self, daily_index):
        s = pd.Series([1.0, 2.0, 3.0], index=daily_index[:3])
        result = validate_series(s, "test_input")
        assert isinstance(result, pd.Series)

    def test_rejects_no_datetime_index(self):
        s = pd.Series([1.0, 2.0], index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_series(s, "test_input")

    def test_rejects_empty(self):
        s = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        with pytest.raises(ValueError, match="empty"):
            validate_series(s, "test_input")

    def test_rejects_all_nan(self, daily_index):
        s = pd.Series([np.nan, np.nan], index=daily_index[:2])
        with pytest.raises(ValueError, match="NaN"):
            validate_series(s, "test_input")


class TestValidateFreq:
    def test_inferred_freq(self, daily_index):
        s = pd.Series(range(len(daily_index)), index=daily_index)
        freq = validate_freq(s)
        assert freq is not None

    def test_explicit_freq(self, daily_index):
        s = pd.Series(range(len(daily_index)), index=daily_index)
        freq = validate_freq(s, freq="D")
        assert freq == "D"

    def test_raises_when_no_freq(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-10"])
        s = pd.Series([1, 2, 3], index=idx)
        with pytest.raises(ValueError, match="freq"):
            validate_freq(s)


class TestValidateSameIndex:
    def test_matching_indices(self, daily_index):
        a = pd.Series(range(5), index=daily_index[:5])
        b = pd.Series(range(5), index=daily_index[:5])
        validate_same_index(a, b)

    def test_mismatched_indices(self, daily_index):
        a = pd.Series(range(5), index=daily_index[:5])
        b = pd.Series(range(5), index=daily_index[5:10])
        with pytest.raises(ValueError, match="index"):
            validate_same_index(a, b)
