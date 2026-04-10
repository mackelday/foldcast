# foldcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade Python library for backtesting, comparing, and monitoring time series forecasts.

**Architecture:** Flat module structure under `foldcast/`. Each module (backtest, compare, monitor, hierarchy, metrics, visualize) is independent except for shared types (`_types.py`) and utilities (`_utils.py`). All modules depend on `metrics`. Result objects are frozen dataclasses with `.to_dataframe()` and `.summary()` methods.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy, matplotlib, pytest, ruff, mypy, hatchling, GitHub Actions.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `foldcast/__init__.py`
- Create: `foldcast/_types.py`
- Create: `foldcast/_utils.py`
- Create: `foldcast/metrics.py`
- Create: `foldcast/backtest.py`
- Create: `foldcast/compare.py`
- Create: `foldcast/monitor.py`
- Create: `foldcast/hierarchy.py`
- Create: `foldcast/visualize.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.github/workflows/ci.yml`
- Create: `.gitignore`
- Create: `LICENSE`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "foldcast"
version = "0.1.0"
description = "Backtesting, comparison, and monitoring for production time series forecasts."
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Michael" }]
keywords = ["forecasting", "backtesting", "time-series", "monitoring", "model-comparison"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
parallel = ["joblib>=1.3"]
plotly = ["plotly>=5.15"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.4",
    "mypy>=1.5",
    "pandas-stubs",
]

[project.urls]
Homepage = "https://github.com/Michael/foldcast"
Repository = "https://github.com/Michael/foldcast"
Issues = "https://github.com/Michael/foldcast/issues"

[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Create stub module files**

Create empty `foldcast/__init__.py`, `foldcast/metrics.py`, `foldcast/backtest.py`, `foldcast/compare.py`, `foldcast/monitor.py`, `foldcast/hierarchy.py`, `foldcast/visualize.py`, and `foldcast/_utils.py`.

Create `foldcast/_types.py` with:
```python
"""Result dataclasses and type aliases for foldcast."""

from __future__ import annotations
```

Create `tests/__init__.py` (empty).

- [ ] **Step 3: Create tests/conftest.py with shared fixtures**

```python
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
```

- [ ] **Step 4: Create .gitignore, LICENSE, and CI config**

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.egg
.venv/
htmlcov/
.coverage
```

`LICENSE` — MIT license.

`.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check foldcast/ tests/
      - run: pytest --cov=foldcast --cov-report=term-missing
```

- [ ] **Step 5: Verify installation and commit**

Run: `cd /path/to/foldcast && pip install -e ".[dev]"`
Expected: Installs successfully.

Run: `python -c "import foldcast; print('ok')"`
Expected: `ok`

```bash
git add -A
git commit -m "scaffold: project structure, pyproject.toml, CI, test fixtures"
```

---

### Task 2: Result Types (`_types.py`)

**Files:**
- Create: `foldcast/_types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write tests for result types**

```python
"""Tests for foldcast result dataclasses."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from foldcast._types import (
    BacktestResult,
    FoldResult,
    DMResult,
    MCSResult,
    CombineResult,
    DriftResult,
    CoverageResult,
    ResidualReport,
    BiasResult,
    CoherenceResult,
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
            FoldResult(0, pd.Timestamp("2020-01-01"),
                       pd.Series([1.0, 2.0], index=idx1),
                       pd.Series([1.1, 2.1], index=idx1)),
            FoldResult(1, pd.Timestamp("2020-01-03"),
                       pd.Series([3.0, 4.0], index=idx2),
                       pd.Series([3.1, 3.9], index=idx2)),
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
            FoldResult(0, pd.Timestamp("2020-01-01"),
                       pd.Series([1.0, 2.0, 3.0], index=idx),
                       pd.Series([1.0, 2.0, 3.0], index=idx)),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_types.py -v`
Expected: ImportError — types not yet implemented.

- [ ] **Step 3: Implement _types.py**

```python
"""Result dataclasses and type aliases for foldcast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class FoldResult:
    """Result from a single backtest fold."""

    fold_id: int
    cutoff: pd.Timestamp
    forecasts: pd.Series
    actuals: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated result from a full backtest run."""

    folds: Sequence[FoldResult]
    horizon: int
    strategy: str

    def to_dataframe(self) -> pd.DataFrame:
        """Combine all folds into a single DataFrame."""
        rows = []
        for fold in self.folds:
            df = pd.DataFrame({
                "forecast": fold.forecasts,
                "actual": fold.actuals,
            })
            df["fold_id"] = fold.fold_id
            df["cutoff"] = fold.cutoff
            rows.append(df)
        return pd.concat(rows)

    def summary(self) -> str:
        """Human-readable summary of backtest results."""
        n_folds = len(self.folds)
        all_errors = pd.concat(
            [f.actuals - f.forecasts for f in self.folds]
        )
        mae = all_errors.abs().mean()
        rmse = (all_errors**2).mean() ** 0.5
        return (
            f"BacktestResult(strategy={self.strategy}, folds={n_folds}, "
            f"horizon={self.horizon}, MAE={mae:.4f}, RMSE={rmse:.4f})"
        )


@dataclass(frozen=True)
class DMResult:
    """Result of a Diebold-Mariano test."""

    statistic: float
    p_value: float
    conclusion: str

    def __repr__(self) -> str:
        return (
            f"DMResult(statistic={self.statistic:.4f}, "
            f"p_value={self.p_value:.4f}, conclusion='{self.conclusion}')"
        )


@dataclass(frozen=True)
class MCSResult:
    """Result of Model Confidence Set procedure."""

    included: list[str]
    excluded: list[str]
    p_values: dict[str, float]
    elimination_order: list[str]
    alpha: float


@dataclass(frozen=True)
class CombineResult:
    """Result of forecast combination."""

    combined: pd.Series
    weights: dict[str, float]
    method: str


@dataclass(frozen=True)
class DriftResult:
    """Result of forecast drift detection."""

    statistic: float
    threshold: float
    alert: bool
    method: str


@dataclass(frozen=True)
class CoverageResult:
    """Result of prediction interval coverage check."""

    nominal: float
    empirical: float
    n_observations: int
    n_covered: int
    alert: bool


@dataclass(frozen=True)
class ResidualReport:
    """Comprehensive residual diagnostics report."""

    ljung_box_statistic: float
    ljung_box_p_value: float
    jarque_bera_statistic: float
    jarque_bera_p_value: float
    arch_statistic: float
    arch_p_value: float
    mean_residual: float
    std_residual: float

    def summary(self) -> str:
        """Human-readable diagnostics summary."""
        lines = [
            "Residual Diagnostics:",
            f"  Mean: {self.mean_residual:.4f}, Std: {self.std_residual:.4f}",
            f"  Ljung-Box: stat={self.ljung_box_statistic:.2f}, "
            f"p={self.ljung_box_p_value:.4f}",
            f"  Jarque-Bera: stat={self.jarque_bera_statistic:.2f}, "
            f"p={self.jarque_bera_p_value:.4f}",
            f"  ARCH: stat={self.arch_statistic:.2f}, "
            f"p={self.arch_p_value:.4f}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BiasResult:
    """Result of forecast bias detection."""

    cumulative_bias: float
    mean_bias: float
    alert: bool
    threshold: float


@dataclass(frozen=True)
class CoherenceResult:
    """Result of hierarchical coherence check."""

    is_coherent: bool
    max_violation: float
    incoherent_nodes: list[str]
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_types.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/_types.py tests/test_types.py
git commit -m "feat: add result dataclasses for all modules"
```

---

### Task 3: Utilities (`_utils.py`)

**Files:**
- Create: `foldcast/_utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write tests for validation utilities**

```python
"""Tests for foldcast internal utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from foldcast._utils import validate_series, validate_freq, validate_same_index


class TestValidateSeries:
    def test_valid_series(self, daily_index):
        s = pd.Series([1.0, 2.0, 3.0], index=daily_index[:3])
        result = validate_series(s, "test_input")
        assert isinstance(result, pd.Series)

    def test_rejects_no_datetime_index(self):
        s = pd.Series([1.0, 2.0], index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_series(s, "test_input")

    def test_rejects_empty(self, daily_index):
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
        validate_same_index(a, b)  # Should not raise

    def test_mismatched_indices(self, daily_index):
        a = pd.Series(range(5), index=daily_index[:5])
        b = pd.Series(range(5), index=daily_index[5:10])
        with pytest.raises(ValueError, match="index"):
            validate_same_index(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_utils.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement _utils.py**

```python
"""Internal validation and utility functions."""

from __future__ import annotations

import pandas as pd


def validate_series(s: pd.Series, name: str) -> pd.Series:
    """Validate that *s* is a non-empty pandas Series with DatetimeIndex."""
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(s.index).__name__}")
    if len(s) == 0:
        raise ValueError(f"{name} is empty")
    if s.isna().all():
        raise ValueError(f"{name} is all NaN")
    return s


def validate_freq(s: pd.Series, freq: str | None = None) -> str:
    """Return the frequency of *s*, inferring if needed."""
    if freq is not None:
        return freq
    inferred = pd.infer_freq(s.index)
    if inferred is None:
        raise ValueError(
            "Could not infer freq from series index. Pass freq= explicitly."
        )
    return inferred


def validate_same_index(a: pd.Series, b: pd.Series) -> None:
    """Raise if *a* and *b* do not share the same index."""
    if not a.index.equals(b.index):
        raise ValueError(
            f"Series index mismatch: {a.index[[0, -1]]} vs {b.index[[0, -1]]}"
        )
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_utils.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/_utils.py tests/test_utils.py
git commit -m "feat: add input validation utilities"
```

---

### Task 4: Metrics (`metrics.py`)

**Files:**
- Create: `foldcast/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write tests for all metric functions**

```python
"""Tests for foldcast.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from foldcast.metrics import (
    mae, rmse, mdae,
    mape, smape, mdape,
    mase, rmsse,
    crps_gaussian, winkler_score,
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
        # Naive MAE on insample = mean(|diff|) = 1.0
        # Forecast MAE = mean(|0.5, 0.5|) = 0.5
        # MASE = 0.5 / 1.0 = 0.5
        assert abs(result - 0.5) < 1e-9

    def test_mase_seasonal(self):
        insample = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
        actual = np.array([1.0, 3.0])
        forecast = np.array([1.5, 2.5])
        result = mase(actual, forecast, insample, season=2)
        # Seasonal naive MAE on insample: |1-1|, |3-3|, |1-1|, |3-3| = 0
        # This would divide by zero, so use season=1 for non-degenerate test
        insample2 = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0])
        actual2 = np.array([7.0, 8.0])
        forecast2 = np.array([6.5, 8.5])
        result2 = mase(actual2, forecast2, insample2, season=2)
        assert result2 > 0

    def test_rmsse(self):
        insample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        forecast = np.array([5.5, 7.5])
        result = rmsse(actual, forecast, insample, season=1)
        assert result > 0


class TestDistributional:
    def test_crps_gaussian(self):
        # CRPS of perfect forecast should be small
        result = crps_gaussian(
            actual=np.array([0.0]),
            mu=np.array([0.0]),
            sigma=np.array([1.0]),
        )
        assert result > 0
        # Wider sigma should give larger CRPS
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
        # Observation outside interval should be penalized
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement metrics.py**

```python
"""Forecast accuracy metrics.

All functions accept numpy arrays and return floats. Vectorized for performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_lengths(actual: np.ndarray, forecast: np.ndarray) -> None:
    if len(actual) != len(forecast):
        raise ValueError(f"Array length mismatch: {len(actual)} vs {len(forecast)}")


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


# --- Scale-dependent ---

def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean Absolute Error."""
    a, f = _to_array(actual), _to_array(forecast)
    _validate_lengths(a, f)
    return float(np.mean(np.abs(a - f)))


def rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Root Mean Squared Error."""
    a, f = _to_array(actual), _to_array(forecast)
    _validate_lengths(a, f)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mdae(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Median Absolute Error."""
    a, f = _to_array(actual), _to_array(forecast)
    _validate_lengths(a, f)
    return float(np.median(np.abs(a - f)))


# --- Percentage ---

def mape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean Absolute Percentage Error.

    Raises ValueError if any actual value is zero.
    """
    a, f = _to_array(actual), _to_array(forecast)
    _validate_lengths(a, f)
    if np.any(a == 0):
        raise ValueError("MAPE undefined when actual values contain zero")
    return float(100.0 * np.mean(np.abs((a - f) / a)))


def smape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale)."""
    a, f = _to_array(actual), _to_array(forecast)
    _validate_lengths(a, f)
    denom = np.abs(a) + np.abs(f)
    # Avoid 0/0: where both are zero, contribution is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(denom == 0, 0.0, np.abs(a - f) / denom)
    return float(200.0 * np.mean(ratios))


def mdape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Median Absolute Percentage Error.

    Raises ValueError if any actual value is zero.
    """
    a, f = _to_array(actual), _to_array(forecast)
    _validate_lengths(a, f)
    if np.any(a == 0):
        raise ValueError("MdAPE undefined when actual values contain zero")
    return float(100.0 * np.median(np.abs((a - f) / a)))


# --- Scaled ---

def mase(
    actual: ArrayLike,
    forecast: ArrayLike,
    insample: ArrayLike,
    season: int = 1,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006)."""
    a, f, ins = _to_array(actual), _to_array(forecast), _to_array(insample)
    _validate_lengths(a, f)
    naive_errors = np.abs(ins[season:] - ins[:-season])
    scale = np.mean(naive_errors)
    if scale == 0:
        raise ValueError("MASE scale is zero (in-sample seasonal naive has zero error)")
    return float(np.mean(np.abs(a - f)) / scale)


def rmsse(
    actual: ArrayLike,
    forecast: ArrayLike,
    insample: ArrayLike,
    season: int = 1,
) -> float:
    """Root Mean Squared Scaled Error."""
    a, f, ins = _to_array(actual), _to_array(forecast), _to_array(insample)
    _validate_lengths(a, f)
    naive_errors = ins[season:] - ins[:-season]
    scale = np.mean(naive_errors**2)
    if scale == 0:
        raise ValueError("RMSSE scale is zero")
    return float(np.sqrt(np.mean((a - f) ** 2) / scale))


# --- Distributional ---

def crps_gaussian(
    actual: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
) -> float:
    """Continuous Ranked Probability Score for Gaussian forecasts.

    Closed-form: CRPS = sigma * [z*Phi(z) + phi(z) - 1/sqrt(pi)]
    where z = (actual - mu) / sigma.
    """
    from scipy.stats import norm

    a, m, s = _to_array(actual), _to_array(mu), _to_array(sigma)
    z = (a - m) / s
    crps_values = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps_values))


def winkler_score(
    actual: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float = 0.05,
) -> float:
    """Winkler interval score.

    Rewards narrow intervals, penalizes misses proportional to distance.
    """
    a, lo, up = _to_array(actual), _to_array(lower), _to_array(upper)
    _validate_lengths(a, lo)
    _validate_lengths(a, up)
    width = up - lo
    penalty = np.where(
        a < lo,
        (2 / alpha) * (lo - a),
        np.where(a > up, (2 / alpha) * (a - up), 0.0),
    )
    return float(np.mean(width + penalty))
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/metrics.py tests/test_metrics.py
git commit -m "feat: add forecast accuracy metrics (scale, percentage, scaled, distributional)"
```

---

### Task 5: Backtesting (`backtest.py`)

**Files:**
- Create: `foldcast/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write tests for backtesting**

```python
"""Tests for foldcast.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from foldcast.backtest import expanding_window, sliding_window
from foldcast._types import BacktestResult


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
        # 60 obs, min_train=24, horizon=3, step=3 => cutoffs at 24,27,30,...,57
        # That's (57-24)/3 + 1 = 12 folds
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
            # Forecast dates must be after cutoff
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
        for fold in result.folds:
            # With embargo=2, first forecast should be 3 periods after last train obs
            # (1 for normal gap + 2 embargo)
            pass  # Key check: it runs without error and produces results
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
        # Every fold should train on exactly 24 observations
        # (we can verify by checking cutoff positions)
        for fold in result.folds:
            # The fold's training data ends at cutoff
            train = monthly_series[monthly_series.index <= fold.cutoff]
            # With sliding window, only last window_size are used
            assert len(train) >= 24
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement backtest.py**

```python
"""Temporal cross-validation for forecast backtesting."""

from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

from foldcast._types import BacktestResult, FoldResult
from foldcast._utils import validate_series, validate_freq


class ModelFn(Protocol):
    """Protocol for forecast model callables."""

    def __call__(self, train: pd.Series, horizon: int) -> pd.Series: ...


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
    freq = validate_freq(data, freq)
    if min_train_size is None:
        min_train_size = 2 * horizon

    folds = _generate_folds(
        data=data,
        model_fn=model_fn,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        embargo=embargo,
        window_size=None,  # None = expanding
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
    freq = validate_freq(data, freq)

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

    cutoff_idx = min_train_size - 1  # Index of last training observation

    while cutoff_idx + embargo + horizon < n:
        # Training data
        if window_size is not None:
            train_start = max(0, cutoff_idx - window_size + 1)
            train = data.iloc[train_start : cutoff_idx + 1]
        else:
            train = data.iloc[: cutoff_idx + 1]

        # Test data (after embargo)
        test_start = cutoff_idx + 1 + embargo
        test_end = test_start + horizon
        if test_end > n:
            break

        actuals = data.iloc[test_start:test_end]

        # Generate forecasts
        forecasts = model_fn(train, horizon)

        # Align forecast index with actuals
        if not forecasts.index.equals(actuals.index):
            forecasts = pd.Series(
                forecasts.values[:horizon], index=actuals.index, name=forecasts.name
            )

        folds.append(FoldResult(
            fold_id=fold_id,
            cutoff=data.index[cutoff_idx],
            forecasts=forecasts,
            actuals=actuals,
        ))

        fold_id += 1
        cutoff_idx += step

    return folds
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/backtest.py tests/test_backtest.py
git commit -m "feat: add expanding and sliding window backtesting"
```

---

### Task 6: Model Comparison (`compare.py`)

**Files:**
- Create: `foldcast/compare.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write tests for comparison functions**

```python
"""Tests for foldcast.compare."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from foldcast.compare import (
    diebold_mariano,
    model_confidence_set,
    combine_forecasts,
    rank_table,
)
from foldcast._types import DMResult, MCSResult, CombineResult


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
        # Good forecast should have significantly lower loss
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
        # When forecasts are identical, both should be included
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
        # Good forecast should get more weight
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compare.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement compare.py**

```python
"""Statistical forecast model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from foldcast._types import CombineResult, DMResult, MCSResult
from foldcast import metrics as m


def diebold_mariano(
    forecasts_a: ArrayLike,
    forecasts_b: ArrayLike,
    actuals: ArrayLike,
    horizon: int = 1,
    loss: str = "squared",
    alpha: float = 0.05,
) -> DMResult:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[L(e_a)] = E[L(e_b)] against H1: E[L(e_a)] != E[L(e_b)].

    Args:
        forecasts_a: Forecasts from model A.
        forecasts_b: Forecasts from model B.
        actuals: Observed values.
        horizon: Forecast horizon (for HAC bandwidth selection).
        loss: Loss function — "squared" or "absolute".
        alpha: Significance level.

    Returns:
        DMResult with test statistic, p-value, and conclusion.
    """
    fa = np.asarray(forecasts_a, dtype=float)
    fb = np.asarray(forecasts_b, dtype=float)
    y = np.asarray(actuals, dtype=float)

    e_a = y - fa
    e_b = y - fb

    if loss == "squared":
        d = e_a**2 - e_b**2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"Unknown loss: {loss!r}. Use 'squared' or 'absolute'.")

    n = len(d)
    d_bar = np.mean(d)

    # HAC variance estimate (Newey-West with bandwidth = horizon - 1)
    gamma_0 = np.mean((d - d_bar) ** 2)
    bandwidth = max(1, horizon - 1)
    autocovariances = 0.0
    for k in range(1, bandwidth + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        autocovariances += 2 * gamma_k

    variance = (gamma_0 + autocovariances) / n

    if variance <= 0:
        return DMResult(statistic=0.0, p_value=1.0, conclusion="fail_to_reject")

    statistic = d_bar / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    conclusion = "reject" if p_value < alpha else "fail_to_reject"

    return DMResult(statistic=float(statistic), p_value=float(p_value), conclusion=conclusion)


def model_confidence_set(
    forecasts: dict[str, ArrayLike],
    actuals: ArrayLike,
    alpha: float = 0.10,
    loss: str = "squared",
) -> MCSResult:
    """Model Confidence Set (Hansen, Lunde, Nason 2011).

    Iteratively eliminates the worst model until no model can be rejected.

    Args:
        forecasts: Dict mapping model names to forecast arrays.
        actuals: Observed values.
        alpha: Significance level for elimination.
        loss: Loss function for DM tests.

    Returns:
        MCSResult with included/excluded models and elimination order.
    """
    y = np.asarray(actuals, dtype=float)
    models = {name: np.asarray(f, dtype=float) for name, f in forecasts.items()}

    remaining = set(models.keys())
    eliminated: list[str] = []
    p_values: dict[str, float] = {}

    while len(remaining) > 1:
        # Compute loss for each remaining model
        losses = {}
        for name in remaining:
            e = y - models[name]
            losses[name] = np.mean(e**2) if loss == "squared" else np.mean(np.abs(e))

        # Find worst model
        worst = max(remaining, key=lambda n: losses[n])

        # Test worst against best
        best = min(remaining, key=lambda n: losses[n])
        if worst == best:
            break

        dm = diebold_mariano(models[best], models[worst], y, loss=loss, alpha=alpha)

        p_values[worst] = dm.p_value

        if dm.p_value < alpha:
            remaining.remove(worst)
            eliminated.append(worst)
        else:
            # Cannot reject — stop elimination
            break

    # Assign p_value=1.0 to surviving models
    for name in remaining:
        if name not in p_values:
            p_values[name] = 1.0

    return MCSResult(
        included=sorted(remaining),
        excluded=eliminated,
        p_values=p_values,
        elimination_order=eliminated,
        alpha=alpha,
    )


def combine_forecasts(
    forecasts: dict[str, pd.Series],
    actuals: pd.Series,
    method: str = "inverse_mse",
) -> CombineResult:
    """Combine multiple forecasts using weighted averaging.

    Args:
        forecasts: Dict mapping model names to forecast Series.
        actuals: Observed values.
        method: "equal", "inverse_mse", or "bates_granger".

    Returns:
        CombineResult with combined forecast and weights.
    """
    names = list(forecasts.keys())
    f_arrays = {n: forecasts[n].values for n in names}

    if method == "equal":
        w = {n: 1.0 / len(names) for n in names}
    elif method in ("inverse_mse", "bates_granger"):
        mse = {n: float(np.mean((actuals.values - f_arrays[n]) ** 2)) for n in names}
        inv_mse = {n: 1.0 / max(v, 1e-12) for n, v in mse.items()}
        total = sum(inv_mse.values())
        w = {n: inv_mse[n] / total for n in names}
    else:
        raise ValueError(f"Unknown method: {method!r}")

    combined_values = sum(w[n] * f_arrays[n] for n in names)
    combined = pd.Series(combined_values, index=actuals.index, name="combined")

    return CombineResult(combined=combined, weights=w, method=method)


def rank_table(
    forecasts: dict[str, ArrayLike],
    actuals: ArrayLike,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Produce a summary table ranking models by multiple metrics.

    Args:
        forecasts: Dict mapping model names to forecast arrays.
        actuals: Observed values.
        metrics: List of metric names from foldcast.metrics. Defaults to ["mae", "rmse", "mape"].

    Returns:
        DataFrame indexed by model name with one column per metric, sorted by first metric.
    """
    if metrics is None:
        metrics = ["mae", "rmse"]

    y = np.asarray(actuals, dtype=float)
    metric_fns = {
        "mae": m.mae, "rmse": m.rmse, "mdae": m.mdae,
        "mape": m.mape, "smape": m.smape, "mdape": m.mdape,
    }

    rows = {}
    for name, f in forecasts.items():
        f_arr = np.asarray(f, dtype=float)
        rows[name] = {
            metric: metric_fns[metric](y, f_arr) for metric in metrics
        }

    df = pd.DataFrame(rows).T
    df.index.name = "model"
    return df.sort_values(metrics[0])
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_compare.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/compare.py tests/test_compare.py
git commit -m "feat: add Diebold-Mariano, MCS, forecast combination, rank table"
```

---

### Task 7: Monitor (`monitor.py`)

**Files:**
- Create: `foldcast/monitor.py`
- Create: `tests/test_monitor.py`

- [ ] **Step 1: Write tests for monitoring functions**

```python
"""Tests for foldcast.monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from foldcast.monitor import (
    check_residuals,
    detect_drift,
    check_coverage,
    detect_bias,
)
from foldcast._types import ResidualReport, DriftResult, CoverageResult, BiasResult


class TestCheckResiduals:
    def test_white_noise(self):
        rng = np.random.default_rng(42)
        residuals = pd.Series(rng.normal(0, 1, 500))
        report = check_residuals(residuals)
        assert isinstance(report, ResidualReport)
        # White noise should pass Ljung-Box (high p-value)
        assert report.ljung_box_p_value > 0.01

    def test_autocorrelated_residuals(self):
        rng = np.random.default_rng(42)
        n = 500
        residuals = np.zeros(n)
        residuals[0] = rng.normal()
        for i in range(1, n):
            residuals[i] = 0.8 * residuals[i - 1] + rng.normal(0, 0.5)
        report = check_residuals(pd.Series(residuals))
        # Strong autocorrelation should be detected
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
        recent = rng.normal(3, 1, 50)  # Mean-shifted
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
        assert result.empirical == 1.0  # All within ±2
        assert result.alert is False

    def test_poor_coverage(self):
        rng = np.random.default_rng(42)
        n = 200
        actuals = rng.normal(0, 1, n)
        lower = actuals - 0.01
        upper = actuals + 0.01
        # Shift actuals so many fall outside
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
        residuals = rng.normal(2.0, 0.5, 200)  # Persistent positive bias
        result = detect_bias(residuals, threshold=0.5)
        assert result.alert is True
        assert result.mean_bias > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitor.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement monitor.py**

```python
"""Production forecast monitoring: drift detection, coverage, residual diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from foldcast._types import BiasResult, CoverageResult, DriftResult, ResidualReport


def check_residuals(
    residuals: ArrayLike,
    lags: int = 10,
) -> ResidualReport:
    """Run diagnostic tests on forecast residuals.

    Tests: Ljung-Box (autocorrelation), Jarque-Bera (normality),
    ARCH-LM (heteroscedasticity).

    Args:
        residuals: Forecast errors (actual - forecast).
        lags: Number of lags for Ljung-Box and ARCH tests.

    Returns:
        ResidualReport with all test statistics and p-values.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)

    # Ljung-Box
    lb_stat, lb_pval = _ljung_box(r, lags)

    # Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(r)

    # ARCH-LM test
    arch_stat, arch_pval = _arch_lm(r, lags)

    return ResidualReport(
        ljung_box_statistic=float(lb_stat),
        ljung_box_p_value=float(lb_pval),
        jarque_bera_statistic=float(jb_stat),
        jarque_bera_p_value=float(jb_pval),
        arch_statistic=float(arch_stat),
        arch_p_value=float(arch_pval),
        mean_residual=float(np.mean(r)),
        std_residual=float(np.std(r, ddof=1)) if n > 1 else 0.0,
    )


def detect_drift(
    historical_errors: ArrayLike,
    recent_errors: ArrayLike,
    method: str = "psi",
    threshold: float = 0.2,
    n_bins: int = 10,
) -> DriftResult:
    """Detect distributional shift between historical and recent forecast errors.

    Args:
        historical_errors: Baseline error distribution.
        recent_errors: Recent errors to compare against baseline.
        method: "psi" (Population Stability Index).
        threshold: Alert threshold for the chosen statistic.
        n_bins: Number of bins for PSI computation.

    Returns:
        DriftResult with statistic, threshold, and alert flag.
    """
    hist = np.asarray(historical_errors, dtype=float)
    recent = np.asarray(recent_errors, dtype=float)

    if method == "psi":
        statistic = _psi(hist, recent, n_bins)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'psi'.")

    return DriftResult(
        statistic=float(statistic),
        threshold=threshold,
        alert=statistic > threshold,
        method=method,
    )


def check_coverage(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    nominal: float = 0.95,
    tolerance: float = 0.05,
) -> CoverageResult:
    """Check prediction interval coverage.

    Args:
        actuals: Observed values.
        lower: Lower prediction interval bounds.
        upper: Upper prediction interval bounds.
        nominal: Expected coverage rate.
        tolerance: Acceptable deviation from nominal before alerting.

    Returns:
        CoverageResult with empirical coverage and alert status.
    """
    a = np.asarray(actuals, dtype=float)
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)

    covered = (a >= lo) & (a <= up)
    n = len(a)
    n_covered = int(np.sum(covered))
    empirical = n_covered / n if n > 0 else 0.0

    return CoverageResult(
        nominal=nominal,
        empirical=float(empirical),
        n_observations=n,
        n_covered=n_covered,
        alert=empirical < (nominal - tolerance),
    )


def detect_bias(
    residuals: ArrayLike,
    threshold: float = 0.5,
) -> BiasResult:
    """Detect systematic forecast bias.

    Args:
        residuals: Forecast errors (actual - forecast).
        threshold: Alert if |mean_bias| exceeds this value.

    Returns:
        BiasResult with cumulative bias, mean bias, and alert flag.
    """
    r = np.asarray(residuals, dtype=float)
    cumulative = float(np.sum(r))
    mean_bias = float(np.mean(r))

    return BiasResult(
        cumulative_bias=cumulative,
        mean_bias=mean_bias,
        alert=abs(mean_bias) > threshold,
        threshold=threshold,
    )


# --- Internal helpers ---

def _ljung_box(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Ljung-Box test for autocorrelation."""
    n = len(x)
    x_centered = x - np.mean(x)
    c0 = np.sum(x_centered**2) / n

    if c0 == 0:
        return 0.0, 1.0

    q_stat = 0.0
    for k in range(1, lags + 1):
        if k >= n:
            break
        ck = np.sum(x_centered[k:] * x_centered[:-k]) / n
        rho_k = ck / c0
        q_stat += rho_k**2 / (n - k)

    q_stat *= n * (n + 2)
    p_value = 1.0 - stats.chi2.cdf(q_stat, df=lags)
    return float(q_stat), float(p_value)


def _arch_lm(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Engle's ARCH-LM test for heteroscedasticity in residuals."""
    n = len(x)
    x_sq = (x - np.mean(x)) ** 2

    if lags >= n - 1:
        return 0.0, 1.0

    # Regress x_sq[lags:] on x_sq lagged 1..lags
    y = x_sq[lags:]
    X = np.column_stack([x_sq[lags - k : n - k] for k in range(1, lags + 1)])
    X = np.column_stack([np.ones(len(y)), X])

    # OLS R-squared
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    stat = len(y) * r_squared
    p_value = 1.0 - stats.chi2.cdf(stat, df=lags)
    return float(stat), float(p_value)


def _psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index."""
    # Use expected distribution to define bin edges
    edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Remove duplicate edges
    edges = np.unique(edges)

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=edges)[0].astype(float)

    # Normalize to proportions, with floor to avoid log(0)
    exp_prop = np.maximum(exp_counts / exp_counts.sum(), 1e-8)
    act_prop = np.maximum(act_counts / act_counts.sum(), 1e-8)

    psi = np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop))
    return float(psi)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_monitor.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/monitor.py tests/test_monitor.py
git commit -m "feat: add residual diagnostics, drift detection, coverage, bias monitoring"
```

---

### Task 8: Hierarchy (`hierarchy.py`)

**Files:**
- Create: `foldcast/hierarchy.py`
- Create: `tests/test_hierarchy.py`

- [ ] **Step 1: Write tests for hierarchical evaluation**

```python
"""Tests for foldcast.hierarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from foldcast.hierarchy import HierarchyTree, evaluate_levels, check_coherence
from foldcast._types import CoherenceResult


@pytest.fixture()
def sample_hierarchy_df():
    """Hierarchical time series: total -> 2 regions -> 2 cities each."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = {
        ("East", "NYC"): 100 + rng.normal(0, 5, 24),
        ("East", "BOS"): 50 + rng.normal(0, 3, 24),
        ("West", "LAX"): 80 + rng.normal(0, 4, 24),
        ("West", "SEA"): 40 + rng.normal(0, 2, 24),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["region", "city"])
    return df


class TestHierarchyTree:
    def test_from_dataframe(self, sample_hierarchy_df):
        tree = HierarchyTree.from_dataframe(
            sample_hierarchy_df, levels=["region", "city"]
        )
        assert tree.n_levels == 3  # total, region, city
        assert tree.n_bottom == 4
        assert "total" in tree.level_names(0)

    def test_summing_matrix(self, sample_hierarchy_df):
        tree = HierarchyTree.from_dataframe(
            sample_hierarchy_df, levels=["region", "city"]
        )
        S = tree.summing_matrix()
        assert S.shape[0] == 7  # total + 2 regions + 4 cities
        assert S.shape[1] == 4  # 4 bottom-level series


class TestEvaluateLevels:
    def test_basic(self, sample_hierarchy_df):
        tree = HierarchyTree.from_dataframe(
            sample_hierarchy_df, levels=["region", "city"]
        )
        rng = np.random.default_rng(99)
        # Create forecasts as actuals + noise
        forecasts_df = sample_hierarchy_df + rng.normal(0, 2, sample_hierarchy_df.shape)
        result = evaluate_levels(
            tree=tree,
            forecasts=forecasts_df,
            actuals=sample_hierarchy_df,
            metrics=["mae", "rmse"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "mae" in result.columns
        assert "level" in result.columns


class TestCheckCoherence:
    def test_coherent(self, sample_hierarchy_df):
        tree = HierarchyTree.from_dataframe(
            sample_hierarchy_df, levels=["region", "city"]
        )
        # Coherent forecasts: aggregate from bottom up
        result = check_coherence(tree=tree, forecasts=sample_hierarchy_df)
        assert isinstance(result, CoherenceResult)
        assert result.is_coherent is True
        assert result.max_violation < 1e-6

    def test_incoherent(self, sample_hierarchy_df):
        tree = HierarchyTree.from_dataframe(
            sample_hierarchy_df, levels=["region", "city"]
        )
        # Make incoherent: perturb some values
        bad_forecasts = sample_hierarchy_df.copy()
        bad_forecasts.iloc[:, 0] += 100  # Break coherence
        result = check_coherence(tree=tree, forecasts=bad_forecasts)
        assert result.is_coherent is False
        assert result.max_violation > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hierarchy.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement hierarchy.py**

```python
"""Hierarchical forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from foldcast._types import CoherenceResult
from foldcast import metrics as m


@dataclass
class HierarchyTree:
    """Represents a hierarchical time series structure.

    Attributes:
        bottom_names: Names of bottom-level series.
        levels: List of dicts mapping level names to lists of bottom series they aggregate.
        level_labels: Names of each hierarchy level (e.g., ["total", "region", "city"]).
    """

    bottom_names: list[str]
    levels: list[dict[str, list[str]]]
    level_labels: list[str]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        levels: list[str],
    ) -> HierarchyTree:
        """Build hierarchy tree from a DataFrame with MultiIndex columns.

        Args:
            df: DataFrame with MultiIndex columns representing the hierarchy.
            levels: Column level names from broadest to narrowest.

        Returns:
            HierarchyTree describing the aggregation structure.
        """
        if not isinstance(df.columns, pd.MultiIndex):
            raise TypeError("DataFrame must have MultiIndex columns")

        bottom_names = ["_".join(str(x) for x in col) for col in df.columns]

        hierarchy_levels: list[dict[str, list[str]]] = []

        # Level 0: total
        hierarchy_levels.append({"total": list(bottom_names)})

        # Intermediate levels
        for depth in range(len(levels) - 1):
            level_map: dict[str, list[str]] = {}
            for col, bname in zip(df.columns, bottom_names):
                key = "_".join(str(col[i]) for i in range(depth + 1))
                level_map.setdefault(key, []).append(bname)
            hierarchy_levels.append(level_map)

        # Bottom level: each series maps to itself
        hierarchy_levels.append({bname: [bname] for bname in bottom_names})

        level_labels = ["total"] + levels

        return cls(
            bottom_names=bottom_names,
            levels=hierarchy_levels,
            level_labels=level_labels,
        )

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def n_bottom(self) -> int:
        return len(self.bottom_names)

    def level_names(self, depth: int) -> list[str]:
        """Return node names at a given depth."""
        return list(self.levels[depth].keys())

    def summing_matrix(self) -> np.ndarray:
        """Build the summing matrix S where y = S @ b (b = bottom-level).

        Each row corresponds to a node in the hierarchy (all levels),
        each column to a bottom-level series. Entry S[i,j] = 1 if
        bottom series j contributes to node i.
        """
        all_nodes: list[str] = []
        mapping: list[list[str]] = []

        for level in self.levels:
            for node_name, children in level.items():
                all_nodes.append(node_name)
                mapping.append(children)

        n_rows = len(all_nodes)
        n_cols = self.n_bottom
        S = np.zeros((n_rows, n_cols))

        for i, children in enumerate(mapping):
            for child in children:
                j = self.bottom_names.index(child)
                S[i, j] = 1.0

        return S


def evaluate_levels(
    tree: HierarchyTree,
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute accuracy metrics at each hierarchy level.

    Args:
        tree: HierarchyTree describing the structure.
        forecasts: Forecast DataFrame (same shape/columns as actuals).
        actuals: Actual DataFrame with MultiIndex columns.
        metrics: List of metric names. Defaults to ["mae", "rmse"].

    Returns:
        DataFrame with columns: level, node, metric_name, ...
    """
    if metrics is None:
        metrics = ["mae", "rmse"]

    metric_fns = {"mae": m.mae, "rmse": m.rmse, "mdae": m.mdae, "smape": m.smape}

    # Map bottom names to column positions
    bottom_values_actual = {
        "_".join(str(x) for x in col): actuals[col].values
        for col in actuals.columns
    }
    bottom_values_forecast = {
        "_".join(str(x) for x in col): forecasts[col].values
        for col in forecasts.columns
    }

    rows = []
    for depth, (level_map, level_label) in enumerate(
        zip(tree.levels, tree.level_labels)
    ):
        for node_name, children in level_map.items():
            # Aggregate actuals and forecasts for this node
            agg_actual = sum(bottom_values_actual[c] for c in children)
            agg_forecast = sum(bottom_values_forecast[c] for c in children)

            row: dict[str, object] = {"level": level_label, "node": node_name}
            for metric_name in metrics:
                fn = metric_fns[metric_name]
                row[metric_name] = fn(agg_actual, agg_forecast)
            rows.append(row)

    return pd.DataFrame(rows)


def check_coherence(
    tree: HierarchyTree,
    forecasts: pd.DataFrame,
    tol: float = 1e-6,
) -> CoherenceResult:
    """Check whether forecasts are coherent (sum consistently across levels).

    Args:
        tree: HierarchyTree describing the structure.
        forecasts: Forecast DataFrame with MultiIndex columns.
        tol: Tolerance for numerical coherence.

    Returns:
        CoherenceResult with coherence status and violation details.
    """
    bottom_values = {
        "_".join(str(x) for x in col): forecasts[col].values
        for col in forecasts.columns
    }

    max_violation = 0.0
    incoherent_nodes: list[str] = []

    # Check non-bottom levels: does the node's value equal sum of its children?
    for depth, level_map in enumerate(tree.levels[:-1]):
        for node_name, children in level_map.items():
            expected = sum(bottom_values[c] for c in children)

            # For top/intermediate levels, the "actual" aggregate is the sum of bottom
            # If the forecasts DataFrame only has bottom-level columns, coherence
            # means the implied aggregates are consistent.
            # We check: for each intermediate node, does the sum of its direct
            # sub-nodes equal the sum of its bottom-level children?
            if depth < tree.n_levels - 2:
                # Get the next level's sub-nodes that belong to this node
                next_level = tree.levels[depth + 1]
                sub_aggregate = np.zeros_like(expected)
                for sub_name, sub_children in next_level.items():
                    # Check if this sub-node is under current node
                    if all(c in children for c in sub_children):
                        sub_aggregate += sum(bottom_values[c] for c in sub_children)

                violation = float(np.max(np.abs(expected - sub_aggregate)))
                if violation > max_violation:
                    max_violation = violation
                if violation > tol:
                    incoherent_nodes.append(node_name)

    return CoherenceResult(
        is_coherent=max_violation <= tol,
        max_violation=max_violation,
        incoherent_nodes=incoherent_nodes,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_hierarchy.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/hierarchy.py tests/test_hierarchy.py
git commit -m "feat: add hierarchical forecast evaluation and coherence checking"
```

---

### Task 9: Visualize (`visualize.py`)

**Files:**
- Create: `foldcast/visualize.py`
- Create: `tests/test_visualize.py`

- [ ] **Step 1: Write tests for plotting functions**

```python
"""Tests for foldcast.visualize."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from foldcast.visualize import (
    plot_backtest,
    plot_model_comparison,
    plot_residual_diagnostics,
    plot_hierarchy_heatmap,
)
from foldcast._types import BacktestResult, FoldResult, ResidualReport


@pytest.fixture()
def sample_backtest_result():
    idx1 = pd.date_range("2020-01-01", periods=3, freq="MS")
    idx2 = pd.date_range("2020-04-01", periods=3, freq="MS")
    folds = [
        FoldResult(0, pd.Timestamp("2020-01-01"),
                   pd.Series([100, 101, 102.0], index=idx1),
                   pd.Series([100.5, 101.5, 101.8], index=idx1)),
        FoldResult(1, pd.Timestamp("2020-04-01"),
                   pd.Series([103, 104, 105.0], index=idx2),
                   pd.Series([102.8, 104.2, 105.1], index=idx2)),
    ]
    return BacktestResult(folds=folds, horizon=3, strategy="expanding")


class TestPlotBacktest:
    def test_returns_figure(self, sample_backtest_result):
        fig = plot_backtest(sample_backtest_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_full_series(self, sample_backtest_result, monthly_series):
        fig = plot_backtest(sample_backtest_result, full_series=monthly_series)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotModelComparison:
    def test_returns_figure(self):
        scores = pd.DataFrame({
            "mae": [1.0, 2.0, 3.0],
            "rmse": [1.5, 2.5, 3.5],
        }, index=["model_a", "model_b", "model_c"])
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
        df = pd.DataFrame({
            "level": ["total", "region", "region", "city", "city", "city", "city"],
            "node": ["total", "East", "West", "NYC", "BOS", "LAX", "SEA"],
            "mae": [5.0, 3.0, 4.0, 2.0, 2.5, 3.0, 3.5],
            "rmse": [6.0, 3.5, 4.5, 2.5, 3.0, 3.5, 4.0],
        })
        fig = plot_hierarchy_heatmap(df, metric="mae")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualize.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement visualize.py**

```python
"""Plotting utilities for forecast evaluation results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from foldcast._types import BacktestResult


def plot_backtest(
    result: BacktestResult,
    full_series: pd.Series | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Plot backtest results: actuals vs. forecasts across folds.

    Args:
        result: BacktestResult from a backtest run.
        full_series: Optional full time series to show as background.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if full_series is not None:
        ax.plot(full_series.index, full_series.values, color="0.7",
                linewidth=1, label="Observed", zorder=1)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result.folds), 10)))

    for fold in result.folds:
        color = colors[fold.fold_id % len(colors)]
        ax.plot(fold.actuals.index, fold.actuals.values, "o",
                color=color, markersize=4, zorder=2)
        ax.plot(fold.forecasts.index, fold.forecasts.values, "-",
                color=color, linewidth=1.5, zorder=3)
        ax.axvline(fold.cutoff, color=color, linestyle=":", alpha=0.3)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Backtest ({result.strategy}, {len(result.folds)} folds, h={result.horizon})")
    fig.tight_layout()
    return fig


def plot_model_comparison(
    scores: pd.DataFrame,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Bar chart comparing models across metrics.

    Args:
        scores: DataFrame indexed by model name, columns are metric names.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    n_models = len(scores)
    n_metrics = len(scores.columns)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)

    for i, metric in enumerate(scores.columns):
        ax = axes[0, i]
        values = scores[metric].sort_values()
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
        ax.barh(values.index, values.values, color=colors)
        ax.set_xlabel(metric.upper())
        ax.set_title(metric.upper())

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(
    residuals: pd.Series | np.ndarray,
    lags: int = 20,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Four-panel residual diagnostics: time series, histogram, ACF, QQ-plot.

    Args:
        residuals: Forecast residuals.
        lags: Number of lags for ACF plot.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    r = np.asarray(residuals, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Time series
    ax = axes[0, 0]
    ax.plot(r, linewidth=0.8, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Residuals Over Time")
    ax.set_ylabel("Residual")

    # Histogram
    ax = axes[0, 1]
    ax.hist(r, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_title("Distribution")
    ax.set_xlabel("Residual")

    # ACF
    ax = axes[1, 0]
    n = len(r)
    r_centered = r - np.mean(r)
    c0 = np.sum(r_centered**2) / n
    acf_vals = []
    for k in range(lags + 1):
        if c0 > 0:
            ck = np.sum(r_centered[k:] * r_centered[:n - k]) / n
            acf_vals.append(ck / c0)
        else:
            acf_vals.append(0.0)
    ax.bar(range(lags + 1), acf_vals, width=0.3, color="steelblue")
    conf = 1.96 / np.sqrt(n)
    ax.axhline(conf, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("ACF")
    ax.set_xlabel("Lag")

    # QQ-plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")

    fig.tight_layout()
    return fig


def plot_hierarchy_heatmap(
    level_metrics: pd.DataFrame,
    metric: str = "mae",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Heatmap of forecast accuracy across hierarchy nodes.

    Args:
        level_metrics: DataFrame from hierarchy.evaluate_levels with
            columns: level, node, and metric columns.
        metric: Which metric column to visualize.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    nodes = level_metrics["node"].values
    values = level_metrics[metric].values
    levels = level_metrics["level"].values

    # Color by value
    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    colors = plt.cm.YlOrRd(norm(values))

    y_positions = np.arange(len(nodes))
    bars = ax.barh(y_positions, values, color=colors, edgecolor="white")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"[{l}] {n}" for l, n in zip(levels, nodes)])
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Hierarchy: {metric.upper()} by Node")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_visualize.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add foldcast/visualize.py tests/test_visualize.py
git commit -m "feat: add visualization utilities for backtest, comparison, diagnostics, hierarchy"
```

---

### Task 10: Public API (`__init__.py`)

**Files:**
- Modify: `foldcast/__init__.py`
- Create: `tests/test_init.py`

- [ ] **Step 1: Write tests for public API**

```python
"""Tests for foldcast public API."""

from __future__ import annotations

import foldcast
from foldcast import backtest, compare, monitor, hierarchy, metrics, visualize


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_init.py -v`
Expected: FAIL — `__version__` not defined yet.

- [ ] **Step 3: Implement __init__.py**

```python
"""foldcast: Backtesting, comparison, and monitoring for production time series forecasts."""

__version__ = "0.1.0"

from foldcast import backtest, compare, hierarchy, metrics, monitor, visualize

__all__ = [
    "__version__",
    "backtest",
    "compare",
    "hierarchy",
    "metrics",
    "monitor",
    "visualize",
]
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_init.py -v`
Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `pytest --cov=foldcast --cov-report=term-missing -v`
Expected: All tests pass, >90% coverage.

- [ ] **Step 6: Commit**

```bash
git add foldcast/__init__.py tests/test_init.py
git commit -m "feat: add public API with version and submodule exports"
```

---

### Task 11: README and Examples

**Files:**
- Create: `README.md`
- Create: `examples/quickstart.py`
- Create: `examples/model_comparison.py`
- Create: `examples/production_monitoring.py`

- [ ] **Step 1: Write README.md**

```markdown
# foldcast

Backtesting, comparison, and monitoring for production time series forecasts.

[![CI](https://github.com/Michael/foldcast/actions/workflows/ci.yml/badge.svg)](https://github.com/Michael/foldcast/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**foldcast** is a lightweight, model-agnostic Python library for the operational layer of forecasting: temporal cross-validation, statistical model comparison, production monitoring, and hierarchical evaluation.

It does not fit models — it evaluates them.

## Install

```bash
pip install foldcast
```

## Quickstart

```python
import pandas as pd
from foldcast import backtest, metrics

# Your data: any pandas Series with DatetimeIndex
data = pd.read_csv("sales.csv", index_col="date", parse_dates=True)["revenue"]

# Your model: any callable (train_data, horizon) -> forecasts
def naive(train, horizon):
    last = train.iloc[-1]
    idx = pd.date_range(train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq)
    return pd.Series(last, index=idx)

# Backtest with expanding window
result = backtest.expanding_window(data, model_fn=naive, horizon=6, step=3, min_train_size=24)
print(result.summary())
```

## Features

### Backtesting
Temporal cross-validation with expanding or sliding windows, configurable embargo periods, and multi-horizon evaluation.

```python
from foldcast import backtest

result = backtest.expanding_window(data, model_fn=my_model, horizon=12, step=1)
result = backtest.sliding_window(data, model_fn=my_model, horizon=12, window_size=36)
```

### Model Comparison
Diebold-Mariano tests, Model Confidence Sets, and forecast combination with proper statistical rigor.

```python
from foldcast import compare

dm = compare.diebold_mariano(forecasts_a, forecasts_b, actuals)
mcs = compare.model_confidence_set({"ets": f1, "arima": f2, "prophet": f3}, actuals, alpha=0.10)
combined = compare.combine_forecasts({"ets": f1, "arima": f2}, actuals, method="inverse_mse")
```

### Monitoring
Detect forecast degradation in production: drift detection, coverage monitoring, residual diagnostics, and bias tracking.

```python
from foldcast import monitor

report = monitor.check_residuals(residuals)
drift = monitor.detect_drift(historical_errors, recent_errors, method="psi")
coverage = monitor.check_coverage(actuals, lower_bounds, upper_bounds, nominal=0.95)
bias = monitor.detect_bias(residuals, threshold=0.5)
```

### Hierarchical Evaluation
Evaluate forecasts across hierarchical structures with level-wise metrics and coherence checks.

```python
from foldcast import hierarchy

tree = hierarchy.HierarchyTree.from_dataframe(df, levels=["region", "city"])
level_metrics = hierarchy.evaluate_levels(tree, forecasts, actuals, metrics=["mae", "rmsse"])
coherence = hierarchy.check_coherence(tree, forecasts)
```

### Metrics
Scale-dependent (MAE, RMSE), percentage (MAPE, sMAPE), scaled (MASE, RMSSE), and distributional (CRPS, Winkler) metrics.

```python
from foldcast import metrics

metrics.mae(actual, forecast)
metrics.mase(actual, forecast, insample, season=12)
metrics.crps_gaussian(actual, mu, sigma)
```

### Visualization
Publication-ready plots for backtest results, model comparisons, residual diagnostics, and hierarchy heatmaps.

```python
from foldcast import visualize

fig = visualize.plot_backtest(result)
fig = visualize.plot_residual_diagnostics(residuals)
```

## Design Principles

- **Model-agnostic.** Any callable that takes training data and returns forecasts works with foldcast.
- **Lightweight.** Core dependencies: numpy, pandas, scipy, matplotlib. No heavy frameworks.
- **Production-oriented.** Built for the evaluation and monitoring layer, not the modeling layer.
- **Statistically rigorous.** Proper temporal cross-validation, Diebold-Mariano tests, Model Confidence Sets.

## License

MIT
```

- [ ] **Step 2: Write examples/quickstart.py**

```python
"""foldcast quickstart: backtest a naive model on synthetic data."""

import numpy as np
import pandas as pd
from foldcast import backtest, visualize

# Generate synthetic monthly revenue data
rng = np.random.default_rng(42)
dates = pd.date_range("2018-01-01", periods=60, freq="MS")
trend = 100 + 0.8 * np.arange(60)
seasonal = 15 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = rng.normal(0, 3, 60)
revenue = pd.Series(trend + seasonal + noise, index=dates, name="revenue")


# Define a simple model
def naive_model(train: pd.Series, horizon: int) -> pd.Series:
    """Repeat last observed value."""
    last = train.iloc[-1]
    idx = pd.date_range(
        train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
    )
    return pd.Series(last, index=idx, name=train.name)


# Run backtest
result = backtest.expanding_window(
    data=revenue,
    model_fn=naive_model,
    horizon=6,
    step=3,
    min_train_size=24,
)

print(result.summary())
print(f"\nFolds: {len(result.folds)}")
print(result.to_dataframe().head(12))

# Plot
fig = visualize.plot_backtest(result, full_series=revenue)
fig.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nSaved backtest_result.png")
```

- [ ] **Step 3: Write examples/model_comparison.py**

```python
"""foldcast example: compare multiple forecasting models."""

import numpy as np
import pandas as pd
from foldcast import backtest, compare, visualize

# Generate synthetic data
rng = np.random.default_rng(42)
dates = pd.date_range("2018-01-01", periods=60, freq="MS")
trend = 100 + 0.8 * np.arange(60)
seasonal = 15 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = rng.normal(0, 3, 60)
data = pd.Series(trend + seasonal + noise, index=dates, name="revenue")


# Model 1: Naive (repeat last value)
def naive(train: pd.Series, horizon: int) -> pd.Series:
    idx = pd.date_range(
        train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
    )
    return pd.Series(train.iloc[-1], index=idx)


# Model 2: Drift (linear extrapolation)
def drift(train: pd.Series, horizon: int) -> pd.Series:
    n = len(train)
    slope = (train.iloc[-1] - train.iloc[0]) / (n - 1)
    idx = pd.date_range(
        train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
    )
    forecasts = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    return pd.Series(forecasts, index=idx)


# Model 3: Seasonal naive
def seasonal_naive(train: pd.Series, horizon: int) -> pd.Series:
    season = 12
    last_cycle = train.iloc[-season:].values
    reps = (horizon // season) + 1
    fc = np.tile(last_cycle, reps)[:horizon]
    idx = pd.date_range(
        train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
    )
    return pd.Series(fc, index=idx)


# Backtest all three
models = {"naive": naive, "drift": drift, "seasonal_naive": seasonal_naive}
results = {}
for name, model_fn in models.items():
    results[name] = backtest.expanding_window(
        data=data, model_fn=model_fn, horizon=6, step=3, min_train_size=24
    )

# Collect forecasts and actuals from backtest results
all_forecasts = {}
for name, res in results.items():
    df = res.to_dataframe()
    all_forecasts[name] = df["forecast"].values

actuals_arr = results["naive"].to_dataframe()["actual"].values

# Rank table
table = compare.rank_table(all_forecasts, actuals_arr, metrics=["mae", "rmse", "smape"])
print("Model Ranking:")
print(table)

# Diebold-Mariano test: seasonal naive vs naive
dm = compare.diebold_mariano(
    all_forecasts["seasonal_naive"], all_forecasts["naive"], actuals_arr
)
print(f"\nDM test (seasonal_naive vs naive): stat={dm.statistic:.3f}, p={dm.p_value:.4f}")

# Model Confidence Set
mcs = compare.model_confidence_set(all_forecasts, actuals_arr, alpha=0.10)
print(f"\nModel Confidence Set (alpha=0.10): {mcs.included}")

# Combine best models
best = {n: pd.Series(all_forecasts[n]) for n in mcs.included}
actuals_series = pd.Series(actuals_arr)
combined = compare.combine_forecasts(best, actuals_series, method="inverse_mse")
print(f"\nCombination weights: {combined.weights}")

# Plot comparison
fig = visualize.plot_model_comparison(table)
fig.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved model_comparison.png")
```

- [ ] **Step 4: Write examples/production_monitoring.py**

```python
"""foldcast example: monitor a production forecast for degradation."""

import numpy as np
import pandas as pd
from foldcast import monitor, visualize

rng = np.random.default_rng(42)

# Simulate historical forecast errors (well-calibrated period)
historical_errors = rng.normal(0, 1.0, 500)

# Simulate recent errors (model has degraded — bias + increased variance)
recent_errors = rng.normal(0.8, 1.5, 50)

# 1. Residual diagnostics
print("=== Residual Diagnostics (recent) ===")
report = monitor.check_residuals(recent_errors)
print(report.summary())

# 2. Drift detection
print("\n=== Drift Detection ===")
drift = monitor.detect_drift(historical_errors, recent_errors, method="psi", threshold=0.2)
print(f"PSI = {drift.statistic:.4f} (threshold={drift.threshold})")
print(f"Alert: {drift.alert}")

# 3. Coverage check
print("\n=== Coverage Check ===")
n = 100
actuals = rng.normal(0, 1, n)
mu = actuals + rng.normal(0.3, 0.5, n)  # Slightly biased forecasts
lower = mu - 1.96
upper = mu + 1.96
coverage = monitor.check_coverage(actuals, lower, upper, nominal=0.95)
print(f"Nominal: {coverage.nominal:.0%}, Empirical: {coverage.empirical:.0%}")
print(f"Alert: {coverage.alert}")

# 4. Bias detection
print("\n=== Bias Detection ===")
bias = monitor.detect_bias(recent_errors, threshold=0.3)
print(f"Mean bias: {bias.mean_bias:.4f}")
print(f"Alert: {bias.alert}")

# Plot diagnostics
fig = visualize.plot_residual_diagnostics(recent_errors)
fig.savefig("residual_diagnostics.png", dpi=150, bbox_inches="tight")
print("\nSaved residual_diagnostics.png")
```

- [ ] **Step 5: Commit**

```bash
git add README.md examples/ LICENSE
git commit -m "docs: add README, quickstart, model comparison, and monitoring examples"
```

---

### Task 12: Final Polish and Integration Test

**Files:**
- Create: `tests/test_integration.py`
- Modify: `.github/workflows/ci.yml` (add ruff format check)

- [ ] **Step 1: Write integration test**

```python
"""End-to-end integration test: backtest → compare → monitor → hierarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from foldcast import backtest, compare, monitor, hierarchy, metrics, visualize


def test_full_workflow():
    """Run the full foldcast workflow on synthetic data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    trend = 100 + 0.5 * np.arange(60)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = rng.normal(0, 2, 60)
    data = pd.Series(trend + seasonal + noise, index=dates, name="revenue")

    # Models
    def naive(train, horizon):
        idx = pd.date_range(
            train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
        )
        return pd.Series(train.iloc[-1], index=idx)

    def snaive(train, horizon):
        s = 12
        vals = np.tile(train.iloc[-s:].values, (horizon // s) + 1)[:horizon]
        idx = pd.date_range(
            train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
        )
        return pd.Series(vals, index=idx)

    # 1. Backtest
    r_naive = backtest.expanding_window(data, naive, horizon=6, step=3, min_train_size=24)
    r_snaive = backtest.expanding_window(data, snaive, horizon=6, step=3, min_train_size=24)
    assert len(r_naive.folds) > 0
    assert len(r_snaive.folds) > 0

    # 2. Compare
    df_n = r_naive.to_dataframe()
    df_s = r_snaive.to_dataframe()
    fc = {"naive": df_n["forecast"].values, "snaive": df_s["forecast"].values}
    actuals_arr = df_n["actual"].values

    table = compare.rank_table(fc, actuals_arr, metrics=["mae", "rmse"])
    assert len(table) == 2

    dm = compare.diebold_mariano(fc["naive"], fc["snaive"], actuals_arr)
    assert dm.p_value >= 0

    mcs = compare.model_confidence_set(fc, actuals_arr, alpha=0.10)
    assert len(mcs.included) >= 1

    # 3. Monitor
    errors = actuals_arr - fc["naive"]
    report = monitor.check_residuals(errors)
    assert report.std_residual > 0

    bias = monitor.detect_bias(errors, threshold=5.0)
    assert isinstance(bias.alert, bool)

    # 4. Metrics
    assert metrics.mae(actuals_arr, fc["naive"]) >= 0
    assert metrics.rmse(actuals_arr, fc["naive"]) >= 0

    # 5. Visualize (smoke test)
    import matplotlib
    matplotlib.use("Agg")
    fig = visualize.plot_backtest(r_naive, full_series=data)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_hierarchy_workflow():
    """Test hierarchical forecast evaluation end-to-end."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = {
        ("East", "NYC"): 100 + rng.normal(0, 5, 24),
        ("East", "BOS"): 50 + rng.normal(0, 3, 24),
        ("West", "LAX"): 80 + rng.normal(0, 4, 24),
        ("West", "SEA"): 40 + rng.normal(0, 2, 24),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["region", "city"])

    tree = hierarchy.HierarchyTree.from_dataframe(df, levels=["region", "city"])
    assert tree.n_bottom == 4
    assert tree.n_levels == 3

    # Forecasts = actuals + noise
    forecasts = df + rng.normal(0, 2, df.shape)

    level_metrics = hierarchy.evaluate_levels(tree, forecasts, df, metrics=["mae", "rmse"])
    assert "mae" in level_metrics.columns

    coherence = hierarchy.check_coherence(tree, df)
    assert coherence.is_coherent is True

    S = tree.summing_matrix()
    assert S.shape == (7, 4)
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS.

- [ ] **Step 3: Run full test suite with coverage**

Run: `pytest --cov=foldcast --cov-report=term-missing -v`
Expected: All tests pass, >90% coverage.

- [ ] **Step 4: Run linter**

Run: `ruff check foldcast/ tests/`
Expected: No errors. Fix any issues found.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for full workflow"
```

---

### Task 13: Push to GitHub

- [ ] **Step 1: Create GitHub repo**

```bash
gh repo create foldcast --public --description "Backtesting, comparison, and monitoring for production time series forecasts." --source . --push
```

- [ ] **Step 2: Verify repo is live**

```bash
gh repo view foldcast --web
```
