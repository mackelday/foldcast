# foldcast — Design Spec

## Purpose

A lightweight Python library for backtesting, comparing, and monitoring production time series forecasts. Model-agnostic. Focused on the operational layer that sits between forecasting models and business decisions.

## Problem

Python has excellent forecasting model libraries (statsforecast, prophet, statsmodels) but no standalone, well-designed library for the evaluation and operations layer: temporal cross-validation, statistical model comparison, production drift detection, and hierarchical forecast evaluation. Teams end up writing this infrastructure ad hoc, poorly, and repeatedly.

## Target Users

Data scientists and ML engineers who operate forecasting systems in production — especially in finance, revenue, demand planning, and supply chain contexts where forecast reliability directly impacts decisions.

## Core Modules

### 1. `backtest` — Temporal Cross-Validation

Run forecasting models through rigorous temporal evaluation.

- **Expanding window**: Train on all data up to cutoff, forecast horizon ahead, slide cutoff forward
- **Sliding window**: Fixed-size training window slides forward
- **Embargo period**: Configurable gap between train and test to prevent leakage
- **Multi-horizon**: Evaluate at h=1, h=2, ..., h=H simultaneously
- **Parallel execution**: Optional joblib parallelism across folds

**Key API:**
```python
from foldcast import backtest

results = backtest.expanding_window(
    data=df,
    model_fn=my_model,       # callable: train_data -> forecasts
    horizon=12,
    step=1,
    embargo=0,
    freq="MS",
)
# Returns: BacktestResult with forecasts, actuals, fold metadata
```

The `model_fn` interface: any callable that accepts a pandas Series/DataFrame of training data and returns a pandas Series/DataFrame of forecasts for the next `horizon` periods. This keeps foldcast model-agnostic.

### 2. `compare` — Statistical Model Comparison

Compare forecast methods with proper statistical rigor.

- **Diebold-Mariano test**: Pairwise comparison of predictive accuracy with HAC standard errors
- **Model Confidence Set (MCS)**: Hansen et al. (2011) — identify the set of models that contains the best with a given confidence level
- **Forecast combination**: Inverse-MSE weighting, Bates-Granger, equal weights
- **Rank table**: Summary table of models ranked by multiple metrics with significance indicators

**Key API:**
```python
from foldcast import compare

dm = compare.diebold_mariano(forecasts_a, forecasts_b, actuals, horizon=1)
# Returns: DMResult with statistic, p_value, conclusion

mcs = compare.model_confidence_set(
    forecasts={"prophet": f1, "ets": f2, "arima": f3},
    actuals=actuals,
    alpha=0.10,
)
# Returns: MCSResult with included models, p-values, elimination order

combined = compare.combine_forecasts(
    forecasts={"prophet": f1, "ets": f2},
    actuals=actuals,
    method="inverse_mse",
)
# Returns: combined forecast Series + weights
```

### 3. `monitor` — Production Forecast Monitoring

Detect when deployed forecasts degrade.

- **Forecast drift detection**: PSI (Population Stability Index) and ADWIN for detecting distributional shift in forecast errors
- **Coverage monitoring**: Track prediction interval coverage rates, alert when miscalibrated
- **Residual diagnostics**: Ljung-Box test for autocorrelation, Jarque-Bera for normality, ARCH test for heteroscedasticity
- **Bias tracking**: Cumulative and rolling bias detection with CUSUM-style alerts
- **Alert thresholds**: Configurable thresholds that return structured alert objects

**Key API:**
```python
from foldcast import monitor

report = monitor.check_residuals(residuals)
# Returns: ResidualReport with autocorrelation, normality, heteroscedasticity results

drift = monitor.detect_drift(
    historical_errors=train_residuals,
    recent_errors=prod_residuals,
    method="psi",
    threshold=0.2,
)
# Returns: DriftResult with statistic, threshold, alert flag

coverage = monitor.check_coverage(
    actuals=actuals,
    lower=lower_bounds,
    upper=upper_bounds,
    nominal=0.95,
)
# Returns: CoverageResult with empirical rate, pass/fail, by-period breakdown
```

### 4. `hierarchy` — Hierarchical Forecast Evaluation

Evaluate forecasts across hierarchical structures (e.g., total → region → city → listing).

- **Level-wise metrics**: Compute accuracy metrics at each level of the hierarchy
- **Coherence check**: Verify that forecasts sum consistently across levels
- **Cross-level comparison**: Identify which levels are well-forecast vs. problematic
- **Reconciliation evaluation**: Compare pre- vs. post-reconciliation accuracy

**Key API:**
```python
from foldcast import hierarchy

tree = hierarchy.HierarchyTree.from_dataframe(df, levels=["region", "city"])

level_metrics = hierarchy.evaluate_levels(
    tree=tree,
    forecasts=forecasts_df,
    actuals=actuals_df,
    metrics=["mase", "rmsse"],
)
# Returns: DataFrame with metric per level per series

coherence = hierarchy.check_coherence(tree=tree, forecasts=forecasts_df)
# Returns: CoherenceResult with max_violation, incoherent_nodes
```

### 5. `metrics` — Forecast Accuracy Metrics

Standard and robust metrics used across all modules.

- **Scale-dependent**: MAE, RMSE, MdAE
- **Percentage**: MAPE, sMAPE, MdAPE
- **Scaled**: MASE, RMSSE (require in-sample data for scaling)
- **Distributional**: CRPS, log score, Winkler score (for probabilistic forecasts)
- All metrics accept arrays and return float. Vectorized for performance.

### 6. `visualize` — Plotting Utilities

Matplotlib-based plots, optional Plotly support.

- **Backtest plot**: Actuals vs. forecasts across folds with fan chart for intervals
- **Model comparison plot**: Bar chart of metrics with significance groupings
- **Monitoring dashboard**: Residual time series, ACF, coverage over time, drift indicator
- **Hierarchy heatmap**: Metric values across hierarchy levels

All plotting functions return matplotlib Figure objects for customization. Optional `plotly=True` kwarg for interactive versions.

## Data Contracts

- **Input**: pandas Series (univariate) or DataFrame (multivariate/hierarchical). DatetimeIndex required.
- **Forecasts**: Same structure as input. Must have matching index.
- **model_fn**: `Callable[[pd.Series], pd.Series]` or `Callable[[pd.DataFrame], pd.DataFrame]`
- **Result objects**: Frozen dataclasses with `.to_dataframe()`, `.summary()`, `.__repr__()` methods.

## Non-Goals

- **Not a forecasting library**: foldcast does not fit models. It evaluates them.
- **Not a pipeline orchestrator**: No Airflow/Prefect integration. It's a library, not a framework.
- **Not a dashboard**: Plotting utilities are included, but this is not a Streamlit app.
- **No database connectors**: Data comes in as pandas objects.

## Technical Decisions

- **Python 3.10+** (for `match` statements, `ParamSpec`, modern type unions)
- **Dependencies**: numpy, pandas, scipy, matplotlib. Optional: plotly, joblib.
- **Packaging**: pyproject.toml with hatchling backend
- **Testing**: pytest with >90% coverage
- **CI**: GitHub Actions — tests, linting (ruff), type checking (mypy)
- **Docs**: README with quickstart + examples/ directory with annotated scripts

## Package Structure

```
foldcast/
├── __init__.py          # Public API re-exports
├── backtest.py          # Temporal cross-validation
├── compare.py           # Statistical model comparison
├── monitor.py           # Production monitoring
├── hierarchy.py         # Hierarchical evaluation
├── metrics.py           # Accuracy metrics
├── visualize.py         # Plotting utilities
├── _types.py            # Result dataclasses, type aliases
└── _utils.py            # Shared internals (validation, indexing)
```

## Success Criteria

1. `pip install foldcast` works
2. A user can backtest a forecasting model in <10 lines of code
3. All public APIs have type hints and docstrings
4. >90% test coverage
5. README lets someone understand and use foldcast in under 60 seconds
