# foldcast

<p align="center">
  <img src="assets/banner.png" alt="foldcast banner" width="100%">
</p>

<p align="center">
  <strong>Backtesting, comparison, and monitoring for production time series forecasts.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

---

## Why foldcast?

Every forecasting team eventually builds the same internal tooling: temporal cross-validation loops, model comparison scripts, production monitoring checks. This infrastructure is critical — a forecast that looks good in a notebook but fails silently in production can cost millions in misallocated inventory, missed revenue targets, or bad financial plans.

Yet Python's forecasting ecosystem has a gap. There are excellent libraries for *fitting* models (statsforecast, prophet, statsmodels, NeuralForecast) but no standalone, well-designed library for the *operational layer* — the backtesting, evaluation, and monitoring that determines whether those models actually work.

**foldcast fills that gap.** It is a lightweight, model-agnostic toolkit for the part of forecasting that matters most in production: knowing whether your forecasts are reliable, which model to deploy, and when a deployed model starts to degrade.

### Who is this for?

- **Forecasting teams in finance, revenue, and FP&A** who need rigorous backtesting with proper temporal separation — not sklearn's `cross_val_score` leaking future data into training folds
- **ML engineers operating forecast pipelines** who need automated drift detection and coverage monitoring instead of manually eyeballing residual plots
- **Data scientists comparing models** who want Diebold-Mariano tests and Model Confidence Sets instead of "this one has a lower RMSE so it must be better"
- **Anyone building hierarchical forecasts** (total → region → store → SKU) who needs to evaluate accuracy at every level and verify coherence

### The problem it solves

| Without foldcast | With foldcast |
|---|---|
| Ad-hoc backtesting scripts copied between projects | `backtest.expanding_window()` — one line, proper temporal CV |
| "Model A has lower RMSE" (is the difference significant?) | `compare.diebold_mariano()` — statistically rigorous comparison |
| Model degrades in production, nobody notices for weeks | `monitor.detect_drift()` — automated distributional shift detection |
| Forecasts don't sum across hierarchy levels, finance is confused | `hierarchy.check_coherence()` — catch incoherence before it ships |

## Install

```bash
pip install foldcast
```

## Quickstart

```python
import pandas as pd
from foldcast import backtest

# Your data: any pandas Series with DatetimeIndex
data = pd.read_csv("sales.csv", index_col="date", parse_dates=True)["revenue"]

# Your model: any callable (train_data, horizon) -> forecasts
def naive(train, horizon):
    last = train.iloc[-1]
    idx = pd.date_range(
        train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
    )
    return pd.Series(last, index=idx)

# Backtest with expanding window
result = backtest.expanding_window(data, model_fn=naive, horizon=6, step=3, min_train_size=24)
print(result.summary())
```

## Features

### Backtesting

Temporal cross-validation with expanding or sliding windows, configurable embargo periods, and multi-horizon evaluation. Embargo periods prevent data leakage in financial contexts where values are revised or lagged.

```python
from foldcast import backtest

result = backtest.expanding_window(data, model_fn=my_model, horizon=12, step=1)
result = backtest.sliding_window(data, model_fn=my_model, horizon=12, window_size=36)
```

### Model Comparison

Diebold-Mariano tests with HAC standard errors, Model Confidence Sets (Hansen et al., 2011), and forecast combination via inverse-MSE or equal weighting. Go beyond "lowest RMSE wins" to statistically defensible model selection.

```python
from foldcast import compare

dm = compare.diebold_mariano(forecasts_a, forecasts_b, actuals)
mcs = compare.model_confidence_set(
    {"ets": f1, "arima": f2, "prophet": f3}, actuals, alpha=0.10
)
combined = compare.combine_forecasts({"ets": f1, "arima": f2}, actuals, method="inverse_mse")
```

### Monitoring

Detect forecast degradation in production before it impacts decisions. PSI-based drift detection, prediction interval coverage monitoring, Ljung-Box / Jarque-Bera / ARCH residual diagnostics, and CUSUM-style bias tracking.

```python
from foldcast import monitor

report = monitor.check_residuals(residuals)
drift = monitor.detect_drift(historical_errors, recent_errors, method="psi")
coverage = monitor.check_coverage(actuals, lower_bounds, upper_bounds, nominal=0.95)
bias = monitor.detect_bias(residuals, threshold=0.5)
```

### Hierarchical Evaluation

Evaluate forecasts across hierarchical structures (total → region → city → SKU) with level-wise metrics, coherence verification, and cross-level comparison. Essential for revenue forecasting where numbers must reconcile across organizational levels.

```python
from foldcast import hierarchy

tree = hierarchy.HierarchyTree.from_dataframe(df, levels=["region", "city"])
level_metrics = hierarchy.evaluate_levels(tree, forecasts, actuals, metrics=["mae", "rmsse"])
coherence = hierarchy.check_coherence(tree, forecasts)
```

### Metrics

17 forecast accuracy metrics across four categories:

- **Scale-dependent:** MAE, RMSE, MdAE
- **Percentage:** MAPE, sMAPE, MdAPE
- **Scaled:** MASE, RMSSE — the recommended metrics for comparing across series (Hyndman & Koehler, 2006)
- **Distributional:** CRPS, Winkler score — for evaluating probabilistic forecasts

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

fig = visualize.plot_backtest(result, full_series=data)
fig = visualize.plot_residual_diagnostics(residuals)
fig = visualize.plot_model_comparison(rank_table)
fig = visualize.plot_hierarchy_heatmap(level_metrics, metric="mae")
```

## Design Principles

- **Model-agnostic.** Any callable that takes training data and returns forecasts works with foldcast. Use it with statsforecast, prophet, sklearn, PyTorch, or your own custom models.
- **Lightweight.** Four core dependencies: numpy, pandas, scipy, matplotlib. No heavy frameworks, no lock-in.
- **Production-oriented.** Built for the evaluation and monitoring layer, not the modeling layer. foldcast is to forecasting what pytest is to application code.
- **Statistically rigorous.** Proper temporal cross-validation (no future leakage), Diebold-Mariano tests with HAC variance, Model Confidence Sets. Not just metrics — inference.

## Examples

See the [`examples/`](examples/) directory for complete, runnable scripts:

- **[quickstart.py](examples/quickstart.py)** — Backtest a naive model on synthetic revenue data
- **[model_comparison.py](examples/model_comparison.py)** — Compare three models with DM tests and Model Confidence Sets
- **[production_monitoring.py](examples/production_monitoring.py)** — Detect drift, check coverage, diagnose residuals

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a PR.

## License

MIT
