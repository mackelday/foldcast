# foldcast

Backtesting, comparison, and monitoring for production time series forecasts.

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
mcs = compare.model_confidence_set(
    {"ets": f1, "arima": f2, "prophet": f3}, actuals, alpha=0.10
)
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
