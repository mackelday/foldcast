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


def naive_model(train: pd.Series, horizon: int) -> pd.Series:
    """Repeat last observed value."""
    last = train.iloc[-1]
    idx = pd.date_range(
        train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq
    )
    return pd.Series(last, index=idx, name=train.name)


# Run expanding-window backtest
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

# Plot results
fig = visualize.plot_backtest(result, full_series=revenue)
fig.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nSaved backtest_result.png")
