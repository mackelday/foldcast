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

# Collect forecasts and actuals
all_forecasts = {}
for name, res in results.items():
    df = res.to_dataframe()
    all_forecasts[name] = df["forecast"].values
actuals_arr = results["naive"].to_dataframe()["actual"].values

# Rank table
table = compare.rank_table(all_forecasts, actuals_arr, metrics=["mae", "rmse", "smape"])
print("Model Ranking:")
print(table)
print()

# Diebold-Mariano test
dm = compare.diebold_mariano(
    all_forecasts["seasonal_naive"], all_forecasts["naive"], actuals_arr
)
print(f"DM test (seasonal_naive vs naive): stat={dm.statistic:.3f}, p={dm.p_value:.4f}")
print()

# Model Confidence Set
mcs = compare.model_confidence_set(all_forecasts, actuals_arr, alpha=0.10)
print(f"Model Confidence Set (alpha=0.10): {mcs.included}")
print()

# Combine best models
best_series = {n: pd.Series(all_forecasts[n]) for n in mcs.included}
actuals_series = pd.Series(actuals_arr)
combined = compare.combine_forecasts(best_series, actuals_series, method="inverse_mse")
print(f"Combination weights: {combined.weights}")

# Plot
fig = visualize.plot_model_comparison(table)
fig.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved model_comparison.png")
