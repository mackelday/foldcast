"""End-to-end integration test: backtest -> compare -> monitor -> hierarchy."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foldcast import backtest, compare, hierarchy, metrics, monitor, visualize


def test_full_workflow():
    """Run the full foldcast workflow on synthetic data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    trend = 100 + 0.5 * np.arange(60)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = rng.normal(0, 2, 60)
    data = pd.Series(trend + seasonal + noise, index=dates, name="revenue")

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
    fig = visualize.plot_backtest(r_naive, full_series=data)
    assert fig is not None
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

    forecasts = df + rng.normal(0, 2, df.shape)
    level_metrics = hierarchy.evaluate_levels(tree, forecasts, df, metrics=["mae", "rmse"])
    assert "mae" in level_metrics.columns

    coherence = hierarchy.check_coherence(tree, df)
    assert coherence.is_coherent is True

    S = tree.summing_matrix()
    assert S.shape == (7, 4)
