"""Plotting utilities for forecast evaluation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        ax.plot(
            full_series.index,
            full_series.values,
            color="0.7",
            linewidth=1,
            label="Observed",
            zorder=1,
        )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result.folds), 10)))

    for fold in result.folds:
        color = colors[fold.fold_id % len(colors)]
        ax.plot(
            fold.actuals.index,
            fold.actuals.values,
            "o",
            color=color,
            markersize=4,
            zorder=2,
        )
        ax.plot(
            fold.forecasts.index,
            fold.forecasts.values,
            "-",
            color=color,
            linewidth=1.5,
            zorder=3,
        )
        ax.axvline(fold.cutoff, color=color, linestyle=":", alpha=0.3)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(
        f"Backtest ({result.strategy}, {len(result.folds)} folds, h={result.horizon})"
    )
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
    from scipy import stats as sp_stats

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
            ck = np.sum(r_centered[k:] * r_centered[: n - k]) / n
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
    sp_stats.probplot(r, dist="norm", plot=ax)
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

    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    colors = plt.cm.YlOrRd(norm(values))

    ax.barh(np.arange(len(nodes)), values, color=colors, edgecolor="white")
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(
        [f"[{lev}] {n}" for lev, n in zip(levels, nodes, strict=True)]
    )
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Hierarchy: {metric.upper()} by Node")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig
