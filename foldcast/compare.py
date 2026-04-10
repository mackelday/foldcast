"""Statistical forecast model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from foldcast import metrics as m
from foldcast._types import CombineResult, DMResult, MCSResult


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

    return DMResult(
        statistic=float(statistic), p_value=float(p_value), conclusion=conclusion
    )


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
        losses = {}
        for name in remaining:
            e = y - models[name]
            losses[name] = np.mean(e**2) if loss == "squared" else np.mean(np.abs(e))

        worst = max(remaining, key=lambda n: losses[n])
        best = min(remaining, key=lambda n: losses[n])
        if worst == best:
            break

        dm = diebold_mariano(models[best], models[worst], y, loss=loss, alpha=alpha)
        p_values[worst] = dm.p_value

        if dm.p_value < alpha:
            remaining.remove(worst)
            eliminated.append(worst)
        else:
            break

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
        metrics: List of metric names from foldcast.metrics.
            Defaults to ["mae", "rmse"].

    Returns:
        DataFrame indexed by model name, one column per metric, sorted by first metric.
    """
    if metrics is None:
        metrics = ["mae", "rmse"]

    y = np.asarray(actuals, dtype=float)
    metric_fns = {
        "mae": m.mae,
        "rmse": m.rmse,
        "mdae": m.mdae,
        "mape": m.mape,
        "smape": m.smape,
        "mdape": m.mdape,
    }

    rows = {}
    for name, f in forecasts.items():
        f_arr = np.asarray(f, dtype=float)
        rows[name] = {metric: metric_fns[metric](y, f_arr) for metric in metrics}

    df = pd.DataFrame(rows).T
    df.index.name = "model"
    return df.sort_values(metrics[0])
