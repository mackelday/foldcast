"""Production forecast monitoring: drift detection, coverage, residual diagnostics."""

from __future__ import annotations

import numpy as np
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

    lb_stat, lb_pval = _ljung_box(r, lags)
    jb_stat, jb_pval = stats.jarque_bera(r)
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
        ck = np.sum(x_centered[k:] * x_centered[: n - k]) / n
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

    y = x_sq[lags:]
    X = np.column_stack([x_sq[lags - k : n - k] for k in range(1, lags + 1)])
    X = np.column_stack([np.ones(len(y)), X])

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
    edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=edges)[0].astype(float)

    exp_prop = np.maximum(exp_counts / exp_counts.sum(), 1e-8)
    act_prop = np.maximum(act_counts / act_counts.sum(), 1e-8)

    psi = np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop))
    return float(psi)
