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

    Closed-form: CRPS = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
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
