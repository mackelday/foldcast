"""Result dataclasses and type aliases for foldcast."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FoldResult:
    """Result from a single backtest fold."""

    fold_id: int
    cutoff: pd.Timestamp
    forecasts: pd.Series
    actuals: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated result from a full backtest run."""

    folds: Sequence[FoldResult]
    horizon: int
    strategy: str

    def to_dataframe(self) -> pd.DataFrame:
        """Combine all folds into a single DataFrame."""
        rows = []
        for fold in self.folds:
            df = pd.DataFrame({
                "forecast": fold.forecasts,
                "actual": fold.actuals,
            })
            df["fold_id"] = fold.fold_id
            df["cutoff"] = fold.cutoff
            rows.append(df)
        return pd.concat(rows)

    def summary(self) -> str:
        """Human-readable summary of backtest results."""
        n_folds = len(self.folds)
        all_errors = pd.concat([f.actuals - f.forecasts for f in self.folds])
        mae = all_errors.abs().mean()
        rmse = (all_errors**2).mean() ** 0.5
        return (
            f"BacktestResult(strategy={self.strategy}, folds={n_folds}, "
            f"horizon={self.horizon}, MAE={mae:.4f}, RMSE={rmse:.4f})"
        )


@dataclass(frozen=True)
class DMResult:
    """Result of a Diebold-Mariano test."""

    statistic: float
    p_value: float
    conclusion: str

    def __repr__(self) -> str:
        return (
            f"DMResult(statistic={self.statistic:.4f}, "
            f"p_value={self.p_value:.4f}, conclusion='{self.conclusion}')"
        )


@dataclass(frozen=True)
class MCSResult:
    """Result of Model Confidence Set procedure."""

    included: list[str]
    excluded: list[str]
    p_values: dict[str, float]
    elimination_order: list[str]
    alpha: float


@dataclass(frozen=True)
class CombineResult:
    """Result of forecast combination."""

    combined: pd.Series
    weights: dict[str, float]
    method: str


@dataclass(frozen=True)
class DriftResult:
    """Result of forecast drift detection."""

    statistic: float
    threshold: float
    alert: bool
    method: str


@dataclass(frozen=True)
class CoverageResult:
    """Result of prediction interval coverage check."""

    nominal: float
    empirical: float
    n_observations: int
    n_covered: int
    alert: bool


@dataclass(frozen=True)
class ResidualReport:
    """Comprehensive residual diagnostics report."""

    ljung_box_statistic: float
    ljung_box_p_value: float
    jarque_bera_statistic: float
    jarque_bera_p_value: float
    arch_statistic: float
    arch_p_value: float
    mean_residual: float
    std_residual: float

    def summary(self) -> str:
        """Human-readable diagnostics summary."""
        lines = [
            "Residual Diagnostics:",
            f"  Mean: {self.mean_residual:.4f}, Std: {self.std_residual:.4f}",
            f"  Ljung-Box: stat={self.ljung_box_statistic:.2f}, "
            f"p={self.ljung_box_p_value:.4f}",
            f"  Jarque-Bera: stat={self.jarque_bera_statistic:.2f}, "
            f"p={self.jarque_bera_p_value:.4f}",
            f"  ARCH: stat={self.arch_statistic:.2f}, "
            f"p={self.arch_p_value:.4f}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BiasResult:
    """Result of forecast bias detection."""

    cumulative_bias: float
    mean_bias: float
    alert: bool
    threshold: float


@dataclass(frozen=True)
class CoherenceResult:
    """Result of hierarchical coherence check."""

    is_coherent: bool
    max_violation: float
    incoherent_nodes: list[str]
