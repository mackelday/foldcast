"""Hierarchical forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from foldcast import metrics as m
from foldcast._types import CoherenceResult


@dataclass
class HierarchyTree:
    """Represents a hierarchical time series structure.

    Attributes:
        bottom_names: Names of bottom-level series.
        levels: List of dicts mapping level names to lists of bottom series they aggregate.
        level_labels: Names of each hierarchy level (e.g., ["total", "region", "city"]).
    """

    bottom_names: list[str]
    levels: list[dict[str, list[str]]]
    level_labels: list[str]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        levels: list[str],
    ) -> HierarchyTree:
        """Build hierarchy tree from a DataFrame with MultiIndex columns.

        Args:
            df: DataFrame with MultiIndex columns representing the hierarchy.
            levels: Column level names from broadest to narrowest.

        Returns:
            HierarchyTree describing the aggregation structure.
        """
        if not isinstance(df.columns, pd.MultiIndex):
            raise TypeError("DataFrame must have MultiIndex columns")

        bottom_names = ["_".join(str(x) for x in col) for col in df.columns]

        hierarchy_levels: list[dict[str, list[str]]] = []

        # Level 0: total
        hierarchy_levels.append({"total": list(bottom_names)})

        # Intermediate levels
        for depth in range(len(levels) - 1):
            level_map: dict[str, list[str]] = {}
            for col, bname in zip(df.columns, bottom_names, strict=True):
                key = "_".join(str(col[i]) for i in range(depth + 1))
                level_map.setdefault(key, []).append(bname)
            hierarchy_levels.append(level_map)

        # Bottom level: each series maps to itself
        hierarchy_levels.append({bname: [bname] for bname in bottom_names})

        level_labels = ["total"] + levels

        return cls(
            bottom_names=bottom_names,
            levels=hierarchy_levels,
            level_labels=level_labels,
        )

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def n_bottom(self) -> int:
        return len(self.bottom_names)

    def level_names(self, depth: int) -> list[str]:
        """Return node names at a given depth."""
        return list(self.levels[depth].keys())

    def summing_matrix(self) -> np.ndarray:
        """Build the summing matrix S where y = S @ b (b = bottom-level).

        Each row corresponds to a node in the hierarchy (all levels),
        each column to a bottom-level series. Entry S[i,j] = 1 if
        bottom series j contributes to node i.
        """
        all_nodes: list[str] = []
        mapping: list[list[str]] = []

        for level in self.levels:
            for node_name, children in level.items():
                all_nodes.append(node_name)
                mapping.append(children)

        n_rows = len(all_nodes)
        n_cols = self.n_bottom
        S = np.zeros((n_rows, n_cols))

        for i, children in enumerate(mapping):
            for child in children:
                j = self.bottom_names.index(child)
                S[i, j] = 1.0

        return S


def evaluate_levels(
    tree: HierarchyTree,
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute accuracy metrics at each hierarchy level.

    Args:
        tree: HierarchyTree describing the structure.
        forecasts: Forecast DataFrame (same shape/columns as actuals).
        actuals: Actual DataFrame with MultiIndex columns.
        metrics: List of metric names. Defaults to ["mae", "rmse"].

    Returns:
        DataFrame with columns: level, node, and one column per metric.
    """
    if metrics is None:
        metrics = ["mae", "rmse"]

    metric_fns = {"mae": m.mae, "rmse": m.rmse, "mdae": m.mdae, "smape": m.smape}

    bottom_values_actual = {
        "_".join(str(x) for x in col): actuals[col].values for col in actuals.columns
    }
    bottom_values_forecast = {
        "_".join(str(x) for x in col): forecasts[col].values for col in forecasts.columns
    }

    rows = []
    for level_map, level_label in zip(tree.levels, tree.level_labels, strict=True):
        for node_name, children in level_map.items():
            agg_actual = sum(bottom_values_actual[c] for c in children)
            agg_forecast = sum(bottom_values_forecast[c] for c in children)

            row: dict[str, object] = {"level": level_label, "node": node_name}
            for metric_name in metrics:
                fn = metric_fns[metric_name]
                row[metric_name] = fn(agg_actual, agg_forecast)
            rows.append(row)

    return pd.DataFrame(rows)


def check_coherence(
    tree: HierarchyTree,
    forecasts: pd.DataFrame,
    tol: float = 1e-6,
) -> CoherenceResult:
    """Check whether forecasts are coherent (sum consistently across levels).

    Args:
        tree: HierarchyTree describing the structure.
        forecasts: Forecast DataFrame with MultiIndex columns.
        tol: Tolerance for numerical coherence.

    Returns:
        CoherenceResult with coherence status and violation details.
    """
    bottom_values = {
        "_".join(str(x) for x in col): forecasts[col].values for col in forecasts.columns
    }

    max_violation = 0.0
    incoherent_nodes: list[str] = []

    for depth, level_map in enumerate(tree.levels[:-1]):
        for node_name, children in level_map.items():
            expected = sum(bottom_values[c] for c in children)

            if depth < tree.n_levels - 2:
                next_level = tree.levels[depth + 1]
                sub_aggregate = np.zeros_like(expected)
                for _sub_name, sub_children in next_level.items():
                    if all(c in children for c in sub_children):
                        sub_aggregate += sum(bottom_values[c] for c in sub_children)

                violation = float(np.max(np.abs(expected - sub_aggregate)))
                if violation > max_violation:
                    max_violation = violation
                if violation > tol:
                    incoherent_nodes.append(node_name)

    return CoherenceResult(
        is_coherent=max_violation <= tol,
        max_violation=max_violation,
        incoherent_nodes=incoherent_nodes,
    )
