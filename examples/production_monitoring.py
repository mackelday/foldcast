"""foldcast example: monitor a production forecast for degradation."""

import numpy as np

from foldcast import monitor, visualize

rng = np.random.default_rng(42)

# Simulate historical forecast errors (well-calibrated period)
historical_errors = rng.normal(0, 1.0, 500)

# Simulate recent errors (model has degraded: bias + increased variance)
recent_errors = rng.normal(0.8, 1.5, 50)

# 1. Residual diagnostics
print("=== Residual Diagnostics (recent) ===")
report = monitor.check_residuals(recent_errors)
print(report.summary())

# 2. Drift detection
print("\n=== Drift Detection ===")
drift = monitor.detect_drift(historical_errors, recent_errors, method="psi", threshold=0.2)
print(f"PSI = {drift.statistic:.4f} (threshold={drift.threshold})")
print(f"Alert: {drift.alert}")

# 3. Coverage check
print("\n=== Coverage Check ===")
n = 100
actuals = rng.normal(0, 1, n)
mu = actuals + rng.normal(0.3, 0.5, n)
lower = mu - 1.96
upper = mu + 1.96
coverage = monitor.check_coverage(actuals, lower, upper, nominal=0.95)
print(f"Nominal: {coverage.nominal:.0%}, Empirical: {coverage.empirical:.0%}")
print(f"Alert: {coverage.alert}")

# 4. Bias detection
print("\n=== Bias Detection ===")
bias = monitor.detect_bias(recent_errors, threshold=0.3)
print(f"Mean bias: {bias.mean_bias:.4f}")
print(f"Alert: {bias.alert}")

# Plot diagnostics
fig = visualize.plot_residual_diagnostics(recent_errors)
fig.savefig("residual_diagnostics.png", dpi=150, bbox_inches="tight")
print("\nSaved residual_diagnostics.png")
