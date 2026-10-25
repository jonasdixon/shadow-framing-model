"""Calibration of predicted called-strike probabilities.

The main curve is smooth: observed outcome regressed on a spline of the predicted logit, so there
are no arbitrary bin edges. Decile points are drawn only as a descriptive overlay.
Slope and intercept come from a logistic recalibration y ~ a + b * logit(p); a perfectly calibrated
model has a = 0 and b = 1. Intervals resample whole games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def smooth_calibration(y, p, grid=None, weights=None) -> pd.DataFrame:
    """Observed rate as a smooth function of predicted probability."""
    z = logit(p).reshape(-1, 1)
    tf = SplineTransformer(n_knots=7, degree=3, knots="quantile", extrapolation="constant").fit(z)
    clf = LogisticRegression(C=1e3, max_iter=1000).fit(tf.transform(z), y, sample_weight=weights)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    obs = clf.predict_proba(tf.transform(logit(grid).reshape(-1, 1)))[:, 1]
    return pd.DataFrame({"predicted": grid, "observed": obs})


def slope_intercept(y, p, weights=None) -> tuple[float, float]:
    """(intercept, slope) of the logistic recalibration of y on logit(p)."""
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y, sample_weight=weights)
    return float(clf.intercept_[0]), float(clf.coef_[0, 0])


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss_(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(y, p, weights=None) -> float:
    """Mean absolute gap between the smooth calibration curve and the diagonal, weighted by the
    distribution of predictions (an ECE without bins)."""
    curve = smooth_calibration(y, p, grid=np.asarray(p), weights=weights)
    return float(np.average(np.abs(curve["observed"] - curve["predicted"]), weights=weights))


def summarize(y, p, games, n_boot: int = 100, seed: int = 0) -> dict:
    """Brier, log loss, calibration slope/intercept with game-cluster bootstrap intervals."""
    y, p = np.asarray(y), np.asarray(p)
    a, b = slope_intercept(y, p)
    codes, uniq = pd.factorize(games)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boots.append(slope_intercept(y, p, weights=counts[codes]))
    boots = np.array(boots)
    return {
        "n": len(y), "brier": brier(y, p), "log_loss": log_loss_(y, p),
        "ece_smooth": expected_calibration_error(y, p),
        "cal_intercept": a, "cal_intercept_lo": float(np.percentile(boots[:, 0], 2.5)),
        "cal_intercept_hi": float(np.percentile(boots[:, 0], 97.5)),
        "cal_slope": b, "cal_slope_lo": float(np.percentile(boots[:, 1], 2.5)),
        "cal_slope_hi": float(np.percentile(boots[:, 1], 97.5)),
    }
