"""Converting framing effects to runs.

A taken pitch that becomes a strike instead of a ball is worth different amounts in different
counts (strike three ends the plate appearance; ball four walks the batter). The value of a call
is estimated from the data: in each count, the average change in run expectancy on called balls
minus on called strikes, from the batting side's point of view (`delta_run_exp`, which is negative
when the pitch helps the defence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shadow_zone.model import FramingModel, sigmoid


def strike_value_by_count(taken: pd.DataFrame, min_n: int = 200) -> pd.DataFrame:
    """Runs saved by the defence when a taken pitch is called a strike rather than a ball, per count."""
    d = taken.dropna(subset=["delta_run_exp"])
    g = d.groupby(["count", "strike"])["delta_run_exp"].agg(["mean", "size"]).unstack("strike")
    mean_ball, mean_strike = g[("mean", 0)], g[("mean", 1)]
    n_ball, n_strike = g[("size", 0)], g[("size", 1)]
    tbl = pd.DataFrame({
        "rv_strike": mean_ball - mean_strike, "n_ball": n_ball, "n_strike": n_strike,
    }).reset_index()
    overall = d.groupby("strike")["delta_run_exp"].mean()
    league = overall[0] - overall[1]
    thin = (tbl["n_ball"] < min_n) | (tbl["n_strike"] < min_n)
    tbl["rv_strike"] = np.where(thin, league, tbl["rv_strike"])
    return tbl


def catcher_runs(
    model: FramingModel,
    df: pd.DataFrame,
    effects: pd.DataFrame,
    rv: pd.DataFrame,
    block: str = "catcher",
) -> pd.DataFrame:
    """Runs from each catcher's (umpire's) effect over the taken pitches he was involved in.

    extra strikes = sum over his pitches of P(strike | fitted effect) - P(strike | average effect);
    runs = extra strikes weighted by the count-specific value of a strike. Per-1,000 and
    per-150-game rates use taken pitches and full-game equivalents; the standard error propagates
    the effect's standard error through the average slope of the strike probability.
    """
    value = df["count"].map(rv.set_index("count")["rv_strike"]).to_numpy()
    eta = model.linear_predictor(df)
    eta0 = model.linear_predictor(df, zero=(block,))
    p, p0 = sigmoid(eta), sigmoid(eta0)
    work = pd.DataFrame({
        block: df[block].to_numpy(), "extra": p - p0, "runs": (p - p0) * value,
        "slope_value": p * (1 - p) * value, "game_pk": df["game_pk"].to_numpy(),
    })
    agg = work.groupby(block).agg(
        extra_strikes=("extra", "sum"), runs=("runs", "sum"),
        runs_slope=("slope_value", "sum"), pitches=("extra", "size"),
    ).reset_index()
    per_game = df.groupby([block, "game_pk"]).size()
    full = per_game[per_game >= 50]
    pitches_per_game = float(full.mean()) if len(full) else float(per_game.mean())
    out = effects[[block, "effect_logit", "se"]].merge(agg, on=block, how="left")
    out["runs_se"] = out["runs_slope"].abs() * out["se"]
    out["runs_per_1000"] = 1000 * out["runs"] / out["pitches"]
    out["runs_per_150g"] = out["runs"] / (out["pitches"] / pitches_per_game) * 150
    out["runs_per_150g_se"] = out["runs_se"] / (out["pitches"] / pitches_per_game) * 150
    out["extra_strikes_per_1000"] = 1000 * out["extra_strikes"] / out["pitches"]
    return out.drop(columns="runs_slope")
