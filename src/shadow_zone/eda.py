"""Checks on the derived features, run before any model is fitted.

Everything downstream rests on `edge_distance`: the shadow zone is defined by it, and a catcher's
effect is estimated from pitches it places near the boundary. It is a derived quantity, not a
measurement, so it is worth asking three things of it before trusting a single framing number.

1. What does it look like? `feature_table` gives the distribution of each derived feature and how
   much of it is missing.
2. Does it relate to the outcome the way the definition claims? `response_curve` gives the raw
   called-strike rate against edge distance with no model involved.
3. Where does it disagree with the data? `half_strike_distance` finds the edge distance at which
   umpires call a strike half the time. If the geometry in `zone.py` matched the zone umpires
   actually call, that crossing would sit at zero. `by_region` splits it by side of the zone,
   which is where the disagreement turns out to live.

The crossing is read off a fine-binned rate curve by linear interpolation. No curve is fitted and
no continuous variable is bucketed into a comparison; the bins exist only to plot and interpolate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shadow_zone.zone import edge_region, zone_height_fraction

DERIVED = ("edge_dist", "z_frac", "sz_top", "sz_bot", "plate_x", "plate_z")


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of each derived feature, with the share of rows missing it."""
    rows = []
    for col in DERIVED:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "n": int(s.notna().sum()),
                "missing_pct": round(100.0 * s.isna().mean(), 4),
                "mean": s.mean(),
                "sd": s.std(),
                "p01": s.quantile(0.01),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "p99": s.quantile(0.99),
            }
        )
    return pd.DataFrame(rows).round(4)


def response_curve(df: pd.DataFrame, step: float = 0.02, lo: float = -0.6, hi: float = 0.8,
                   min_n: int = 100) -> pd.DataFrame:
    """Raw called-strike rate against signed edge distance. No model, no smoothing."""
    d = df.loc[df["edge_dist"].between(lo, hi)]
    bin_ = (np.round(d["edge_dist"] / step) * step).rename("edge_dist_bin")
    g = d.groupby(bin_, observed=True)["strike"].agg(["mean", "size"]).reset_index()
    g = g.loc[g["size"] >= min_n].rename(columns={"mean": "strike_rate", "size": "n"})
    return g.sort_values("edge_dist_bin").reset_index(drop=True)


def half_strike_distance(df: pd.DataFrame, step: float = 0.02, min_n: int = 100) -> float:
    """Edge distance (ft) where the raw called-strike rate crosses 50%.

    Zero means umpires and `zone.edge_distance` agree on where the boundary is. A negative value
    means the called zone is smaller than the geometry says on that side: a pitch has to be inside
    the computed boundary by that much before it is called a strike half the time.

    Returns nan when the curve never crosses 50% over the binned range.
    """
    g = response_curve(df, step=step, min_n=min_n)
    if len(g) < 2:
        return float("nan")
    x = g["edge_dist_bin"].to_numpy()
    y = g["strike_rate"].to_numpy()
    above = y >= 0.5
    cross = np.flatnonzero(above[:-1] & ~above[1:])  # last bin at or above 50%, then below
    if len(cross) == 0:
        return float("nan")
    i = int(cross[-1])
    y0, y1 = y[i], y[i + 1]
    if y0 == y1:
        return float(x[i])
    return float(x[i] + (y0 - 0.5) * (x[i + 1] - x[i]) / (y0 - y1))


def by_region(df: pd.DataFrame, step: float = 0.02, min_n: int = 100) -> pd.DataFrame:
    """The 50% crossing for each side of the zone, and for each batter handedness."""
    d = df.copy()
    if "region" not in d.columns:
        d["region"] = edge_region(d["plate_x"], d["plate_z"], d["sz_top"], d["sz_bot"], d["stand"]).to_numpy()
    rows = [{"split": "all", "group": "all", "n": len(d),
             "half_strike_ft": half_strike_distance(d, step, min_n)}]
    for split, col in (("region", "region"), ("batter hand", "stand")):
        for value, part in d.groupby(col, observed=True):
            rows.append({"split": split, "group": str(value), "n": len(part),
                         "half_strike_ft": half_strike_distance(part, step, min_n)})
    out = pd.DataFrame(rows)
    out["half_strike_in"] = (12.0 * out["half_strike_ft"]).round(2)
    return out.round({"half_strike_ft": 4})


def zone_bounds_check(df: pd.DataFrame) -> dict:
    """Statcast's per-batter zone bounds are estimates. This is what they look like."""
    top, bot = pd.to_numeric(df["sz_top"]), pd.to_numeric(df["sz_bot"])
    height = top - bot
    return {
        "sz_top_mean_ft": round(float(top.mean()), 4),
        "sz_top_sd_ft": round(float(top.std()), 4),
        "sz_bot_mean_ft": round(float(bot.mean()), 4),
        "sz_bot_sd_ft": round(float(bot.std()), 4),
        "zone_height_mean_ft": round(float(height.mean()), 4),
        "zone_height_sd_ft": round(float(height.std()), 4),
        "nonpositive_height": int((height <= 0).sum()),
        "height_outside_1_to_3_ft": int((~height.between(1.0, 3.0)).sum()),
    }


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in `z_frac` and `region` if a frame carries only the stored columns."""
    out = df.copy()
    if "z_frac" not in out.columns:
        out["z_frac"] = zone_height_fraction(out["plate_z"], out["sz_top"], out["sz_bot"])
    if "region" not in out.columns:
        out["region"] = edge_region(out["plate_x"], out["plate_z"], out["sz_top"], out["sz_bot"],
                                    out["stand"]).to_numpy()
    return out
