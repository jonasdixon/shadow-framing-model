"""Exploratory checks: the summaries must describe the data, and the geometry check must be able
to find a boundary that has been moved on purpose."""

import numpy as np
import pandas as pd
import pytest

from shadow_zone import eda
from shadow_zone.model import sigmoid
from simulation import simulate_taken


def planted_offset(delta_ft: float, n: int = 200_000, seed: int = 3) -> pd.DataFrame:
    """Taken pitches whose true 50% strike boundary sits `delta_ft` away from edge_dist == 0."""
    rng = np.random.default_rng(seed)
    edge = rng.uniform(-0.8, 1.0, n)
    strike = rng.random(n) < sigmoid(-14.0 * (edge - delta_ft))
    return pd.DataFrame({"edge_dist": edge, "strike": strike.astype(int)})


# ---------- the geometry check ----------


def test_crossing_is_zero_when_the_boundary_is_where_the_geometry_says():
    assert eda.half_strike_distance(planted_offset(0.0)) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("delta", [-0.08, -0.04, 0.05, 0.10])
def test_crossing_recovers_a_boundary_moved_on_purpose(delta):
    """If a real called zone were tighter or wider than `zone.edge_distance`, this must say so."""
    assert eda.half_strike_distance(planted_offset(delta)) == pytest.approx(delta, abs=0.015)


def test_crossing_is_nan_when_the_rate_never_reaches_half():
    rng = np.random.default_rng(0)
    never = pd.DataFrame({"edge_dist": rng.uniform(0.3, 0.8, 20_000), "strike": 0})
    assert np.isnan(eda.half_strike_distance(never))


def test_simulated_data_crosses_near_zero_by_construction():
    df, _, _ = simulate_taken(n=120_000, seed=5)
    assert eda.half_strike_distance(df) == pytest.approx(0.0, abs=0.02)


# ---------- descriptive summaries ----------


def test_response_curve_falls_as_the_pitch_leaves_the_zone():
    df, _, _ = simulate_taken(n=120_000, seed=6)
    g = eda.response_curve(df)
    assert len(g) > 20
    assert g["strike_rate"].iloc[0] > 0.9 > 0.1 > g["strike_rate"].iloc[-1]
    assert g["n"].min() >= 100


def test_response_curve_drops_thin_bins():
    df, _, _ = simulate_taken(n=60_000, seed=7)
    loose = eda.response_curve(df, step=0.05, min_n=1)
    strict = eda.response_curve(df, step=0.05, min_n=1_200)
    assert strict["n"].min() >= 1_200
    assert len(strict) < len(loose)


def test_feature_table_reports_missingness():
    df, _, _ = simulate_taken(n=5_000, seed=8)
    df.loc[df.index[:250], "sz_top"] = np.nan
    t = eda.feature_table(df).set_index("feature")
    assert t.loc["sz_top", "missing_pct"] == pytest.approx(5.0, abs=0.01)
    assert t.loc["edge_dist", "missing_pct"] == 0.0
    assert t.loc["edge_dist", "p25"] < t.loc["edge_dist", "median"] < t.loc["edge_dist", "p75"]


def test_by_region_reports_every_side_and_both_hands():
    df, _, _ = simulate_taken(n=150_000, seed=9)
    t = eda.by_region(df)
    assert set(t.loc[t["split"] == "region", "group"]) == {"high", "low", "inside", "outside"}
    assert set(t.loc[t["split"] == "batter hand", "group"]) == {"L", "R"}
    assert t.loc[t["split"] == "region", "n"].sum() == len(df)
    # the simulation uses one boundary for every side, so no side should look displaced
    assert t["half_strike_in"].abs().max() < 0.6


def test_zone_bounds_check_counts_impossible_zones():
    df, _, _ = simulate_taken(n=4_000, seed=10)
    df.loc[df.index[:7], "sz_top"] = 0.5  # top below the bottom
    out = eda.zone_bounds_check(df)
    assert out["nonpositive_height"] == 7
    assert out["height_outside_1_to_3_ft"] >= 7
    assert 1.5 < out["zone_height_mean_ft"] < 2.1


def test_add_derived_fills_missing_columns_and_leaves_present_ones_alone():
    df, _, _ = simulate_taken(n=3_000, seed=11)
    stripped = df.drop(columns=["z_frac", "region"])
    filled = eda.add_derived(stripped)
    assert {"z_frac", "region"}.issubset(filled.columns)
    assert np.allclose(filled["z_frac"], df["z_frac"])
    assert (filled["region"] == df["region"]).all()
