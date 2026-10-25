import numpy as np
import pandas as pd
import pytest

from shadow_zone import calibration, runs
from shadow_zone.data import integrity
from shadow_zone.data.prepare import build_taken
from shadow_zone.effects import coefficient_covariances, effect_table
from shadow_zone.model import FramingModel
from shadow_zone.zone import (
    BALL_RADIUS_FT,
    ZONE_HALF_WIDTH_FT,
    edge_distance,
    edge_region,
    zone_height_fraction,
)
from simulation import simulate_taken

# ---------- zone geometry ----------


def test_edge_distance_sign_and_magnitude():
    top, bot = 3.4, 1.6
    inside_centre = edge_distance(0.0, 2.5, top, bot)
    just_outside_right = edge_distance(ZONE_HALF_WIDTH_FT + 0.1, 2.5, top, bot)
    on_edge = edge_distance(ZONE_HALF_WIDTH_FT, 2.5, top, bot)
    assert inside_centre < 0 < just_outside_right
    assert np.isclose(just_outside_right, 0.1) and np.isclose(on_edge, 0.0)
    # depth inside is distance to the nearest side: the zone is 0.9 ft deep vertically, 0.83 wide
    nearest = min(ZONE_HALF_WIDTH_FT, 2.5 - (bot - BALL_RADIUS_FT), (top + BALL_RADIUS_FT) - 2.5)
    assert np.isclose(inside_centre, -nearest, atol=1e-3)


def test_edge_distance_to_a_corner_is_euclidean():
    top, bot = 3.4, 1.6
    d = edge_distance(ZONE_HALF_WIDTH_FT + 0.3, top + BALL_RADIUS_FT + 0.4, top, bot)
    assert np.isclose(d, 0.5, atol=1e-6)


def test_edge_distance_is_mirror_symmetric():
    assert np.isclose(edge_distance(0.9, 2.5, 3.4, 1.6), edge_distance(-0.9, 2.5, 3.4, 1.6))


def test_zone_height_fraction_and_regions():
    assert np.isclose(zone_height_fraction(1.6, 3.4, 1.6), 0) and np.isclose(zone_height_fraction(3.4, 3.4, 1.6), 1)
    reg = edge_region([-1.0, 1.0, 0.0, 0.0], [2.5, 2.5, 4.5, 0.5], 3.4, 1.6, ["R", "R", "R", "R"])
    assert list(reg) == ["inside", "outside", "high", "low"]
    reg_l = edge_region([-1.0, 1.0], [2.5, 2.5], 3.4, 1.6, ["L", "L"])
    assert list(reg_l) == ["outside", "inside"]


# ---------- integrity and exclusions ----------


def _raw(n=6):
    return pd.DataFrame({
        "game_pk": [1] * n, "at_bat_number": range(1, n + 1), "pitch_number": [1] * n,
        "game_year": 2023, "description": ["ball"] * n, "pitch_type": ["FF"] * n,
        "plate_x": [0.1] * n, "plate_z": [2.5] * n, "sz_top": 3.4, "sz_bot": 1.6,
        "fielder_2": 5, "balls": 0, "strikes": 0, "stand": "R", "p_throws": "R",
        "delta_run_exp": 0.05, "pitcher": 1, "batter": 2,
    })


def test_exclusions_account_for_every_planted_problem():
    raw = _raw(10)
    raw.loc[0, "description"] = "swinging_strike"  # not a taken pitch
    raw.loc[1, "description"] = "automatic_ball"  # pitch-clock ball, not an umpire call
    raw.loc[2, "pitch_type"] = "IN"  # intentional ball
    raw.loc[3, "plate_x"] = np.nan
    raw.loc[4, "plate_z"] = 12.0  # impossible height
    raw.loc[5, "fielder_2"] = np.nan
    key = ["game_pk", "at_bat_number", "pitch_number"]
    raw.loc[6, key] = raw.loc[7, key].to_numpy()
    umpires = pd.DataFrame({"game_pk": [1], "hp_umpire_id": [77], "hp_umpire_name": ["A"]})
    taken, ex = build_taken(raw, umpires)
    assert ex["swung_or_other_description"] == 2  # swinging_strike and automatic_ball
    assert ex["intentional_ball_or_pitchout"] == 1
    assert ex["rule_duplicate_key"] == 1 and ex["rule_missing_location"] == 1
    assert ex["rule_impossible_values"] == 1 and ex["rule_missing_catcher"] == 1
    assert len(taken) == ex["taken_pitches_modelled"] == 3
    assert set(taken["umpire"]) == {77}


def test_missing_umpire_join_is_counted_not_silently_dropped():
    raw = _raw(4)
    raw.loc[2:, "game_pk"] = 2
    umpires = pd.DataFrame({"game_pk": [1], "hp_umpire_id": [77], "hp_umpire_name": ["A"]})
    taken, ex = build_taken(raw, umpires)
    assert ex["rule_missing_umpire"] == 2 and len(taken) == 2


def test_handedness_sign_check_detects_flipped_side():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"stand": ["R"] * 500 + ["L"] * 500,
                       "plate_x": np.r_[rng.normal(0.2, 0.7, 500), rng.normal(-0.2, 0.7, 500)]})
    assert integrity.handedness_sign_check(df) > 0.2
    flipped = df.assign(plate_x=np.where(df["stand"] == "L", -df["plate_x"], df["plate_x"]))
    assert integrity.handedness_sign_check(flipped) < 0.05


# ---------- model: recovery of planted effects ----------


@pytest.fixture(scope="module")
def fitted():
    df, true_c, true_u = simulate_taken(n=200_000, n_games=3_000)
    model = FramingModel(scales={"catcher": 0.2, "umpire": 0.2, "pitcher": 0.3, "batter": 0.2}).fit(df)
    return df, true_c, true_u, model


def test_planted_catcher_and_umpire_effects_are_recovered(fitted):
    df, true_c, true_u, model = fitted
    ec = model.effects("catcher").set_index("catcher")["effect_logit"].reindex(range(len(true_c)))
    eu = model.effects("umpire").set_index("umpire")["effect_logit"].reindex(range(len(true_u)))
    assert np.corrcoef(ec, true_c)[0, 1] > 0.85
    assert np.corrcoef(eu, true_u)[0, 1] > 0.9


def test_strike_probability_falls_moving_out_through_the_edge(fitted):
    df, *_ , model = fitted
    d = df.loc[df["region"].eq("outside") & df["stand"].eq("R")].copy()
    d = d.iloc[:2000].copy()
    pr = []
    for e in (-0.3, -0.1, 0.0, 0.1, 0.3):
        t = d.copy()
        t["edge_dist"] = e
        pr.append(model.predict(t, zero=("catcher", "umpire")).mean())
    assert np.all(np.diff(pr) < 0)


def test_neutral_catcher_prediction_differs_only_through_catcher_block(fitted):
    df, *_ , model = fitted
    sample = df.iloc[:500]
    full, neutral = model.linear_predictor(sample), model.linear_predictor(sample, zero=("catcher",))
    sl = model.slices["catcher"]
    manual = np.asarray(model.design(sample)[:, sl] @ model.coef[sl]).ravel()
    assert np.allclose(full - neutral, manual)


def test_effect_table_orders_and_intervals(fitted):
    df, true_c, _, model = fitted
    covs = coefficient_covariances(model, df, ("catcher",))
    tbl = effect_table(model, df, "catcher", covs)
    assert tbl["effect_logit"].is_monotonic_decreasing
    assert (tbl["se"] > 0).all() and (tbl["edge_pp_hi95"] > tbl["edge_pp_lo95"]).all()
    best = tbl.iloc[0]["catcher"]
    assert true_c[int(best)] > np.median(true_c)  # the top-ranked catcher really is above median


# ---------- calibration ----------


def test_calibrated_predictions_have_slope_one_and_overconfident_ones_do_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 40_000)
    y = (rng.random(40_000) < p).astype(int)
    a, b = calibration.slope_intercept(y, p)
    assert abs(a) < 0.08 and abs(b - 1) < 0.06
    overconfident = calibration.logit(p) * 2.0
    a2, b2 = calibration.slope_intercept(y, 1 / (1 + np.exp(-overconfident)))
    assert 0.4 < b2 < 0.6
    assert calibration.expected_calibration_error(y, p) < 0.02


def test_brier_and_logloss_arithmetic():
    y, p = np.array([1, 0, 1, 0]), np.array([0.8, 0.2, 0.6, 0.4])
    assert np.isclose(calibration.brier(y, p), np.mean([0.04, 0.04, 0.16, 0.16]))
    assert np.isclose(calibration.log_loss_(y, p), -np.mean(np.log([0.8, 0.8, 0.6, 0.6])))


# ---------- runs ----------


def test_strike_value_by_count_is_ball_minus_strike_run_expectancy():
    rng = np.random.default_rng(0)
    rows = []
    for count, (ball_v, strike_v) in {"0-0": (0.03, -0.04), "3-2": (0.30, -0.45)}.items():
        for s, v in ((0, ball_v), (1, strike_v)):
            rows.append(pd.DataFrame({"count": count, "strike": s, "delta_run_exp": v + rng.normal(0, 1e-4, 300)}))
    rv = runs.strike_value_by_count(pd.concat(rows)).set_index("count")["rv_strike"]
    assert np.isclose(rv["0-0"], 0.07, atol=1e-3) and np.isclose(rv["3-2"], 0.75, atol=1e-3)


def test_catcher_runs_zero_for_zero_effect_and_scales_with_value(fitted):
    df, *_, model = fitted
    rv = pd.DataFrame({"count": sorted(df["count"].unique()), "rv_strike": 0.1})
    eff = model.effects("catcher").assign(se=0.05)
    out = runs.catcher_runs(model, df, eff, rv).set_index("catcher")
    # runs = extra strikes x 0.1 exactly when every count is worth 0.1
    assert np.allclose(out["runs"], 0.1 * out["extra_strikes"])
    # a catcher with a positive fitted effect gains strikes; negative loses them
    assert (np.sign(out["extra_strikes"]) == np.sign(out["effect_logit"])).mean() > 0.95
