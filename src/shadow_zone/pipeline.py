"""End-to-end run: prepare -> tune -> fit -> evaluate -> effects -> runs -> reliability -> figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_zone import calibration, effects, reliability, runs, viz
from shadow_zone.data import integrity
from shadow_zone.data.prepare import build_taken
from shadow_zone.data.statcast import fetch_home_plate_umpires, load_cached
from shadow_zone.model import DEFAULT_SCALES, RANDOM_BLOCKS, FramingModel, tune_scales

SEASON_DATES = {2023: ("2023-03-20", "2023-10-02"), 2024: ("2024-03-20", "2024-10-02")}


def prepare_seasons(cache: str, years: list[int], out_dir: str) -> pd.DataFrame:
    """Build taken-pitch data for each season, write parquet plus the exclusions table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    umpires_path = Path(cache) / "umpires.parquet"
    umpires = pd.read_parquet(umpires_path) if umpires_path.exists() else None
    frames, rows, checks = [], [], {}
    for y in years:
        lo, hi = SEASON_DATES[y]
        raw = load_cached(cache, lo, hi)
        if umpires is None or not set(raw["game_pk"]).issubset(set(umpires["game_pk"])):
            umpires = fetch_home_plate_umpires(lo, hi, cache)
        taken, ex = build_taken(raw, umpires)
        rows.append({"season": y, **ex})
        checks[y] = {
            "handedness_sign_check_ft": integrity.handedness_sign_check(taken),
            "games_with_two_or_more_catchers_for_one_side": integrity.mid_game_catcher_changes(raw),
            "umpire_games_without_home_plate_official": int(
                umpires.loc[umpires["game_pk"].isin(raw["game_pk"]), "hp_umpire_id"].isna().sum()
            ),
        }
        taken.to_parquet(out / f"taken_{y}.parquet", index=False)
        frames.append(taken)
    pd.DataFrame(rows).to_csv(out / "exclusions.csv", index=False)
    (out / "integrity_checks.json").write_text(json.dumps(checks, indent=1))
    return pd.concat(frames, ignore_index=True)


def _shadow(df: pd.DataFrame, band: float) -> pd.Series:
    return df["edge_dist"].abs() <= band


def evaluate_holdout(train: pd.DataFrame, test: pd.DataFrame, scales: dict, band: float) -> tuple[dict, FramingModel]:
    """Log loss and calibration on `test` for the full model and two ablations."""
    full = FramingModel(scales=scales).fit(train)
    no_entities = FramingModel(scales={b: 1e-4 for b in RANDOM_BLOCKS}).fit(train)
    y = test["strike"].to_numpy()
    sh = _shadow(test, band).to_numpy()
    p_full = full.predict(test)
    variants = {
        "full": p_full,
        "neutral_catcher_and_umpire": full.predict(test, zero=("catcher", "umpire")),
        "no_catcher_umpire_pitcher_batter_effects_fit": no_entities.predict(test),
        "base_rate": np.full(len(test), train["strike"].mean()),
    }
    table = {}
    for name, p in variants.items():
        table[name] = {
            "log_loss_all": calibration.log_loss_(y, p),
            "log_loss_shadow": calibration.log_loss_(y[sh], p[sh]),
            "brier_shadow": calibration.brier(y[sh], p[sh]),
        }
    summary = {
        "n_test": len(test), "n_test_shadow": int(sh.sum()), "variants": table,
        "calibration_all": calibration.summarize(y, p_full, test["game_pk"].to_numpy()),
        "calibration_shadow": calibration.summarize(y[sh], p_full[sh], test["game_pk"].to_numpy()[sh]),
    }
    return summary, full


def run_all(taken: pd.DataFrame, out_dir: str, player_names: dict | None = None, train_year: int = 2023,
            test_year: int = 2024, tune: bool = True, shadow_ft: float = 0.25, min_pitches: int = 1000) -> dict:
    player_names = player_names or {}
    out = Path(out_dir)
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    train = taken.loc[taken["season"] == train_year].reset_index(drop=True)
    test = taken.loc[taken["season"] == test_year].reset_index(drop=True)

    scales = dict(DEFAULT_SCALES)
    if tune:
        scales, log = tune_scales(train)
        log.to_csv(tables / "scale_tuning.csv", index=False)
    elif (tables / "scales.json").exists():
        scales = json.loads((tables / "scales.json").read_text())
    (tables / "scales.json").write_text(json.dumps(scales, indent=1))

    summary, model_train = evaluate_holdout(train, test, scales, shadow_ft)
    summary["scales"] = scales
    p_test = model_train.predict(test)
    curve = calibration.smooth_calibration(test["strike"].to_numpy(), p_test)
    viz.calibration_plot(curve, test["strike"].to_numpy(), p_test, str(figs / "calibration.png"))
    viz.residual_map(test, p_test, str(figs / "residual_map.png"))
    viz.edge_curves(test, p_test, str(figs / "edge_response.png"))
    viz.probability_map(model_train, str(figs / "probability_map.png"))

    rv = runs.strike_value_by_count(taken)
    rv.to_csv(tables / "strike_value_by_count.csv", index=False)
    summary["league_strike_value_runs"] = float(
        taken.dropna(subset=["delta_run_exp"]).groupby("strike")["delta_run_exp"].mean().pipe(lambda s: s[0] - s[1])
    )

    fits = {train_year: (model_train, train), test_year: (FramingModel(scales=scales).fit(test), test)}
    for year, (m, d) in fits.items():
        covs = effects.coefficient_covariances(m, d)
        for block in ("catcher", "umpire"):
            tbl = effects.effect_table(m, d, block, covs, shadow_ft)
            if block == "catcher":
                extra = tbl.drop(columns=[c for c in ("effect_logit", "se", "pitches") if c in tbl.columns])
                tbl = runs.catcher_runs(m, d, tbl, rv, block).merge(extra, on="catcher")
                tbl.insert(1, "catcher_name", tbl["catcher"].map(player_names).fillna(tbl["catcher"].astype(str)))
            else:
                names = d.groupby("umpire")["hp_umpire_name"].first()
                tbl["umpire_name"] = tbl["umpire"].map(names)
            tbl = tbl.loc[tbl["pitches"] >= min_pitches].reset_index(drop=True)
            tbl.insert(0, "season", year)
            tbl.round(4).to_csv(tables / f"{block}_effects_{year}.csv", index=False)
            if block == "umpire":
                viz.forest(tbl.assign(se_pp=tbl["edge_pp_se"]), "umpire_name", str(figs / f"umpire_effects_{year}.png"),
                           value="edge_pp", se="se_pp", top=10,
                           title="Edge called-strike probability vs average umpire (points)")
            else:
                viz.forest(tbl, "catcher_name", str(figs / f"catcher_runs_{year}.png"),
                           title=f"{year} framing runs per 150 games (95% interval)")
        region = effects.region_residuals(m, d, "catcher", shadow_ft)
        region = region.loc[region["pitches"] >= 100]
        region.round(3).to_csv(tables / f"catcher_region_residuals_{year}.csv", index=False)

    summary["reliability"] = {
        "split_half_catcher_train_season": reliability.split_half(train, scales, "catcher", min_pitches),
        "split_half_umpire_train_season": reliability.split_half(train, scales, "umpire", 500),
        "season_to_season_catcher": reliability.season_to_season(train, test, scales, "catcher", 2 * min_pitches),
        "season_to_season_umpire": reliability.season_to_season(train, test, scales, "umpire", 1000),
    }
    (tables / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
