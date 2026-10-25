"""Build the dataset of taken pitches (called strike vs ball) with catcher, umpire and edge distance."""

from __future__ import annotations

import pandas as pd

from shadow_zone.data import integrity
from shadow_zone.zone import edge_distance, edge_region, zone_height_fraction

TAKEN = {"called_strike": 1, "ball": 0, "blocked_ball": 0}
# Intentional balls and pitchouts are not attempts to throw a strike. Automatic balls and strikes
# (pitch-clock violations) are never called by the umpire's zone; they are excluded via TAKEN.
NON_ATTEMPT_PITCH_TYPES = {"IN", "PO", "AB", "AS"}


def attach_umpires(df: pd.DataFrame, umpires: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_pk", "hp_umpire_id", "hp_umpire_name"]
    u = umpires[cols].drop_duplicates("game_pk")
    return df.merge(u, on="game_pk", how="left", validate="many_to_one")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the model needs, computed from raw Statcast fields plus the umpire join."""
    out = df.copy()
    out["strike"] = out["description"].map(TAKEN).astype(int)
    out["edge_dist"] = edge_distance(out["plate_x"], out["plate_z"], out["sz_top"], out["sz_bot"])
    out["z_frac"] = zone_height_fraction(out["plate_z"], out["sz_top"], out["sz_bot"])
    out["region"] = edge_region(
        out["plate_x"], out["plate_z"], out["sz_top"], out["sz_bot"], out["stand"]
    ).to_numpy()
    out["count"] = out["balls"].astype(int).astype(str) + "-" + out["strikes"].astype(int).astype(str)
    out["season"] = out["game_year"].astype(int)
    out["catcher"] = out["fielder_2"].astype(int)
    out["umpire"] = out["hp_umpire_id"].astype(int)
    return out.reset_index(drop=True)


def build_taken(df: pd.DataFrame, umpires: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Taken pitches ready for modelling, plus a dict of how many rows each exclusion removed."""
    exclusions = {"all_pitches": len(df)}
    df = attach_umpires(df, umpires)
    taken = df["description"].isin(TAKEN)
    exclusions["swung_or_other_description"] = int((~taken).sum())
    df = df.loc[taken]
    non_attempt = df["pitch_type"].isin(NON_ATTEMPT_PITCH_TYPES)
    exclusions["intentional_ball_or_pitchout"] = int(non_attempt.sum())
    df = df.loc[~non_attempt]
    df, counts = integrity.apply_rules(df)
    exclusions.update({f"rule_{k}": v for k, v in counts.items() if k not in {"rows_in", "rows_out"}})
    exclusions["taken_pitches_modelled"] = len(df)
    return add_derived(df), exclusions
