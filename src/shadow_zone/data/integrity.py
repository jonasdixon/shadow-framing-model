"""Integrity checks on the joined Statcast + umpire data.

Each rule returns a boolean mask (True = bad row). `prepare.build_taken` applies them in the order
listed in RULES and records how many rows each one removed, so the exclusions are a table, not a
silent filter.
"""

from __future__ import annotations

import pandas as pd

KEY = ["game_pk", "at_bat_number", "pitch_number"]
PLATE_X_MAX_FT = 4.0
PLATE_Z_RANGE_FT = (-3.0, 7.0)  # balls in the dirt extrapolate below 0; they are real, just far from the zone
ZONE_TOP_RANGE_FT = (2.0, 5.0)
ZONE_BOT_RANGE_FT = (0.5, 3.0)


def duplicate_key(df: pd.DataFrame) -> pd.Series:
    return df.duplicated(KEY, keep="first")


def missing_location(df: pd.DataFrame) -> pd.Series:
    return df[["plate_x", "plate_z"]].isna().any(axis=1)


def missing_zone_bounds(df: pd.DataFrame) -> pd.Series:
    return df[["sz_top", "sz_bot"]].isna().any(axis=1)


def impossible_values(df: pd.DataFrame) -> pd.Series:
    """Locations or zone bounds no real pitch/batter produces (tracking or data-entry errors)."""
    return (
        (df["plate_x"].abs() > PLATE_X_MAX_FT)
        | ~df["plate_z"].between(*PLATE_Z_RANGE_FT)
        | ~df["sz_top"].between(*ZONE_TOP_RANGE_FT)
        | ~df["sz_bot"].between(*ZONE_BOT_RANGE_FT)
        | (df["sz_top"] <= df["sz_bot"])
    )


def missing_catcher(df: pd.DataFrame) -> pd.Series:
    return df["fielder_2"].isna()


def missing_umpire(df: pd.DataFrame) -> pd.Series:
    return df["hp_umpire_id"].isna()


def handedness_sign_check(df: pd.DataFrame) -> float:
    """Mean plate_x against right-handed batters minus mean against left-handed batters.

    In the catcher's view, right-handed batters stand on the negative-x side and pitchers work
    away from them, so the difference should be clearly positive. A negative value means the sign
    convention is flipped for at least one handedness in the data.
    """
    return float(df.loc[df["stand"] == "R", "plate_x"].mean() - df.loc[df["stand"] == "L", "plate_x"].mean())


def mid_game_catcher_changes(df: pd.DataFrame) -> int:
    """Games where one fielding side used more than one catcher. Shows `fielder_2` is per pitch."""
    side = df["inning_topbot"]
    per = df.groupby(["game_pk", side])["fielder_2"].nunique()
    return int((per > 1).sum())


RULES = [
    ("duplicate_key", duplicate_key),
    ("missing_location", missing_location),
    ("missing_zone_bounds", missing_zone_bounds),
    ("impossible_values", impossible_values),
    ("missing_catcher", missing_catcher),
    ("missing_umpire", missing_umpire),
]


def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop flagged rows rule by rule; each row is counted under the first rule that catches it."""
    kept, counts = df, {"rows_in": len(df)}
    for name, rule in RULES:
        bad = rule(kept)
        counts[name] = int(bad.sum())
        kept = kept.loc[~bad]
    counts["rows_out"] = len(kept)
    return kept, counts
