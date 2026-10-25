"""A few hundred real taken pitches (whole games) shipped for offline tests and the quickstart."""

from __future__ import annotations

from importlib import resources

import pandas as pd

from shadow_zone.data.prepare import add_derived

SAMPLE_COLUMNS = [
    "game_pk", "game_year", "pitcher", "batter", "fielder_2", "hp_umpire_id", "hp_umpire_name",
    "stand", "p_throws", "balls", "strikes", "pitch_type", "description",
    "plate_x", "plate_z", "sz_top", "sz_bot", "delta_run_exp",
]


def load_sample() -> pd.DataFrame:
    with resources.files("shadow_zone").joinpath("data/sample_taken_pitches.csv").open("rb") as fh:
        return add_derived(pd.read_csv(fh))
