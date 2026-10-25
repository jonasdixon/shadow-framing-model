"""Write the offline sample (a few whole games of taken pitches) shipped inside the package.

Usage: python scripts/make_sample.py --cache /path/to/cache
"""

import argparse

import numpy as np
import pandas as pd

from shadow_zone.data.prepare import build_taken
from shadow_zone.data.sample import SAMPLE_COLUMNS
from shadow_zone.data.statcast import load_cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--games", type=int, default=16)
    a = ap.parse_args()
    raw = load_cached(a.cache, "2024-06-10", "2024-06-16")
    taken, _ = build_taken(raw, pd.read_parquet(f"{a.cache}/umpires.parquet"))
    games = np.random.default_rng(0).choice(taken["game_pk"].unique(), size=a.games, replace=False)
    sample = taken.loc[taken["game_pk"].isin(games), SAMPLE_COLUMNS].round(4)
    sample.to_csv("src/shadow_zone/data/sample_taken_pitches.csv", index=False)
    n_catchers, n_umpires = sample["fielder_2"].nunique(), sample["hp_umpire_id"].nunique()
    print(len(sample), "taken pitches,", n_catchers, "catchers,", n_umpires, "umpires")


if __name__ == "__main__":
    main()
