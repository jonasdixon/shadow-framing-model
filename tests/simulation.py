"""Simulated taken pitches with known catcher/umpire effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shadow_zone.model import sigmoid
from shadow_zone.zone import edge_distance, edge_region, zone_height_fraction


def simulate_taken(n=60_000, n_catchers=30, n_umpires=30, n_games=1_500, seed=1, sd_c=0.15, sd_u=0.2):
    rng = np.random.default_rng(seed)
    game = rng.integers(0, n_games, n)
    ump_of_game = rng.integers(0, n_umpires, n_games)
    cat_of_game_side = rng.integers(0, n_catchers, (n_games, 2))
    side = rng.integers(0, 2, n)
    catcher = cat_of_game_side[game, side]
    umpire = ump_of_game[game]
    pitcher = rng.integers(0, 120, n)
    batter = rng.integers(0, 150, n)
    true_c, true_u = rng.normal(0, sd_c, n_catchers), rng.normal(0, sd_u, n_umpires)
    true_p, true_b = rng.normal(0, 0.25, 120), rng.normal(0, 0.15, 150)

    stand = np.where(rng.random(n) < 0.55, "R", "L")
    top, bot = rng.normal(3.4, 0.15, n), rng.normal(1.6, 0.1, n)
    x = rng.normal(0, 0.75, n)
    z = rng.normal(2.5, 0.85, n)
    edge = edge_distance(x, z, top, bot)
    eta = -14.0 * edge + true_c[catcher] + true_u[umpire] + true_p[pitcher] + true_b[batter]
    strike = rng.random(n) < sigmoid(eta + 0.1)
    df = pd.DataFrame({
        "game_pk": game, "catcher": catcher, "umpire": umpire, "pitcher": pitcher, "batter": batter,
        "plate_x": x, "plate_z": z, "sz_top": top, "sz_bot": bot, "stand": stand,
        "p_throws": np.where(rng.random(n) < 0.7, "R", "L"),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
        "pitch_type": rng.choice(["FF", "SL", "CH", "SI"], n), "strike": strike.astype(int),
    })
    df["edge_dist"] = edge
    df["z_frac"] = zone_height_fraction(z, top, bot)
    df["region"] = edge_region(x, z, top, bot, stand).to_numpy()
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    return df, true_c, true_u
