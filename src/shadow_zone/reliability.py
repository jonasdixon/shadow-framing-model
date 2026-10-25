"""How repeatable are catcher effects? Split-half within a season and season to season."""

from __future__ import annotations

import pandas as pd

from shadow_zone.model import FramingModel


def _fit_effects(df: pd.DataFrame, scales: dict, block: str) -> pd.DataFrame:
    eff = FramingModel(scales=scales).fit(df).effects(block)
    eff["pitches"] = eff[block].map(df.groupby(block).size()).fillna(0).astype(int)
    return eff


def _compare(a: pd.DataFrame, b: pd.DataFrame, block: str, min_pitches: int) -> dict:
    m = a.merge(b, on=block, suffixes=("_a", "_b"))
    m = m[(m["pitches_a"] >= min_pitches) & (m["pitches_b"] >= min_pitches)]
    r = float(m["effect_logit_a"].corr(m["effect_logit_b"]))
    return {
        "n_players": len(m),
        "pearson": r,
        "spearman": float(m["effect_logit_a"].corr(m["effect_logit_b"], method="spearman")),
        "spearman_brown_full_season": 2 * r / (1 + r) if r > -1 else float("nan"),
    }


def split_half(df: pd.DataFrame, scales: dict, block: str = "catcher", min_pitches: int = 1000) -> dict:
    """Fit odd- and even-numbered games separately and correlate the player effects."""
    odd = _fit_effects(df.loc[df["game_pk"] % 2 == 1], scales, block)
    even = _fit_effects(df.loc[df["game_pk"] % 2 == 0], scales, block)
    return _compare(odd, even, block, min_pitches)


def season_to_season(df_a: pd.DataFrame, df_b: pd.DataFrame, scales: dict, block: str = "catcher",
                     min_pitches: int = 2000) -> dict:
    """Correlate player effects fitted on two different seasons."""
    return _compare(_fit_effects(df_a, scales, block), _fit_effects(df_b, scales, block), block, min_pitches)
