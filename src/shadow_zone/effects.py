"""Catcher and umpire effects with uncertainty, on the logit scale and as strike-probability points.

Two standard errors are reported for every effect:
  se_laplace  the posterior standard deviation under the penalised model (inverse Hessian). This
              treats pitches as independent given the model.
  se_cluster  a sandwich standard error with each game as a cluster, because pitches in a game
              share an umpire's day and are not independent.
`se` is the larger of the two, so intervals are not narrower than the clustered estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from shadow_zone.model import FramingModel, sigmoid


def coefficient_covariances(model: FramingModel, df: pd.DataFrame, blocks=("catcher", "umpire")) -> dict:
    """Per block: (laplace_var, cluster_var) of each level's coefficient, in scaled-column units."""
    X = model.design(df)
    X = sparse.hstack([X, sparse.csr_matrix(np.ones((X.shape[0], 1)))], format="csr")
    eta = X @ np.append(model.coef, model.intercept)
    p = sigmoid(np.asarray(eta).ravel())
    w = p * (1 - p)
    penalty = np.ones(X.shape[1])
    penalty[-1] = 0.0
    H = (X.T @ sparse.diags(w) @ X).toarray() + np.diag(penalty)
    cov = np.linalg.inv(H)

    resid = df["strike"].to_numpy() - p
    game_codes, _ = pd.factorize(df["game_pk"])
    A = sparse.csr_matrix(
        (np.ones(len(df)), (game_codes, np.arange(len(df)))), shape=(game_codes.max() + 1, len(df))
    )
    G = A @ sparse.diags(resid) @ X
    S = (G.T @ G).toarray()

    out = {}
    for b in blocks:
        sl = model.slices[b]
        cb = cov[sl, :]
        out[b] = (np.diag(cov)[sl], np.einsum("ij,jk,ik->i", cb, S, cb))
    return out


def effect_table(
    model: FramingModel,
    df: pd.DataFrame,
    block: str,
    covs: dict | None = None,
    shadow_ft: float = 0.25,
) -> pd.DataFrame:
    """One row per catcher (or umpire): effect, uncertainty, workload and edge strike-probability points.

    `edge_pp` is the change in called-strike probability, in percentage points, if this catcher
    (umpire) replaced an average one on every shadow-zone pitch in `df`, other blocks as fitted.
    """
    covs = covs or coefficient_covariances(model, df, (block,))
    lap, clu = covs[block]
    scale = model.scales[block]
    tbl = model.effects(block)
    tbl["se_laplace"] = scale * np.sqrt(lap)
    tbl["se_cluster"] = scale * np.sqrt(np.maximum(clu, 0))
    tbl["se"] = tbl[["se_laplace", "se_cluster"]].max(axis=1)

    shadow = df.loc[df["edge_dist"].abs() <= shadow_ft]
    eta0 = model.linear_predictor(shadow, zero=(block,))
    base = sigmoid(eta0)
    v = tbl["effect_logit"].to_numpy()
    shifted = sigmoid(eta0[None, :] + v[:, None])
    tbl["edge_pp"] = 100 * (shifted.mean(axis=1) - base.mean())
    slope = (shifted * (1 - shifted)).mean(axis=1)
    tbl["edge_pp_se"] = 100 * slope * tbl["se"].to_numpy()

    entity = model.levels[block]
    grp = df.groupby(block)
    tbl["pitches"] = tbl[block].map(grp.size()).fillna(0).astype(int)
    tbl["games"] = tbl[block].map(grp["game_pk"].nunique()).fillna(0).astype(int)
    tbl["shadow_pitches"] = tbl[block].map(shadow.groupby(block).size()).fillna(0).astype(int)
    for q, name in ((1.96, "95"),):
        tbl[f"edge_pp_lo{name}"] = tbl["edge_pp"] - q * tbl["edge_pp_se"]
        tbl[f"edge_pp_hi{name}"] = tbl["edge_pp"] + q * tbl["edge_pp_se"]
    assert len(tbl) == len(entity)
    return tbl.sort_values("effect_logit", ascending=False).reset_index(drop=True)


def region_residuals(
    model: FramingModel, df: pd.DataFrame, block: str = "catcher", shadow_ft: float = 0.25
) -> pd.DataFrame:
    """Extra called strikes per 100 shadow-zone pitches by zone side, relative to a model that
    sets the `block` effect to average. Descriptive: shows where an effect is concentrated."""
    shadow = df.loc[df["edge_dist"].abs() <= shadow_ft].copy()
    shadow["expected"] = model.predict(shadow, zero=(block,))
    shadow["extra"] = shadow["strike"] - shadow["expected"]
    g = shadow.groupby([block, "region"])
    out = g.agg(pitches=("extra", "size"), extra_per_100=("extra", lambda s: 100 * s.mean())).reset_index()
    return out
