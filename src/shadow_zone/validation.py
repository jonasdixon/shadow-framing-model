"""Independent check of the catcher/umpire ranking with a Bayesian mixed model (statsmodels).

The main model is a penalised logistic regression. This fits a different estimator on shadow-zone
pitches only (variational Bayes, crossed random intercepts for catcher and umpire, a plain
parametric edge response) and asks whether it ranks catchers and umpires the same way. Agreement
means the ranking is not an artefact of the penalised fit; it does not make either one causal.
Requires the optional `validate` extra (statsmodels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mixed_model_check(df: pd.DataFrame, effects: dict, shadow_ft: float = 0.25, max_rows: int = 120_000,
                      seed: int = 0) -> dict:
    """Spearman correlation between the penalised-fit effects (`effects[block]` frames) and the
    posterior means from a Bayesian mixed model, on one season of shadow-zone pitches."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.loc[df["edge_dist"].abs() <= shadow_ft]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    d = d.reset_index(drop=True).assign(
        edge2=lambda x: x["edge_dist"] ** 2, left=(d["stand"].reset_index(drop=True) == "L").astype(int),
        catcher_s=lambda x: x["catcher"].astype(str), umpire_s=lambda x: x["umpire"].astype(str),
    )
    formula = "strike ~ edge_dist + edge2 + C(region) + C(count) + left"
    vc = {"catcher": "0 + C(catcher_s)", "umpire": "0 + C(umpire_s)"}
    model = BinomialBayesMixedGLM.from_formula(formula, vc, d)
    fit = model.fit_vb()
    names, means = model.vcp_names, fit.vc_mean
    out = {"rows": len(d), "variance_components_sd": dict(zip(names, np.exp(fit.vcp_mean), strict=True))}
    # random-effect posterior means are ordered catcher levels then umpire levels (sorted as strings)
    n_c = d["catcher_s"].nunique()
    levels = {"catcher": sorted(d["catcher_s"].unique()), "umpire": sorted(d["umpire_s"].unique())}
    post = {"catcher": means[:n_c], "umpire": means[n_c:]}
    for block in ("catcher", "umpire"):
        mine = effects[block].assign(key=lambda x: x[block].astype(str)).set_index("key")["effect_logit"]
        theirs = pd.Series(post[block], index=levels[block])
        both = pd.concat([mine, theirs], axis=1, join="inner").dropna()
        out[f"{block}_spearman"] = float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
        out[f"{block}_levels"] = len(both)
    return out
