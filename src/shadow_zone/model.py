"""Called-strike model: a location surface plus crossed catcher, umpire, pitcher and batter effects.

Everything is one logistic regression, fit with an L2 penalty. Each block of columns is multiplied
by a scale s; with unit penalty that is a Gaussian prior with standard deviation s on the logit
scale for that block's coefficients. Fixed blocks (location surface, edge response, count) get a
large s so they are effectively unpenalised. The catcher, umpire, pitcher and batter blocks get small
s, so each level is a random intercept shrunk toward zero (an "average" catcher or umpire). The
scales are chosen by game-grouped cross-validation (`tune_scales`), not assumed.

Why one joint model: catchers frame to many umpires and pitchers, and umpires see many catchers.
Fitting everything together is what lets the data separate a catcher's effect from the strike
zone of the umpire he happened to work with and from the quality of the pitchers he caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

RANDOM_BLOCKS = ("catcher", "umpire", "pitcher", "batter")
DEFAULT_SCALES = {"catcher": 0.15, "umpire": 0.15, "pitcher": 0.3, "batter": 0.2}
FIXED_SCALES = {"edge": 4.0, "surface": 1.5, "count": 4.0, "platoon": 4.0, "pitch_type": 2.0}
EDGE_KNOTS = np.array([-0.8, -0.5, -0.3, -0.2, -0.12, -0.06, 0.0, 0.06, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2])
REGION_STAND = [f"{r}_{s}" for r in ("high", "low", "inside", "outside") for s in ("L", "R")]
ENTITY_COL = {"catcher": "catcher", "umpire": "umpire", "pitcher": "pitcher", "batter": "batter"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class FramingModel:
    scales: dict = field(default_factory=lambda: dict(DEFAULT_SCALES))
    max_iter: int = 3000

    # ----- encoders -----
    def _fit_encoders(self, df: pd.DataFrame) -> None:
        self.edge_tf = SplineTransformer(
            degree=3, knots=EDGE_KNOTS.reshape(-1, 1), extrapolation="constant", include_bias=True
        ).fit(np.linspace(-0.8, 1.2, 50).reshape(-1, 1))
        self.sx = SplineTransformer(n_knots=6, degree=3, extrapolation="constant").fit(
            np.linspace(-2, 2, 50).reshape(-1, 1)
        )
        self.sz = SplineTransformer(n_knots=6, degree=3, extrapolation="constant").fit(
            np.linspace(-0.6, 1.6, 50).reshape(-1, 1)
        )
        self.pitch_types = list(df["pitch_type"].value_counts().index[:8])
        self.counts = sorted(df["count"].unique())
        self.levels = {b: pd.unique(df[ENTITY_COL[b]]) for b in RANDOM_BLOCKS}
        self.index = {b: {v: i for i, v in enumerate(lv)} for b, lv in self.levels.items()}

    @staticmethod
    def _onehot(values: pd.Series, levels: list) -> np.ndarray:
        return np.column_stack([(values == lv).to_numpy(float) for lv in levels])

    def design(self, df: pd.DataFrame) -> sparse.csr_matrix:
        n = len(df)
        blocks, self.slices, start = [], {}, 0

        def add(name, mat, scale):
            nonlocal start
            m = mat if sparse.issparse(mat) else sparse.csr_matrix(mat)
            blocks.append(m * scale)
            self.slices[name] = slice(start, start + m.shape[1])
            start += m.shape[1]

        stand = df["stand"].to_numpy()
        edge = self.edge_tf.transform(np.clip(df["edge_dist"].to_numpy(float), -0.8, 1.2).reshape(-1, 1))
        rs = self._onehot(df["region"] + "_" + df["stand"], REGION_STAND)
        edge_by_side = (edge[:, :, None] * rs[:, None, :]).reshape(n, -1)
        add("edge", edge_by_side, FIXED_SCALES["edge"])

        bx = self.sx.transform(np.clip(df["plate_x"].to_numpy(float), -2, 2).reshape(-1, 1))
        bz = self.sz.transform(np.clip(df["z_frac"].to_numpy(float), -0.6, 1.6).reshape(-1, 1))
        tensor = (bx[:, :, None] * bz[:, None, :]).reshape(n, -1)
        side = np.column_stack([(stand == "L").astype(float), (stand == "R").astype(float)])
        add("surface", (tensor[:, :, None] * side[:, None, :]).reshape(n, -1), FIXED_SCALES["surface"])

        add("count", self._onehot(df["count"], self.counts), FIXED_SCALES["count"])
        platoon = df["stand"] + df["p_throws"]
        add("platoon", self._onehot(platoon, ["LL", "LR", "RL", "RR"]), FIXED_SCALES["platoon"])
        add("pitch_type", self._onehot(df["pitch_type"], self.pitch_types), FIXED_SCALES["pitch_type"])

        for b in RANDOM_BLOCKS:
            idx = df[ENTITY_COL[b]].map(self.index[b])
            ok = idx.notna().to_numpy()
            mat = sparse.csr_matrix(
                (np.ones(ok.sum()), (np.flatnonzero(ok), idx[ok].astype(int))),
                shape=(n, len(self.levels[b])),
            )
            add(b, mat, self.scales[b])
        return sparse.hstack(blocks, format="csr")

    # ----- fit / predict -----
    def fit(self, df: pd.DataFrame) -> FramingModel:
        self._fit_encoders(df)
        X = self.design(df)
        self.clf = LogisticRegression(C=1.0, max_iter=self.max_iter, tol=1e-6).fit(X, df["strike"].to_numpy())
        self.coef = self.clf.coef_[0]
        self.intercept = float(self.clf.intercept_[0])
        return self

    def linear_predictor(self, df: pd.DataFrame, zero: tuple[str, ...] = ()) -> np.ndarray:
        """Logit of P(called strike). Blocks named in `zero` (e.g. "catcher") are set to the average."""
        X = self.design(df)
        eta = X @ self.coef + self.intercept
        for b in zero:
            sl = self.slices[b]
            eta = eta - X[:, sl] @ self.coef[sl]
        return np.asarray(eta).ravel()

    def predict(self, df: pd.DataFrame, zero: tuple[str, ...] = ()) -> np.ndarray:
        return sigmoid(self.linear_predictor(df, zero))

    def effects(self, block: str) -> pd.DataFrame:
        """Fitted effect of every level of `block`, in logit units (positive = more called strikes)."""
        sl = self.slices[block]
        vals = self.coef[sl] * self.scales[block]
        return pd.DataFrame({block: self.levels[block], "effect_logit": vals})


def tune_scales(
    df: pd.DataFrame,
    grid=(0.05, 0.1, 0.2, 0.4, 0.8),
    n_folds: int = 2,
    max_rows: int = 250_000,
    seed: int = 0,
    start: dict | None = None,
    sweeps: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Coordinate search over the prior scale of each random block, scored by held-out log loss.

    Folds are whole games. Returns the chosen scales and the full search log.
    """
    rng = np.random.default_rng(seed)
    games = df["game_pk"].unique()
    if len(df) > max_rows:
        keep_games = rng.choice(games, size=int(len(games) * max_rows / len(df)), replace=False)
        df = df.loc[df["game_pk"].isin(keep_games)].reset_index(drop=True)
    folds = list(GroupKFold(n_splits=n_folds).split(df, groups=df["game_pk"]))

    def cv_loss(scales):
        total, n = 0.0, 0
        for tr, te in folds:
            m = FramingModel(scales=scales).fit(df.iloc[tr])
            p = np.clip(m.predict(df.iloc[te]), 1e-6, 1 - 1e-6)
            y = df["strike"].to_numpy()[te]
            total += -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            n += len(te)
        return total / n

    best = dict(start or DEFAULT_SCALES)
    best_loss, log = cv_loss(best), []
    for _ in range(sweeps):
        for block in RANDOM_BLOCKS:
            for s in grid:
                trial = {**best, block: s}
                loss = cv_loss(trial)
                log.append({"block": block, "scale": s, "cv_log_loss": loss})
                if loss < best_loss:
                    best, best_loss = trial, loss
    return best, pd.DataFrame(log)
