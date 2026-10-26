"""Figures. Binned residual maps are descriptive diagnostics and labelled as such."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shadow_zone.zone import ZONE_HALF_WIDTH_FT  # noqa: E402


def _zone_box(ax, top=3.4, bot=1.6):
    w = ZONE_HALF_WIDTH_FT
    ax.plot([-w, w, w, -w, -w], [bot, bot, top, top, bot], color="black", lw=1)


def probability_map(model, path: str, stand: str = "R", p_throws: str = "R") -> None:
    """Strike probability over the plate for a typical 3.4 / 1.6 ft zone, neutral catcher and umpire."""
    gx, gz = np.meshgrid(np.linspace(-1.6, 1.6, 81), np.linspace(0.6, 4.4, 81))
    grid = pd.DataFrame({"plate_x": gx.ravel(), "plate_z": gz.ravel()})
    grid["sz_top"], grid["sz_bot"] = 3.4, 1.6
    from shadow_zone.zone import edge_distance, edge_region, zone_height_fraction

    grid["edge_dist"] = edge_distance(grid["plate_x"], grid["plate_z"], grid["sz_top"], grid["sz_bot"])
    grid["z_frac"] = zone_height_fraction(grid["plate_z"], grid["sz_top"], grid["sz_bot"])
    grid["region"] = edge_region(grid["plate_x"], grid["plate_z"], grid["sz_top"], grid["sz_bot"], stand).to_numpy()
    grid["stand"], grid["p_throws"] = stand, p_throws
    grid["count"] = "0-0"
    grid["pitch_type"] = "FF"
    for b in ("catcher", "umpire", "pitcher", "batter"):
        grid[b] = -1
    p = model.predict(grid, zero=("catcher", "umpire", "pitcher", "batter")).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    im = ax.contourf(gx, gz, p, levels=np.linspace(0, 1, 11), cmap="RdBu_r")
    _zone_box(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal location, catcher's view (ft)")
    ax.set_ylabel("Height (ft)")
    ax.set_title(f"P(called strike), {stand}HB, 0-0 fastball", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def residual_map(df: pd.DataFrame, p: np.ndarray, path: str, min_n: int = 40) -> None:
    """Observed minus predicted strike rate over the plate. Descriptive: 0.2 ft cells."""
    d = df.assign(res=df["strike"].to_numpy() - p)
    d = d.loc[d["plate_x"].between(-1.6, 1.6) & d["plate_z"].between(0.6, 4.4)]
    xb = np.floor(d["plate_x"] / 0.2) * 0.2
    zb = np.floor(d["plate_z"] / 0.2) * 0.2
    g = d.groupby([zb.rename("z"), xb.rename("x")])["res"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    sc = ax.scatter(g["x"] + 0.1, g["z"] + 0.1, c=100 * g["mean"], s=95, marker="s", cmap="PuOr_r", vmin=-5, vmax=5)
    _zone_box(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal location, catcher's view (ft)")
    ax.set_ylabel("Height (ft)")
    ax.set_title("Held-out residual, observed - predicted (points)\n0.2 ft cells, descriptive", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def calibration_plot(curve: pd.DataFrame, y, p, path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.plot(curve["predicted"], curve["observed"], color="#1d3f63", lw=2, label="smooth curve")
    q = pd.qcut(pd.Series(p), 20, duplicates="drop")
    pts = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
    ax.plot(pts["p"], pts["y"], "o", color="#c2410c", ms=4, label="20 quantile groups (descriptive)")
    ax.set_xlabel("Predicted P(called strike)")
    ax.set_ylabel("Observed rate")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def forest(table: pd.DataFrame, label_col: str, path: str, value="runs_per_150g", se="runs_per_150g_se",
           top: int = 12, title: str = "") -> None:
    """Top and bottom players by `value` with 95% intervals."""
    t = table.sort_values(value, ascending=False)
    t = pd.concat([t.head(top), t.tail(top)]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(t) + 1.2))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t[value], y, xerr=1.96 * t[se], fmt="o", color="#1d3f63", capsize=2, ms=4)
    ax.set_yticks(y, t[label_col], fontsize=8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(title or value)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def feature_distributions(df: pd.DataFrame, path: str, shadow_ft: float = 0.25) -> None:
    """Pre-model look at the derived features: what they look like and how they track the outcome.

    The lower-left panel is the one that matters. If `zone.edge_distance` described the zone umpires
    actually call, every curve would cross 50% at zero.
    """
    from shadow_zone import eda

    d = eda.add_derived(df)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))

    ax = axes[0, 0]
    shown = d["edge_dist"].between(-1.0, 1.5)
    ax.hist(d.loc[shown, "edge_dist"], bins=90, color="#1d3f63")
    ax.axvspan(-shadow_ft, shadow_ft, color="#c2410c", alpha=0.18, label=f"shadow zone (+/- {shadow_ft} ft)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Signed distance to zone edge (ft); negative is inside\n"
                  f"{100 * (1 - shown.mean()):.1f}% fall outside this range and are not drawn")
    ax.set_ylabel("Taken pitches")
    ax.set_title("Where taken pitches sit", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(d["sz_top"], bins=60, color="#1d3f63", alpha=0.85, label="sz_top")
    ax.hist(d["sz_bot"], bins=60, color="#c2410c", alpha=0.85, label="sz_bot")
    ax.set_xlabel("Height (ft)")
    ax.set_ylabel("Taken pitches")
    ax.set_title("Statcast's per-batter zone bounds", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    colors = {"high": "#c2410c", "low": "#1d3f63", "inside": "#2f7d32", "outside": "#7b3fa0"}
    for region, part in d.groupby("region", observed=True):
        g = eda.response_curve(part)
        if len(g):
            ax.plot(g["edge_dist_bin"], g["strike_rate"], lw=1.6, color=colors.get(str(region)), label=str(region))
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Signed distance to zone edge (ft)")
    ax.set_ylabel("Called-strike rate (raw)")
    ax.set_title("Raw strike rate by side of the zone\ncurves would cross 50% at 0 if the geometry matched", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    t = eda.by_region(d)
    t = t[t["split"] == "region"].sort_values("half_strike_in")
    y = np.arange(len(t))
    ax.barh(y, t["half_strike_in"], color=[colors.get(g, "#555") for g in t["group"]])
    ax.set_yticks(y, t["group"], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Distance where umpires call 50% strikes (inches)")
    ax.set_title("Negative: called zone is tighter than geometry", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def edge_curves(df: pd.DataFrame, p: np.ndarray, path: str) -> None:
    """Observed and predicted strike rate against signed edge distance, in 0.05 ft steps (descriptive)."""
    d = df.assign(p=p, bin=np.round(df["edge_dist"] / 0.05) * 0.05)
    d = d.loc[d["bin"].between(-0.6, 0.8)]
    g = d.groupby("bin")[["strike", "p"]].mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(g.index, g["strike"], "o", ms=3.5, color="#c2410c", label="observed (0.05 ft steps)")
    ax.plot(g.index, g["p"], color="#1d3f63", label="model")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Distance outside the zone edge (ft); negative is inside")
    ax.set_ylabel("Called-strike rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
