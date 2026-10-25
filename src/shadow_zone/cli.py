"""Command line: shadow-zone fetch | prepare | fit."""

from __future__ import annotations

import argparse

import pandas as pd

from shadow_zone import pipeline
from shadow_zone.data import statcast


def _cmd_fetch(a):
    for y in a.years:
        lo, hi = pipeline.SEASON_DATES[y]
        statcast.fetch_range(lo, hi, a.cache)
        statcast.fetch_home_plate_umpires(lo, hi, a.cache)


def _cmd_prepare(a):
    taken = pipeline.prepare_seasons(a.cache, a.years, a.out)
    print(f"{len(taken):,} taken pitches written to {a.out}")


def _cmd_fit(a):
    import shutil
    from pathlib import Path

    taken = pd.concat([pd.read_parquet(f"{a.data}/taken_{y}.parquet") for y in a.years], ignore_index=True)
    names = statcast.fetch_player_names(taken["catcher"].unique(), a.cache)
    summary = pipeline.run_all(
        taken, a.out, names, train_year=a.years[0], test_year=a.years[-1], tune=not a.no_tune,
        shadow_ft=a.shadow_ft,
    )
    for fname in ("exclusions.csv", "integrity_checks.json"):
        shutil.copy(Path(a.data) / fname, Path(a.out) / "tables" / fname)
    print(pd.json_normalize(summary).T.to_string())


def _cmd_validate(a):
    import json

    from shadow_zone import validation
    from shadow_zone.model import FramingModel

    scales = json.loads(open(f"{a.out}/tables/scales.json").read())
    d = pd.read_parquet(f"{a.data}/taken_{a.years[-1]}.parquet")
    m = FramingModel(scales=scales).fit(d)
    res = validation.mixed_model_check(d, {"catcher": m.effects("catcher"), "umpire": m.effects("umpire")})
    with open(f"{a.out}/tables/validation_mixed_model.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(res)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="shadow-zone", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", _cmd_fetch), ("prepare", _cmd_prepare), ("fit", _cmd_fit), ("validate", _cmd_validate)):
        s = sub.add_parser(name)
        s.add_argument("--cache", default="data/cache")
        s.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
        if name != "fetch":
            s.add_argument("--data", default="data/processed")
        if name == "prepare":
            s.add_argument("--out", default="data/processed")
        if name in ("fit", "validate"):
            s.add_argument("--out", default="reports")
        if name == "fit":
            s.add_argument("--no-tune", action="store_true")
            s.add_argument("--shadow-ft", type=float, default=0.25)
        s.set_defaults(func=fn)
    a = p.parse_args(argv)
    a.func(a)
    return 0
