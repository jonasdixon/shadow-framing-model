# Contributing

- Install: `pip install -e '.[dev]'`, then `pre-commit install`.
- Lint: `ruff check src tests scripts`. Test: `pytest`.
- Branches: `feature/<short-name>` or `fix/<short-name>`; open a pull request and keep CI green.
- Raw Statcast data and umpire assignments are never committed. `data/` is git-ignored; the package
  ships a small sample and `scripts/make_sample.py` shows how it was made.
- A change to a definition (the shadow zone, a framing effect, the run value of a strike) needs a
  test that fails on the old behaviour and an entry in `docs/definitions.md`.
- A change to the model or to a derived feature needs the matching check to still pass on real data,
  not only on the sample: `shadow-zone fit` writes `reports/tables/summary.json`, and the calibration
  slope and the 50% crossing in `reports/tables/eda_edge_geometry.csv` are the two numbers to read
  before and after.
- New checks belong in `shadow_zone.eda` or `shadow_zone.data.integrity` with a test that plants the
  fault they are meant to catch. The existing tests plant a flipped handedness sign and a displaced
  zone boundary; follow that pattern rather than asserting on a value the current data happens to
  produce.

Layout: `src/shadow_zone/` (package), `tests/`, `docs/`, `notebooks/`, `reports/` (tables and figures
from the runs described in the README), `scripts/`.
