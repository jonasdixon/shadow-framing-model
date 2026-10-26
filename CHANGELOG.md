# Changelog

## [Unreleased]

- Exploratory checks on the derived features (`shadow_zone.eda`), run before the model as part of
  `shadow-zone fit`: distributions and missingness for every derived feature, the raw called-strike
  rate against edge distance, Statcast's zone bounds, and the distance at which umpires call a
  strike half the time.
- The 50% crossing is a model-free test of the zone geometry. It sits within 0.02 in of the computed
  boundary overall, but the top of the zone is called 0.72-0.78 in tighter and the outside edge
  0.59-0.67 in wider, in both seasons. Written to `reports/tables/eda_edge_geometry.csv` and
  `reports/figures/feature_distributions.png`, and recorded in `summary.json`.
- Tests move the strike boundary by a known amount and confirm the check recovers it.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- Fixed: `python -m shadow_zone.cli` exited silently without running anything, because the module
  had no `__main__` guard. Only the installed `shadow-zone` script worked.
- Fixed: `fetch_player_names` failed when the cache directory did not exist, so `fit` could not be
  run from a clean checkout without running `fetch` first.

## [0.1.0]

First release.

- Resumable Statcast and home-plate-umpire download; integrity rules with counted exclusions.
- Called-strike model: spline location surface and edge response, with crossed catcher, umpire,
  pitcher and batter effects fit jointly and shrunk by cross-validated prior scales.
- Effects with model-based and game-clustered standard errors, in logit units, shadow-zone
  strike-probability points and runs (count-specific run values from `delta_run_exp`).
- Smooth calibration, residual and effect plots; split-half and season-to-season reliability;
  independent Bayesian mixed-model check of the ranking.
- Tests include recovery of planted catcher and umpire effects and planted data bugs.
