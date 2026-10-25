# Changelog

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
