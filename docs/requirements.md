# Requirements

Written before the build and revised as the data forced changes (noted at the end).

## Technical

- T1. Raw data comes from Baseball Savant pitch-level exports (regular season 2023 and 2024) and the
  MLB StatsAPI schedule (home-plate umpire per game). Savant caps an export at 25,000 rows, so
  the fetcher requests one day at a time and caches each day. A killed download restarts where it
  stopped.
- T2. The home-plate umpire is joined on `game_pk`. Statcast's own `umpire` column is empty, so it is not used.
  Coverage of the join is reported, and games without a home-plate official are dropped and counted.
- T3. Catcher is `fielder_2`, which is recorded per pitch (so mid-game catcher changes are handled).
- T4. Every exclusion is counted. The exclusion table is a result, not a log line.
- T5. Raw Statcast data is never committed; the cache is git-ignored. A sample of a few thousand
  taken pitches and the result tables are committed so tests and the quickstart run offline.
- T6. The whole fit (two seasons, about 1.4 million taken pitches) completes on a laptop in minutes.

## User

- U1. A reader should be able to see a catcher's effect as runs per 150 games *with an interval*, and
  see the same effect as strike-probability points at the edge.
- U2. A reader should be able to ask where an effect comes from: which side of the zone, which
  umpires, which counts.
- U3. A reader should be able to tell whether the predicted probabilities can be trusted at all:
  calibration overall and inside the shadow band on a season the model was not fit to.
- U4. Definitions and limitations are written in plain language that a coach or a front-office
  analyst could read.

## Functional

- F1. `shadow-zone fetch` downloads and caches; `prepare` builds the taken-pitch tables and the
  exclusion table; `fit` tunes, fits, evaluates, writes tables and figures.
- F2. The model predicts P(called strike) for any pitch and can set the catcher, umpire, pitcher or
  batter effect to the average, so counterfactual probabilities are available.
- F3. Unseen catchers, umpires, pitchers and batters get the average effect (a zero), so the model
  can score a new season.
- F4. Effects come with two standard errors (model-based and game-clustered); the larger is used.
- F5. Run values are re-estimated from whatever data is supplied, not hard-coded.

## Changes after looking at the data

- Plate heights below -0.5 ft are real balls in the dirt, not errors, so the plausible range for
  `plate_z` was widened to -3 to 7 ft after the first integrity run flagged 830 of them.
