# Design decisions

Each section says what was chosen, what was considered and dropped, and what is still weak. These
are written so that someone can push on them.

## One joint model for location, catcher, umpire, pitcher and batter

**Chosen.** A single logistic regression with a flexible location surface and four blocks of
shrunken random intercepts, fit together.

**Why.** Pitch location explains most of the call, but the remainder is shared between the catcher,
the umpire, the pitcher (command, how the pitch moves late) and the batter (zone height, stance).
Fitting the blocks separately would give one block credit for another's work. Fitting jointly lets
the data use the fact that catchers appear with many umpires and pitchers.

**Rejected.**
- Catcher effect as raw observed-minus-expected strikes against a location-only baseline. It credits
  the catcher with the umpire's zone and with the pitchers he happened to catch.
- Separate models per catcher. Far too noisy.
- A full generalised linear mixed model with game-level effects. The crossed structure with
  roughly 2,400 games per season and 2,000 other levels is slow in standard software, and
  game effects are handled instead with clustered standard errors.

**Weakness.** Shrinkage is by one prior scale per block, chosen by cross-validation. The search
landed on 0.2 for all four blocks (logit scale). Held-out log loss was flat across 0.1 to 0.4
(it moved by under 0.0001 for the catcher block; `reports/tables/scale_tuning.csv`), so the ranking
of catchers is not sensitive to it, but the *size* of the extremes is. A tighter prior pulls every
catcher toward zero.

## Penalised logistic regression, not a GAM library

**Chosen.** Splines (cubic B-splines on a tensor grid, plus a spline in signed edge distance
interacted with the side of the zone and batter handedness) and a ridge penalty, implemented with
scikit-learn.

**Why.** The edge response is sharp, with strike probability going from about 0.9 to 0.1 across a third
of a foot (just inside to just outside the edge) and different on each side of the zone. The knot positions
for the edge distance are concentrated within 0.3 ft of the boundary. A ridge on spline coefficients is a
standard way to get a smooth penalised GAM, and the same machinery handles the random intercepts.

**Not tried.** A dedicated GAM package. Crossed random effects with thousands of levels sit awkwardly
in most of them, and one extra dependency for the part that is not the interesting part.

**Weakness.** The penalty is on coefficients, not on wiggliness, so the smoothness is controlled by
knot placement more than by a smoothing parameter. The residual map (`reports/figures/residual_map.png`)
shows that the fit is not perfect at the edge. Scoring 2024 pitches, the model over-predicts called
strikes just outside the top and bottom edges by up to about 4 points (observed minus predicted, in the
band 0 to 0.1 ft outside), and under-predicts them on the outside (away from the batter) edge by about
2 points. The second looks like the called zone moving between seasons: a model fit on one half of 2024
and scored on the other half shows no outside-edge gap (+0.1 points) but keeps a smaller version of the
vertical gap (about -1 point on average, and -3.8 just outside the bottom edge). Averaged over
the whole shadow zone the residual is within half a point.

## Edge distance as a feature

**Chosen.** Signed distance to the zone grown by one ball radius (0.121 ft), using each batter's
`sz_top`/`sz_bot`.

**Why.** Absolute plate height means different things for a 5'6" and a 6'6" batter. The distance
measure removes that. The rulebook says a strike is any part of the ball over the zone, so the
zone the umpire judges is the larger rectangle.

**Weakness.** Umpires do not call the rulebook zone. The model does not assume they do; the surface
and the edge response are learned. The geometric edge is only a coordinate system.

## Shadow-zone width

**Chosen.** 0.25 ft (3 inches) either side of the edge for reporting strike-probability points.

**Sensitivity.** The model and the logit effects do not depend on the band. The percentage-point
effect does. With a 0.5 ft band the ranking of catchers by points is identical (Spearman 1.0) and
the spread is 0.61 times as large, because a wider band adds pitches where the call is nearly
certain either way, which dilutes the average difference. Runs per 150 games are computed over all
taken pitches and do not depend on the band.

## Runs from `delta_run_exp`

**Chosen.** The value of a strike over a ball in each count is the mean `delta_run_exp` on called
balls minus called strikes in that count (12 counts, 0.08 to 0.62 runs).

**Rejected.** A single 0.125-run constant (the common shortcut). It undervalues
strikes at 3-2 and overvalues them at 0-0, so catchers who frame more in high-leverage counts
would be under-credited.

**Weakness.** These are league averages across base-out states. A strike in a bases-loaded
two-out 3-2 count is worth more than the count average and a strike in a blowout less. The
`delta_run_exp` value also bakes in what happens after a called ball or strike on that pitch only; it ignores any change
in how the pitcher or batter behaves afterwards.

## Uncertainty

**Chosen.** The larger of (a) the posterior standard deviation from the inverse Hessian of the
penalised likelihood and (b) a sandwich standard error clustered by game.

**Why.** (a) respects shrinkage but treats pitches as independent, and (b) respects the
within-game dependence of pitches but ignores the prior. Using the larger one is the conservative
choice. For 2024 catchers (a) was larger on average (0.085 against 0.071).

**Weakness.** Neither includes uncertainty in the prior scales, and the run intervals propagate only
the effect's own standard error. Treat a run interval as a rough guide.

## Time

**Chosen.** Each season is fit separately for its leaderboard, and 2023 predicts 2024 for evaluation.
No catcher effect varies within a season.

**Weakness.** Framing skill can change within a season (injury, adjustments, a new pitching staff).
Called-strike probability also has a time structure that is not modelled: pitches within a game,
fatigue, and the umpire's own zone drifting over a game are folded into the clustered
errors, not the model.

## Identification

A catcher's effect is identified from catchers who share umpires and pitchers with other
catchers. This works when catchers move between umpires and pitchers enough to connect
them. Umpires rotate across teams, so each catcher works with many umpires, but a catcher's
pitching staff is a fixed set for the season. A catcher who always catches
a pitcher with extreme command is partly credited or blamed for that pitcher, up to what the
pitcher effect absorbs. The connectedness of the catcher-pitcher graph was not measured.

## What the model cannot know

- Where the catcher received the ball and how he moved the glove. The measured location is at the
  front of the plate.
- Pitch intent. A pitcher missing a target by a foot and one hitting it may produce the same
  location and different umpire expectations.
- Whether the umpire sees the pitch the way the tracking system does. Umpire effects here are
  deviations from average calls, not errors against an objective truth.
- Anything about base runners, game state or umpire fatigue beyond count.

## Checks that would change the conclusions

- A catcher whose effect flips sign between 2023 and 2024 despite many pitches in both. The
  season-to-season correlation of 0.55 says this happens for some catchers; it is not shown for
  every one.
- Residual structure in the strike-probability surface at particular spots. It exists (see
  above): a few points just outside the top and bottom edges. A model with the zone's top and bottom
  response varying by batter height would be the next thing to try. It would change probabilities at
  those edges and is unlikely to change how catchers rank.
- A year-over-year drift in the called zone. The 2023 model under-predicts 2024 strikes on the
  outside edge by about 2 points, which is why the leaderboards refit each season.
