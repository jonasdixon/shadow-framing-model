# Data quality

What the integrity checks found in the 2023 and 2024 regular-season pulls, and what was done about
each finding. Numbers come from `reports/tables/exclusions.csv` and `integrity_checks.json`
written by `shadow-zone prepare`.

## Where every pitch went

| | 2023 | 2024 |
|---|---|---|
| Pitches in the export | 720,684 | 711,899 |
| Not a taken pitch (swings, hit by pitch, automatic calls, pitchouts) | 346,161 | 344,882 |
| Intentional ball or pitchout pitch type | 0 | 0 |
| Duplicate (game, at-bat, pitch) key | 0 | 0 |
| Missing plate location | 138 | 144 |
| Impossible location or zone bounds | 66 | 48 |
| Missing zone bounds | 0 | 0 |
| Missing catcher | 0 | 0 |
| No home-plate umpire for the game | 0 | 0 |
| **Taken pitches modelled** | **374,319** | **366,825** |

All 4,859 regular-season games in the two seasons had a home-plate official in the StatsAPI schedule,
so the join to `game_pk` lost nothing. The missing and impossible rows are about 0.05% of taken
pitches.

## Findings

**The first plausibility range for plate height was wrong.** An early run flagged 830 pitches in two
months of 2023 as impossible because `plate_z` was below -0.5 ft. They are balls in the dirt; the
tracked path continues below ground level when extrapolated to the front of the plate. The range was
widened to -3 to 7 ft. Pitches this low are near-certain balls and carry no information about the
catcher, but dropping them would have been the wrong reason to drop them.

**`fielder_2` changes within games.** In 654 team-games in 2023 and 587 in 2024 one side used more than one
catcher. The catcher id is recorded per pitch, so a game is not tied to a single catcher and each
pitch is attributed to whoever was catching it.

**Catcher names are not in the export.** Savant's `player_name` is the pitcher. Catcher names
come from the StatsAPI people endpoint, keyed by id.

**Statcast's own `umpire` column is empty.** It is null in every row, which is why the StatsAPI
join is needed at all.

**Sign convention.** The mean `plate_x` against right-handed batters minus left-handed batters is
+0.53 ft in 2023 and +0.59 ft in 2024 (catcher's view, pitchers work away from the batter). A
negative value would mean one handedness is flipped. The check is in `integrity.handedness_sign_check` and the
tests plant a flipped side to confirm that it triggers.

**Automatic balls and strikes.** None of the 2023-24 `ball`/`called_strike` descriptions carry an intentional-ball
pitch type (intentional walks are no longer thrown), but automatic balls and strikes from pitch-clock
violations exist as separate `automatic_*` descriptions and are excluded by construction.

**The called zone is not the computed zone, and the error differs by side.** Reading the raw
called-strike rate against `edge_dist` and finding where it crosses 50% gives +0.01 in (2023) and
+0.02 in (2024) over all taken pitches, so the ball-radius adjustment in `zone.py` is right on
average. By side of the zone it is not: high -0.72 / -0.78 in, outside +0.59 / +0.67 in, low and
inside within 0.15 in of zero. The aggregate passes only because the high and outside errors cancel.
Umpires call the top tighter and the outside edge wider than the geometry says. The model absorbs
this by interacting its edge spline with side of zone and batter handedness; the check is in
`shadow_zone.eda` and the table in `reports/tables/eda_edge_geometry.csv`. The tests move the
boundary by a known amount and confirm the check recovers it.

## Coverage

| | 2023 | 2024 |
|---|---|---|
| Games | 2,430 | 2,429 |
| Catchers (any taken pitch / 1,000+) | 102 / 77 | 100 / 77 |
| Umpires | 94 | 90 |
| Pitchers | 863 | 854 |
| Batters | 655 | 650 |
| Overall called-strike rate | 31.5% | 31.7% |
| Shadow-zone share of taken pitches | 32.0% | 32.2% |
| Shadow-zone called-strike rate | 47.9% | 48.1% |

Catchers with fewer than 1,000 taken pitches stay in the model, where their effect is shrunk hard
toward zero, but are left out of the published tables because their intervals are too wide to rank on.

## What is not checked

- **Tracking accuracy at the edge.** Location error of a fraction of an inch matters most exactly where
  this model works. There is no second tracking source here to compare against.
- **Batter zone bounds.** `sz_top` and `sz_bot` are Statcast's per-pitch estimates, not measurements
  of that batter's knees and belt. The 50% crossing above shows `sz_top` runs about three quarters of
  an inch above the top of the zone umpires call, but that cannot separate a biased estimate from
  umpires genuinely calling the top of the zone small. Either way the model does not have to assume
  the bound is right.
- **Challenge and ABS rules.** Neither was in use in MLB regular-season games in 2023-24. A
  later season would need the call-source field.
