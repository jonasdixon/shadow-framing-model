# Definitions

These are fixed before any model is fit. Where a definition is a choice rather than a fact, the
alternatives and the reason for the choice are in [design_decisions.md](design_decisions.md).

## Taken pitch

A pitch the batter did not swing at that the umpire called either a strike or a ball:
`description` is `called_strike`, `ball` or `blocked_ball`. Swings, hit-by-pitches, pitchouts,
intentional balls (`pitch_type` IN or PO) and automatic balls/strikes from pitch-clock violations
are excluded, because the home-plate umpire's zone was not what decided them.

## Called strike zone

The rulebook zone is 17 inches wide (half-width 8.5 in = 0.708 ft) and runs from the batter's knees to
the midpoint between shoulders and belt (Statcast's `sz_bot` and `sz_top`, per batter). A pitch is a
strike if any part of the ball touches the zone, so the zone the umpire has to decide about is the
rectangle grown by one ball radius (1.45 in = 0.121 ft) on every side. The effective half-width is
therefore 0.829 ft.

## Edge distance

The signed distance in feet from the pitch location (`plate_x`, `plate_z`, measured as the ball crosses the
front of the plate) to the boundary of that grown rectangle. Negative means inside, and the value
is how far inside; positive means outside. Near a corner the distance is Euclidean.

## Shadow zone

Taken pitches whose edge distance is within 0.25 ft (3 inches) of the boundary on either side. This is the
band where called-strike probability is neither near 0 nor near 1 and where an umpire's judgement,
and so a catcher's presentation, can change the call. The width is a parameter (`--shadow-ft`);
results with 0.5 ft are in `docs/design_decisions.md`.

## Framing effect

The change in called-strike probability attributable to which catcher was behind the plate, with
pitch location, batter, pitcher, count and home-plate umpire held fixed. It is reported
three ways: a logit offset relative to an average catcher, percentage points of called-strike
probability on shadow-zone pitches, and runs.

It is **not** a claim about what the catcher did with his glove. The data record where the pitch crossed the
plate, not where the catcher received it or how he moved his glove, so the effect is
everything a catcher contributes to the call: receiving, positioning, and anything correlated
with the catcher that the model does not see.

## Umpire effect

The same quantity for the home-plate umpire: change in called-strike probability on shadow-zone
pitches relative to an average umpire.

## Runs

Value of a strike call over a ball call in a given count, estimated from the data as the mean
`delta_run_exp` on called balls minus called strikes in that count. A catcher's runs are the
sum, over the taken pitches he caught, of the change in strike probability due to his effect
times that value. "Per 1,000 taken pitches" divides by his taken pitches; "per 150 games"
rescales to full-game equivalents using a typical taken-pitch count for a catcher's full game.
