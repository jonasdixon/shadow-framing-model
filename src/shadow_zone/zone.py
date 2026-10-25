"""Where a pitch sits relative to the called strike zone.

The rulebook zone is a rectangle over the plate: 17 inches wide, bounded vertically by the batter's
knees and the midpoint between shoulders and belt. A pitch is a strike if any part of the ball
touches it, so the zone the umpire actually calls is the rectangle grown by one ball radius
(about 1.45 in = 0.121 ft) on every side. Half the plate is 8.5 in = 0.708 ft, so the effective
half-width is 0.708 + 0.121 = 0.829 ft. `sz_top` and `sz_bot` are Statcast's per-batter bounds.

`edge_distance` is the signed distance in feet from the pitch to the boundary of that rectangle:
negative inside (how deep), positive outside (how far), measured to the nearest edge or corner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLATE_HALF_WIDTH_FT = 8.5 / 12.0
BALL_RADIUS_FT = 1.45 / 12.0
ZONE_HALF_WIDTH_FT = PLATE_HALF_WIDTH_FT + BALL_RADIUS_FT
DEFAULT_SHADOW_FT = 0.25  # about 3 inches either side of the edge


def edge_distance(plate_x, plate_z, sz_top, sz_bot) -> np.ndarray:
    """Signed distance (ft) to the called-strike-zone boundary; positive means outside the zone."""
    x, z = np.asarray(plate_x, float), np.asarray(plate_z, float)
    top, bot = np.asarray(sz_top, float) + BALL_RADIUS_FT, np.asarray(sz_bot, float) - BALL_RADIUS_FT
    dx = np.abs(x) - ZONE_HALF_WIDTH_FT
    dz = np.maximum(z - top, bot - z)
    inside = (dx <= 0) & (dz <= 0)
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dz, 0.0))
    return np.where(inside, np.maximum(dx, dz), outside)


def zone_height_fraction(plate_z, sz_top, sz_bot) -> np.ndarray:
    """Height as a fraction of this batter's zone: 0 at the bottom edge, 1 at the top edge."""
    plate_z, sz_top, sz_bot = (np.asarray(a, float) for a in (plate_z, sz_top, sz_bot))
    return (plate_z - sz_bot) / (sz_top - sz_bot)


def edge_region(plate_x, plate_z, sz_top, sz_bot, stand) -> pd.Series:
    """Which side of the zone a pitch is nearest: high, low, inside (toward the batter) or outside.

    Catcher's view: a right-handed batter stands on the negative-x side, so "inside" is negative x
    for right-handed batters and positive x for left-handed batters.
    """
    x, z = np.asarray(plate_x, float), np.asarray(plate_z, float)
    top, bot = np.asarray(sz_top, float) + BALL_RADIUS_FT, np.asarray(sz_bot, float) - BALL_RADIUS_FT
    d_side = np.abs(x) - ZONE_HALF_WIDTH_FT
    d_top, d_bot = z - top, bot - z
    d_vert = np.maximum(d_top, d_bot)
    horizontal = d_side >= d_vert
    toward_batter = np.where(np.asarray(stand) == "R", x < 0, x > 0)
    return pd.Series(
        np.where(
            horizontal,
            np.where(toward_batter, "inside", "outside"),
            np.where(d_top >= d_bot, "high", "low"),
        )
    )
