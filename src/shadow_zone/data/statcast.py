"""Resumable Statcast and home-plate-umpire downloads.

Savant caps a CSV export at 25,000 rows, so the fetcher asks for one game day at a time
(a full slate is roughly 4-5k pitches). Each day is cached as its own parquet file, which makes an
interrupted download restartable and lets later runs skip days that are already on disk.
"""

from __future__ import annotations

import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests

try:  # use the OS trust store; some networks re-sign TLS with a private root CA
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Columns kept from the 118-column export. This is a superset so that the same cache can serve
# a called-strike model, a pitch-shape model and a trajectory model without re-downloading.
KEEP_COLUMNS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "pitcher", "batter", "fielder_2", "player_name",
    "stand", "p_throws", "balls", "strikes", "outs_when_up",
    "pitch_type", "pitch_name", "release_speed", "description", "type", "zone",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "release_pos_x", "release_pos_y", "release_pos_z", "release_extension",
    "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "release_spin_rate", "spin_axis", "arm_angle",
    "delta_run_exp", "events", "bb_type", "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
]


def _get(url: str, params: dict, retries: int = 5, pause: float = 2.0) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=90)
            if resp.status_code == 200:
                return resp
            last = RuntimeError(f"HTTP {resp.status_code} from {url}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_day(day: dt.date, columns: list[str] | None = None) -> pd.DataFrame:
    """One day of regular-season pitch-level Statcast, restricted to `columns`."""
    params = {
        "all": "true", "type": "details", "hfGT": "R|", "player_type": "pitcher",
        "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat(),
        "min_pitches": "0", "min_results": "0",
    }
    text = _get(SAVANT_URL, params).text
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= 25_000:
        raise RuntimeError(f"{day}: hit the 25,000-row export cap; split the request")
    cols = columns or KEEP_COLUMNS
    return df[[c for c in cols if c in df.columns]]


def fetch_range(
    start: str, end: str, cache_dir: str | Path, columns: list[str] | None = None, verbose: bool = True
) -> None:
    """Download every day in [start, end] that is not already cached. Off days write an empty marker."""
    daily = Path(cache_dir) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    day, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while day <= last:
        path = daily / f"{day.isoformat()}.parquet"
        if not path.exists():
            df = fetch_day(day, columns)
            df.to_parquet(path, index=False)
            if verbose:
                print(f"{day} {len(df):>6} pitches", flush=True)
            time.sleep(0.5)
        day += dt.timedelta(days=1)


def load_cached(cache_dir: str | Path, start: str, end: str) -> pd.DataFrame:
    """Concatenate cached daily files in [start, end]."""
    daily = Path(cache_dir) / "daily"
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    files = sorted(
        p for p in daily.glob("*.parquet") if lo <= dt.date.fromisoformat(p.stem) <= hi
    )
    frames = [pd.read_parquet(p) for p in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError(f"no cached Statcast days in {daily} for {start}..{end}")
    return pd.concat(frames, ignore_index=True)


def fetch_home_plate_umpires(start: str, end: str, cache_dir: str | Path) -> pd.DataFrame:
    """Home-plate umpire per regular-season game, from the MLB StatsAPI schedule (two-week chunks)."""
    out = Path(cache_dir) / "umpires.parquet"
    existing = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    rows = []
    cur = lo
    while cur <= hi:
        nxt = min(cur + dt.timedelta(days=13), hi)
        params = {
            "sportId": 1, "gameType": "R", "hydrate": "officials",
            "startDate": cur.isoformat(), "endDate": nxt.isoformat(),
        }
        for date in _get(SCHEDULE_URL, params).json().get("dates", []):
            for g in date["games"]:
                hp = [o["official"] for o in g.get("officials", []) if o["officialType"] == "Home Plate"]
                rows.append(
                    {
                        "game_pk": g["gamePk"],
                        "game_date": date["date"],
                        "hp_umpire_id": hp[0]["id"] if hp else pd.NA,
                        "hp_umpire_name": hp[0]["fullName"] if hp else pd.NA,
                        "status": g.get("status", {}).get("abstractGameState"),
                    }
                )
        cur = nxt + dt.timedelta(days=1)
        time.sleep(0.3)
    new = pd.DataFrame(rows)
    both = pd.concat([existing, new], ignore_index=True).drop_duplicates("game_pk", keep="last")
    both["hp_umpire_id"] = both["hp_umpire_id"].astype("Int64")
    both.to_parquet(out, index=False)
    return both


def fetch_player_names(ids, cache_dir: str | Path) -> dict:
    """Full names for MLB person ids (Statcast's `player_name` is the pitcher, so catchers need this)."""
    path = Path(cache_dir) / "player_names.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    known = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["id", "name"])
    todo = sorted(set(int(i) for i in ids) - set(known["id"].astype(int)))
    rows = []
    for i in range(0, len(todo), 100):
        chunk = todo[i : i + 100]
        data = _get("https://statsapi.mlb.com/api/v1/people", {"personIds": ",".join(map(str, chunk))}).json()
        rows += [{"id": p["id"], "name": p["fullName"]} for p in data.get("people", [])]
    both = pd.concat([known, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("id")
    both["id"] = both["id"].astype(int)
    both.to_parquet(path, index=False)
    return dict(zip(both["id"], both["name"], strict=True))
