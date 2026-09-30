"""Data loading and caching from nflverse via nflreadpy."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import nflreadpy as nfl

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# PLAY-BY-PLAY DATA
# ---------------------------------------------------------------------

PBP_COLS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",

    # EPA / success
    "epa",
    "success",
    "wp",

    # Play type
    "pass",
    "rush",

    # QB / special plays
    "qb_kneel",
    "qb_spike",

    # Turnovers
    "interception",
    "fumble_lost",

    # Passing / rushing detail
    "qb_dropback",
    "sack",

    # Yardage
    "yards_gained",
]


def current_season() -> int:
    return int(nfl.get_current_season())


# ---------------------------------------------------------------------
# SCHEDULE
# ---------------------------------------------------------------------

def load_schedule(seasons) -> pd.DataFrame:
    """Load NFL schedules and create a played-game indicator."""
    s = nfl.load_schedules(list(seasons)).to_pandas()

    s["gameday"] = pd.to_datetime(s["gameday"])

    s["played"] = (
        s["home_score"].notna()
        & s["away_score"].notna()
    )

    return (
        s.sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# PLAY-BY-PLAY TEAM AGGREGATES
# ---------------------------------------------------------------------

def _aggregate_pbp(season: int) -> pd.DataFrame:
    """
    Build one row per team/game.

    All statistics are calculated from individual plays. These values are
    later shifted by features.py so that a team's current game never
    contributes to its own pre-game rating.
    """

    pl = nfl.load_pbp([season])

    cols = [c for c in PBP_COLS if c in pl.columns]

    p = pl.select(cols).to_pandas()

    # ---------------------------------------------------------------
    # Make optional columns available even if a particular nflverse
    # release doesn't contain them.
    # ---------------------------------------------------------------

    defaults = {
        "pass": 0,
        "rush": 0,
        "success": np_nan_series(len(p)),
        "wp": np_nan_series(len(p)),
        "qb_kneel": 0,
        "qb_spike": 0,
        "interception": 0,
        "fumble_lost": 0,
        "qb_dropback": 0,
        "sack": 0,
        "yards_gained": np_nan_series(len(p)),
    }

    for col, default in defaults.items():
        if col not in p.columns:
            p[col] = default

    # ---------------------------------------------------------------
    # Only normal offensive plays with EPA.
    # ---------------------------------------------------------------

    p = p[
        p["epa"].notna()
        & ((p["pass"] == 1) | (p["rush"] == 1))
        & p["posteam"].notna()
    ].copy()

    # Remove kneels and spikes.
    p = p[p["qb_kneel"] != 1]
    p = p[p["qb_spike"] != 1]

    # Remove extreme win-probability garbage time.
    if "wp" in p.columns:
        p = p[
            p["wp"].isna()
            | p["wp"].between(0.05, 0.95)
        ]

    # ---------------------------------------------------------------
    # Play-level derived variables.
    # ---------------------------------------------------------------

    p["epa_pass"] = p["epa"].where(p["pass"] == 1)
    p["epa_rush"] = p["epa"].where(p["rush"] == 1)

    p["pass_attempt"] = (p["pass"] == 1).astype(int)
    p["rush_attempt"] = (p["rush"] == 1).astype(int)

    p["successful_play"] = (
        p["success"].fillna(0)
    )

    p["explosive"] = (
        p["yards_gained"].fillna(0) >= 20
    ).astype(int)

    p["turnover"] = (
        (p["interception"].fillna(0) == 1)
        | (p["fumble_lost"].fillna(0) == 1)
    ).astype(int)

    p["sack"] = p["sack"].fillna(0).astype(int)

    # ---------------------------------------------------------------
    # Aggregate offensive statistics.
    # ---------------------------------------------------------------

    agg = (
        p.groupby(["game_id", "posteam", "defteam"])
        .agg(
            epa=("epa", "mean"),
            epa_pass=("epa_pass", "mean"),
            epa_rush=("epa_rush", "mean"),

            sr=("success", "mean"),

            plays=("epa", "size"),

            pass_attempts=("pass_attempt", "sum"),
            rush_attempts=("rush_attempt", "sum"),

            explosive_plays=("explosive", "sum"),

            turnovers=("turnover", "sum"),
            sacks=("sack", "sum"),

            yards=("yards_gained", "sum"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Offensive statistics.
    # ---------------------------------------------------------------

    off = agg.rename(
        columns={
            "posteam": "team",
            "defteam": "opp",

            "epa": "off_epa",
            "epa_pass": "off_epa_pass",
            "epa_rush": "off_epa_rush",
            "sr": "off_sr",

            "plays": "plays",

            "pass_attempts": "pass_attempts",
            "rush_attempts": "rush_attempts",

            "explosive_plays": "explosive_plays",

            "turnovers": "turnovers",
            "sacks": "sacks",

            "yards": "yards",
        }
    )

    # ---------------------------------------------------------------
    # Defensive statistics.
    #
    # The EPA belongs to the offense, so the defense receives the same
    # play-level EPA from the opposing offense. The feature construction
    # later uses opponent adjustment to interpret this correctly.
    # ---------------------------------------------------------------

    dfn = agg.rename(
        columns={
            "defteam": "team",
            "posteam": "opp",

            "epa": "def_epa",
            "epa_pass": "def_epa_pass",
            "epa_rush": "def_epa_rush",
            "sr": "def_sr",

            "plays": "def_plays",

            "pass_attempts": "def_pass_attempts",
            "rush_attempts": "def_rush_attempts",

            "explosive_plays": "def_explosive_plays",

            "turnovers": "def_turnovers",
            "sacks": "def_sacks",

            "yards": "def_yards",
        }
    )

    # The defense's "plays" should not overwrite offensive plays.
    # Keep both sides available.
    return off.merge(
        dfn,
        on=["game_id", "team", "opp"],
        how="outer",
    )


# ---------------------------------------------------------------------
# QB DATA
# ---------------------------------------------------------------------

QB_COLS = [
    "game_id",
    "posteam",
    "passer_player_id",
    "qb_dropback",
    "epa",
    "wp",
]


def _aggregate_qb(season: int) -> pd.DataFrame:
    """
    Per game / QB:
      - dropbacks
      - EPA per dropback

    This is later converted into a pre-game QB rating.
    """

    pl = nfl.load_pbp([season])

    cols = [
        c for c in QB_COLS
        if c in pl.columns
    ]

    p = pl.select(cols).to_pandas()

    p = p[
        (p["qb_dropback"] == 1)
        & p["epa"].notna()
        & p["passer_player_id"].notna()
    ].copy()

    if "wp" in p.columns:
        p = p[
            p["wp"].isna()
            | p["wp"].between(0.05, 0.95)
        ]

    return (
        p.groupby(
            ["game_id", "posteam", "passer_player_id"]
        )
        .agg(
            qb_db=("epa", "size"),
            qb_epa=("epa", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "passer_player_id": "qb_id"
            }
        )
    )


def qb_game_stats(seasons, refresh=False) -> pd.DataFrame:
    """Load/cache QB game-level statistics."""

    cur = current_season()
    frames = []

    for yr in seasons:
        f = CACHE / f"qb_{yr}.parquet"

        if (
            f.exists()
            and not refresh
            and yr < cur
        ):
            frames.append(pd.read_parquet(f))
            continue

        print(
            f"  building QB aggregates for {yr} ...",
            flush=True,
        )

        df = _aggregate_qb(yr)

        if yr < cur:
            df.to_parquet(f)

        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "game_id",
                "posteam",
                "qb_id",
                "qb_db",
                "qb_epa",
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# TEAM STAT CACHE
# ---------------------------------------------------------------------

def team_game_stats(
    seasons,
    refresh=False,
) -> pd.DataFrame:
    """Load/cache team game-level play-by-play statistics."""

    cur = current_season()
    frames = []

    for yr in seasons:
        f = CACHE / f"tg_{yr}.parquet"

        if (
            f.exists()
            and not refresh
            and yr < cur
        ):
            frames.append(pd.read_parquet(f))
            continue

        print(
            f"  building play-by-play aggregates for {yr} ...",
            flush=True,
        )

        df = _aggregate_pbp(yr)

        if yr < cur:
            df.to_parquet(f)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# SMALL HELPER
# ---------------------------------------------------------------------

def np_nan_series(n: int) -> pd.Series:
    """Return an all-NaN Series with a known length."""
    return pd.Series(float("nan"), index=range(n))