"""Leakage-safe opponent-adjusted NFL pre-game features."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nfl_data import load_schedule, team_game_stats, qb_game_stats
from power_ratings import add_power_ratings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


TEAM_MAP = {
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}


# ---------------------------------------------------------------------
# TEAM STATISTICS
# ---------------------------------------------------------------------

STATS = [
    "off_epa",
    "off_epa_pass",
    "off_epa_rush",
    "off_sr",

    "def_epa",
    "def_epa_pass",
    "def_epa_rush",
    "def_sr",

    "pts_for",
    "pts_against",
    "plays",

    "pass_attempts",
    "rush_attempts",
    "explosive_plays",
    "turnovers",
    "sacks",
    "yards",

    "def_pass_attempts",
    "def_rush_attempts",
    "def_explosive_plays",
    "def_turnovers",
    "def_sacks",
    "def_yards",
]


# Statistics where opponent adjustment makes sense.
PARTNER = {
    "off_epa": "def_epa",
    "def_epa": "off_epa",

    "off_epa_pass": "def_epa_pass",
    "def_epa_pass": "off_epa_pass",

    "off_epa_rush": "def_epa_rush",
    "def_epa_rush": "off_epa_rush",

    "off_sr": "def_sr",
    "def_sr": "off_sr",

    "pts_for": "pts_against",
    "pts_against": "pts_for",

    "plays": "plays",

    "pass_attempts": "def_pass_attempts",
    "def_pass_attempts": "pass_attempts",

    "rush_attempts": "def_rush_attempts",
    "def_rush_attempts": "rush_attempts",

    "explosive_plays": "def_explosive_plays",
    "def_explosive_plays": "explosive_plays",

    "yards": "def_yards",
    "def_yards": "yards",
}


# ---------------------------------------------------------------------
# LONG TEAM/GAME TABLE
# ---------------------------------------------------------------------

def _long_table(sched, tg):
    base = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
    ]

    # Home team
    h = sched[base].copy()
    h["team"] = sched["home_team"]
    h["opp"] = sched["away_team"]
    h["is_home"] = 1
    h["pts_for"] = sched["home_score"]
    h["pts_against"] = sched["away_score"]
    h["rest"] = sched["home_rest"]
    h["qb_id"] = sched["home_qb_id"]

    # Away team
    a = sched[base].copy()
    a["team"] = sched["away_team"]
    a["opp"] = sched["home_team"]
    a["is_home"] = 0
    a["pts_for"] = sched["away_score"]
    a["pts_against"] = sched["home_score"]
    a["rest"] = sched["away_rest"]
    a["qb_id"] = sched["away_qb_id"]

    L = pd.concat(
        [h, a],
        ignore_index=True,
    )

    # Team/game statistics
    stats = (
        tg
        .drop(columns=["opp"], errors="ignore")
        .drop_duplicates(["game_id", "team"])
    )

    L = L.merge(
        stats,
        on=["game_id", "team"],
        how="left",
    )

    L = (
        L.sort_values(
            ["team", "gameday", "game_id"]
        )
        .reset_index(drop=True)
    )

    return L


# ---------------------------------------------------------------------
# PRE-GAME EXPONENTIALLY WEIGHTED AVERAGES
# ---------------------------------------------------------------------

def _pre_game_ewm(L, cols, halflife):
    out = {}

    for c in cols:

        post = L.groupby(
            "team",
            sort=False,
        )[c].transform(
            lambda s: s.ewm(
                halflife=halflife,
                ignore_na=True,
            ).mean()
        )

        post = post.groupby(
            L["team"]
        ).ffill()

        # IMPORTANT:
        # Shift one game so the current game can never
        # influence its own pre-game feature.
        out[c] = post.groupby(
            L["team"]
        ).shift(1)

    return pd.DataFrame(
        out,
        index=L.index,
    )


# ---------------------------------------------------------------------
# TEAM RATINGS
# ---------------------------------------------------------------------

def build_team_ratings(
    sched,
    tg,
    halflife=6.0,
):

    L = _long_table(
        sched,
        tg,
    )

    played = L["pts_for"].notna()

    raw = _pre_game_ewm(
        L,
        STATS,
        halflife,
    )

    # ---------------------------------------------------------------
    # Attach opponent's pre-game ratings
    # ---------------------------------------------------------------

    opp = L[
        [
            "game_id",
            "team",
        ]
    ].copy()

    for c in STATS:
        opp[f"o_{c}"] = raw[c].values

    opp = opp.rename(
        columns={
            "team": "opp"
        }
    )

    n = len(L)

    L = L.merge(
        opp,
        on=[
            "game_id",
            "opp",
        ],
        how="left",
    )

    assert len(L) == n

    # ---------------------------------------------------------------
    # Opponent-adjusted statistics
    # ---------------------------------------------------------------

    adj_cols = []

    for c in STATS:

        # Use league mean as the baseline.
        lvl = L.loc[
            played,
            c
        ].mean()

        if c in PARTNER:

            L[f"adj_{c}"] = (
                L[c]
                - L[f"o_{PARTNER[c]}"]
                + lvl
            )

        else:

            # Some statistics do not have a clean
            # opponent-adjustment interpretation.
            L[f"adj_{c}"] = L[c]

        adj_cols.append(
            f"adj_{c}"
        )

    # ---------------------------------------------------------------
    # Pre-game ratings of adjusted statistics
    # ---------------------------------------------------------------

    rated = _pre_game_ewm(
        L,
        adj_cols,
        halflife,
    )

    for c in STATS:

        L[f"r_{c}"] = rated[
            f"adj_{c}"
        ].values

    # ---------------------------------------------------------------
    # Season experience
    # ---------------------------------------------------------------

    L["games_in_season"] = (
        L.groupby(
            [
                "team",
                "season",
            ]
        ).cumcount()
    )

    # ---------------------------------------------------------------
    # QB change indicator
    # ---------------------------------------------------------------

    prev_qb = (
        L.groupby("team")["qb_id"]
        .shift(1)
    )

    L["qb_changed"] = (
        L["qb_id"].notna()
        & prev_qb.notna()
        & (
            L["qb_id"]
            != prev_qb
        )
    ).astype(int)

    return L


# ---------------------------------------------------------------------
# QB RATINGS
# ---------------------------------------------------------------------

def add_qb_ratings(
    L,
    qbg,
    halflife_games=16.0,
    k=150.0,
):
    """
    Leakage-safe pre-game QB rating.

    Only previous QB dropbacks are used to construct
    the rating available before each game.
    """

    decay = (
        0.5
        ** (
            1.0
            / halflife_games
        )
    )

    # ---------------------------------------------------------------
    # League-average prior
    # ---------------------------------------------------------------

    if qbg.empty:

        prior = 0.0

    else:

        years = (
            qbg["game_id"]
            .astype(str)
            .str[:4]
            .astype(int)
        )

        old = qbg[
            years < 2018
        ]

        denominator = old[
            "qb_db"
        ].sum()

        if (
            len(old) > 0
            and pd.notna(denominator)
            and denominator > 0
        ):

            prior = float(
                (
                    old["qb_epa"]
                    * old["qb_db"]
                ).sum()
                / denominator
            )

        else:

            denominator = qbg[
                "qb_db"
            ].sum()

            if (
                pd.notna(denominator)
                and denominator > 0
            ):

                prior = float(
                    (
                        qbg["qb_epa"]
                        * qbg["qb_db"]
                    ).sum()
                    / denominator
                )

            else:

                prior = 0.0

    # ---------------------------------------------------------------
    # Attach QB game-level performance
    # ---------------------------------------------------------------

    tl = (
        L.loc[
            L["qb_id"].notna(),
            [
                "game_id",
                "team",
                "qb_id",
                "gameday",
            ],
        ]
        .merge(
            qbg[
                [
                    "game_id",
                    "qb_id",
                    "qb_db",
                    "qb_epa",
                ]
            ],
            on=[
                "game_id",
                "qb_id",
            ],
            how="left",
        )
        .sort_values(
            [
                "qb_id",
                "gameday",
                "game_id",
            ]
        )
    )

    rating = {}
    exp = {}

    # ---------------------------------------------------------------
    # Build chronological QB histories
    # ---------------------------------------------------------------

    for _, grp in tl.groupby(
        "qb_id",
        sort=False,
    ):

        num = 0.0
        den = 0.0
        cnt = 0.0

        for (
            gid,
            team,
            n,
            e,
        ) in zip(
            grp["game_id"],
            grp["team"],
            grp["qb_db"],
            grp["qb_epa"],
        ):

            # PRE-GAME rating
            rating[
                (gid, team)
            ] = (
                num
                + k * prior
            ) / (
                den
                + k
            )

            exp[
                (gid, team)
            ] = cnt

            # Add current game only AFTER
            # assigning its pre-game rating.
            if (
                pd.notna(n)
                and n > 0
                and pd.notna(e)
            ):

                num = (
                    num * decay
                    + n * e
                )

                den = (
                    den * decay
                    + n
                )

                cnt += n

    # ---------------------------------------------------------------
    # Attach QB ratings
    # ---------------------------------------------------------------

    keys = list(
        zip(
            L["game_id"],
            L["team"],
        )
    )

    L["qb_rating"] = [
        rating.get(
            key,
            prior,
        )
        for key in keys
    ]

    L["qb_exp_db"] = [
        exp.get(
            key,
            0.0,
        )
        for key in keys
    ]

    return L


# ---------------------------------------------------------------------
# RATING COLUMNS
# ---------------------------------------------------------------------

RCOLS = (
    [
        f"r_{c}"
        for c in STATS
    ]
    + [
        "rest",
        "games_in_season",
        "qb_changed",
        "qb_rating",
        "qb_exp_db",
    ]
)


# ---------------------------------------------------------------------
# GAME FEATURES
# ---------------------------------------------------------------------

def build_game_features(
    seasons,
    halflife=6.0,
):

    seasons = list(seasons)

    # ---------------------------------------------------------------
    # Load schedule
    # ---------------------------------------------------------------

    sched = load_schedule(
        seasons
    )

    for c in (
        "home_team",
        "away_team",
    ):
        sched[c] = sched[
            c
        ].replace(
            TEAM_MAP
        )

    # ---------------------------------------------------------------
    # Load team statistics
    # ---------------------------------------------------------------

    tg = team_game_stats(
        seasons
    )

    for c in (
        "team",
        "opp",
    ):
        if c in tg.columns:
            tg[c] = tg[
                c
            ].replace(
                TEAM_MAP
            )

    # ---------------------------------------------------------------
    # Build ratings
    # ---------------------------------------------------------------

    L = build_team_ratings(
        sched,
        tg,
        halflife,
    )

    L = add_qb_ratings(
        L,
        qb_game_stats(seasons),
    )

    # ---------------------------------------------------------------
    # Split home / away
    # ---------------------------------------------------------------

    H = (
        L[
            L["is_home"] == 1
        ]
        .set_index("game_id")
    )

    A = (
        L[
            L["is_home"] == 0
        ]
        .set_index("game_id")
    )

    # ---------------------------------------------------------------
    # Game-level information
    # ---------------------------------------------------------------

    keep = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",

        "home_team",
        "away_team",

        "home_score",
        "away_score",

        "result",
        "total",
        "played",

        "spread_line",
        "total_line",

        "home_spread_odds",
        "away_spread_odds",

        "over_odds",
        "under_odds",

        "home_moneyline",
        "away_moneyline",

        "div_game",
        "roof",
        "location",
        "temp",
        "wind",
    ]

    g = (
        sched[
            keep
        ]
        .set_index("game_id")
        .copy()
    )

    # ---------------------------------------------------------------
    # Home / away ratings
    # ---------------------------------------------------------------

    for c in RCOLS:

        g[
            f"h_{c}"
        ] = H[c]

        g[
            f"a_{c}"
        ] = A[c]

    # ---------------------------------------------------------------
    # EPA matchup features
    # ---------------------------------------------------------------

    g["h_exp_epa"] = (
        g["h_r_off_epa"]
        + g["a_r_def_epa"]
    )

    g["a_exp_epa"] = (
        g["a_r_off_epa"]
        + g["h_r_def_epa"]
    )

    g["epa_edge"] = (
        g["h_exp_epa"]
        - g["a_exp_epa"]
    )

    g["epa_sum"] = (
        g["h_exp_epa"]
        + g["a_exp_epa"]
    )

    # ---------------------------------------------------------------
    # Pass EPA
    # ---------------------------------------------------------------

    g["h_exp_pass"] = (
        g["h_r_off_epa_pass"]
        + g["a_r_def_epa_pass"]
    )

    g["a_exp_pass"] = (
        g["a_r_off_epa_pass"]
        + g["h_r_def_epa_pass"]
    )

    g["pass_epa_edge"] = (
        g["h_exp_pass"]
        - g["a_exp_pass"]
    )

    # ---------------------------------------------------------------
    # Rush EPA
    # ---------------------------------------------------------------

    g["h_exp_rush"] = (
        g["h_r_off_epa_rush"]
        + g["a_r_def_epa_rush"]
    )

    g["a_exp_rush"] = (
        g["a_r_off_epa_rush"]
        + g["h_r_def_epa_rush"]
    )

    g["rush_epa_edge"] = (
        g["h_exp_rush"]
        - g["a_exp_rush"]
    )

    # ---------------------------------------------------------------
    # Success rate
    # ---------------------------------------------------------------

    g["h_exp_sr"] = (
        g["h_r_off_sr"]
        + g["a_r_def_sr"]
    )

    g["a_exp_sr"] = (
        g["a_r_off_sr"]
        + g["h_r_def_sr"]
    )

    g["sr_edge"] = (
        g["h_exp_sr"]
        - g["a_exp_sr"]
    )

    # ---------------------------------------------------------------
    # Points
    # ---------------------------------------------------------------

    g["h_net_points"] = (
        g["h_r_pts_for"]
        - g["h_r_pts_against"]
    )

    g["a_net_points"] = (
        g["a_r_pts_for"]
        - g["a_r_pts_against"]
    )

    g["pts_edge"] = (
        g["h_net_points"]
        - g["a_net_points"]
    )

    g["pts_sum"] = (
        g["h_r_pts_for"]
        + g["a_r_pts_for"]
        + g["h_r_pts_against"]
        + g["a_r_pts_against"]
    )

    # ---------------------------------------------------------------
    # Passing / rushing volume
    # ---------------------------------------------------------------

    g["h_pass_rate"] = (
        g["h_r_pass_attempts"]
        / (
            g["h_r_pass_attempts"]
            + g["h_r_rush_attempts"]
            + 1e-6
        )
    )

    g["a_pass_rate"] = (
        g["a_r_pass_attempts"]
        / (
            g["a_r_pass_attempts"]
            + g["a_r_rush_attempts"]
            + 1e-6
        )
    )

    g["pass_rate_diff"] = (
        g["h_pass_rate"]
        - g["a_pass_rate"]
    )

    # ---------------------------------------------------------------
    # Yardage
    # ---------------------------------------------------------------

    g["h_net_yards"] = (
        g["h_r_yards"]
        - g["h_r_def_yards"]
    )

    g["a_net_yards"] = (
        g["a_r_yards"]
        - g["a_r_def_yards"]
    )

    g["yards_edge"] = (
        g["h_net_yards"]
        - g["a_net_yards"]
    )

    # ---------------------------------------------------------------
    # Explosive plays
    # ---------------------------------------------------------------

    g["h_explosive_edge_component"] = (
        g["h_r_explosive_plays"]
        - g["h_r_def_explosive_plays"]
    )

    g["a_explosive_edge_component"] = (
        g["a_r_explosive_plays"]
        - g["a_r_def_explosive_plays"]
    )

    g["explosive_edge"] = (
        g["h_explosive_edge_component"]
        - g["a_explosive_edge_component"]
    )

    # ---------------------------------------------------------------
    # Turnovers
    #
    # Offensive turnovers are bad.
    # Defensive turnovers represent turnovers generated against
    # the opposing offense and are therefore beneficial.
    # ---------------------------------------------------------------

    g["h_turnover_margin"] = (
        g["h_r_def_turnovers"]
        - g["h_r_turnovers"]
    )

    g["a_turnover_margin"] = (
        g["a_r_def_turnovers"]
        - g["a_r_turnovers"]
    )

    g["turnover_edge"] = (
        g["h_turnover_margin"]
        - g["a_turnover_margin"]
    )

    # ---------------------------------------------------------------
    # Sacks
    # ---------------------------------------------------------------

    g["h_sack_margin"] = (
        g["h_r_def_sacks"]
        - g["h_r_sacks"]
    )

    g["a_sack_margin"] = (
        g["a_r_def_sacks"]
        - g["a_r_sacks"]
    )

    g["sack_edge"] = (
        g["h_sack_margin"]
        - g["a_sack_margin"]
    )

    # ---------------------------------------------------------------
    # QB
    # ---------------------------------------------------------------

    g["qb_edge"] = (
        g["h_qb_rating"]
        - g["a_qb_rating"]
    )

    g["qb_exp_edge"] = (
        g["h_qb_exp_db"]
        - g["a_qb_exp_db"]
    )

    g["season_exp_diff"] = (
        g["h_games_in_season"]
        - g["a_games_in_season"]
    )

    g["qb_changed_diff"] = (
        g["h_qb_changed"]
        - g["a_qb_changed"]
    )

    # ---------------------------------------------------------------
    # Rest
    # ---------------------------------------------------------------

    g["rest_h"] = (
        g["h_rest"]
        .clip(3, 14)
    )

    g["rest_a"] = (
        g["a_rest"]
        .clip(3, 14)
    )

    g["rest_diff"] = (
        g["rest_h"]
        - g["rest_a"]
    )

    # ---------------------------------------------------------------
    # Environment
    # ---------------------------------------------------------------

    g["is_neutral"] = (
        g["location"]
        == "Neutral"
    ).astype(int)

    g["indoor"] = (
        g["roof"].isin(
            [
                "dome",
                "closed",
            ]
        )
    ).astype(int)

    g["temp_f"] = np.where(
        g["indoor"] == 1,
        68.0,
        g["temp"],
    ).astype(float)

    g["wind_mph"] = np.where(
        g["indoor"] == 1,
        0.0,
        g["wind"],
    ).astype(float)

    g["div_game"] = (
        g["div_game"]
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Totals-oriented combinations
    # ---------------------------------------------------------------

    g = g.copy()                                   # de-fragment after many column inserts
    # weather for totals: strong wind and cold lower scoring (indoor games count as calm and mild)
    g["wind_strong"] = np.clip(g["wind_mph"].fillna(g["wind_mph"].median()) - 10, 0, None)
    g["cold"] = np.clip(40 - g["temp_f"].fillna(60), 0, None)
    g["pace_sum"] = g["h_r_plays"] + g["a_r_plays"]
    g["sr_sum"] = g["h_exp_sr"] + g["a_exp_sr"]
    g["qb_sum"] = g["h_qb_rating"] + g["a_qb_rating"]
    g["explosive_sum"] = g["h_r_explosive_plays"] + g["a_r_explosive_plays"]

    # ---------------------------------------------------------------
    # Ridge power ratings (results-based and market-based, pre-game)
    # ---------------------------------------------------------------

    g = add_power_ratings(g.reset_index())

    # ---------------------------------------------------------------
    # Injuries: regular starters on the official injury report
    # ---------------------------------------------------------------

    from injuries import add_injury_features
    g = add_injury_features(g)

    return g, dict(FEATURES)


# ---------------------------------------------------------------------
# FINAL FEATURE LISTS
#
# v2 fed 98 raw columns to the model; most were noise. These compact
# sets did better out of sample (margin RMSE 13.07 vs 13.29 walk-forward
# 2018-2026). Game-day weather is deliberately excluded: history holds
# ACTUAL weather while live predictions would only have forecasts.
# ---------------------------------------------------------------------

FEATURES = {
    "margin": [
        "epa_edge", "pass_epa_edge", "rush_epa_edge", "sr_edge", "pts_edge",
        "turnover_edge", "sack_edge", "qb_edge", "qb_changed_diff",
        "rest_diff", "is_neutral", "div_game",
        "res_prior", "mkt_prior",
        "inj_off_diff", "inj_def_diff",
    ],
    "total": [
        "epa_sum", "pts_sum", "pace_sum", "sr_sum", "qb_sum", "explosive_sum",
        "indoor", "is_neutral", "wind_strong", "cold",
        "res_total_prior", "mkt_total_prior",
        "inj_off_sum", "inj_def_sum",
    ],
}
