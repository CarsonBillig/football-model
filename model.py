"""Independent ("fundamentals") projection model and leakage-safe walk-forward validation.

The model predicts the final home margin and the game total from pre-game football features only. It never sees
the current game's betting line; the market enters later, in pricing.py, where the two views are blended with a
weight the backtest has earned (see backtest.py).

A model is NEVER trained on games that occur on or after the week it predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {"margin": "result", "total": "total"}          # model name -> column in the schedule


def make_ridge():
    """Standardised ridge regression; median-imputes the few missing ratings (early-season, new teams)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-1, 4, 21)))


def usable(games: pd.DataFrame) -> pd.DataFrame:
    """Completed regular-season games, chronologically ordered. Missing features are imputed, not dropped."""
    d = games[games["played"] & (games["game_type"] == "REG")].dropna(subset=["result", "total"])
    d = d.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    return d.assign(unit=d["season"] * 100 + d["week"])


def fit_models(train: pd.DataFrame, features: dict) -> dict:
    return {name: make_ridge().fit(train[features[name]], train[tgt]) for name, tgt in TARGETS.items()}


def predict_models(models: dict, df: pd.DataFrame, features: dict) -> pd.DataFrame:
    return pd.DataFrame({f"fund_{name}": models[name].predict(df[features[name]]) for name in TARGETS},
                        index=df.index)


def walk_forward(games: pd.DataFrame, features: dict, first_test_season: int = 2018,
                 min_train: int = 500, verbose: bool = True) -> pd.DataFrame:
    """Retrain every week on all earlier weeks and predict that week. Returns one row per test game."""
    d = usable(games)
    parts = []
    for unit in sorted(d.loc[d["season"] >= first_test_season, "unit"].unique()):
        train, test = d[d["unit"] < unit], d[d["unit"] == unit]
        if len(train) < min_train:
            continue
        models = fit_models(train, features)
        parts.append(pd.concat([test, predict_models(models, test, features)], axis=1))
        if verbose and unit % 100 == 1:
            print(f"  walk-forward {unit // 100}: trained on {len(train)} games", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
