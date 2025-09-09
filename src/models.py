from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score

def _season_shift(df: pd.DataFrame, id_col: str, target_col: str, season_col: str = "season") -> pd.DataFrame:
    """
    Align features from season t-1 to predict target in season t for the same player.
    """
    prev = df.copy()
    prev[season_col] = prev[season_col] + 1
    merged = pd.merge(df[[id_col, season_col, target_col]], prev, on=[id_col, season_col], how="inner", suffixes=("_y", "_x"))
    # after merge: *_x are t-1 features, *_y is the target at t. We'll clean up outside.
    return merged

def train_hitters_gbr(features: pd.DataFrame, random_state: int = 42) -> Tuple[GradientBoostingRegressor, float]:
    """
    Train a simple GBR to predict next-season wOBA using prior-season features.
    Returns model, cross-validated R^2.
    """
    req_cols = ["player_id","season","pa","woba","xwoba","avg_ev","k_rate","bb_rate","barrel_rate"]
    for c in req_cols:
        if c not in features.columns:
            raise ValueError(f"Missing required column: {c}")

    f = features.dropna(subset=["woba"]).copy()
    # Build supervised pairs: predict t (woba_y) from t-1 features (suffix _x)
    merged = _season_shift(f, id_col="player_id", target_col="woba", season_col="season")
    # Keep relevant features from *_x (t-1 season)
    X = merged[["pa_x","woba_x","xwoba_x","avg_ev_x","k_rate_x","bb_rate_x","barrel_rate_x"]].fillna(0.0).values
    y = merged["woba_y"].values

    model = GradientBoostingRegressor(random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    r2 = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=make_scorer(r2_score))))
    model.fit(X, y)
    return model, r2
