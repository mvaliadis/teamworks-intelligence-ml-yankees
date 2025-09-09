# scripts/train_model.py
import json, sys, pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# allow "src" imports when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models import _season_shift

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQ = ["player_id","season","pa","woba","xwoba","avg_ev","k_rate","bb_rate","barrel_rate"]

def _load_features() -> pd.DataFrame:
    f = Path("data/processed/hitters_features.csv")
    if not f.exists():
        raise SystemExit("Missing data/processed/hitters_features.csv. Run build_features.py first.")
    df = pd.read_csv(f)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df

def _build_next_season(df: pd.DataFrame):
    m = _season_shift(df.dropna(subset=["woba"]).copy(), id_col="player_id", target_col="woba", season_col="season")
    X_cols = ["pa_x","woba_x","xwoba_x","avg_ev_x","k_rate_x","bb_rate_x","barrel_rate_x"]
    have = [c for c in X_cols if c in m.columns]
    if len(have) < 2 or m.empty:
        return None
    X = m[have].fillna(0.0).values
    y = m["woba_y"].values
    groups = m["player_id"].values
    return {"mode":"next_season","X":X,"y":y,"groups":groups,"X_cols":have}

def _build_within_season(df: pd.DataFrame):
    X_cols = ["xwoba","avg_ev","k_rate","bb_rate","barrel_rate","pa"]
    have = [c for c in X_cols if c in df.columns]
    X = df[have].fillna(0.0).values
    y = df["woba"].values
    groups = df["player_id"].values
    return {"mode":"within_season","X":X,"y":y,"groups":groups,"X_cols":have}

def main():
    df = _load_features()
    ds = _build_next_season(df)
    if ds is None:
        ds = _build_within_season(df)

    # CV strategy
    uniq = np.unique(ds["groups"])
    n_splits = max(2, min(5, len(uniq), len(ds["y"])))
    cv = GroupKFold(n_splits=n_splits)

    gbr = GradientBoostingRegressor(random_state=42)
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    gs = GridSearchCV(gbr, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
    gs.fit(ds["X"], ds["y"], groups=ds["groups"])
    yhat = gs.predict(ds["X"])

    report = {
        "mode": ds["mode"],
        "cv": f"GroupKFold(n_splits={n_splits})",
        "model": {
            "r2": float(r2_score(ds["y"], yhat)),
            "mae": float(mean_absolute_error(ds["y"], yhat)),
        },
        "best_params": gs.best_params_,
        "n_samples": int(len(ds["y"])),
        "x_features": ds["X_cols"],
    }
    (OUT_DIR / "model_report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(gs.best_estimator_, OUT_DIR / "model.pkl")
    (OUT_DIR / "model_features.json").write_text(json.dumps(ds["X_cols"], indent=2))

    print("Saved:", OUT_DIR / "model.pkl")
    print("Report:", report)

if __name__ == "__main__":
    main()
