import json, os, sys, pathlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

# allow "src" imports when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.models import _season_shift   # aligns season t-1 -> t

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REQ = ["player_id","season","pa","woba","xwoba","avg_ev","k_rate","bb_rate","barrel_rate"]

def _check_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def _load_hitters_features() -> pd.DataFrame:
    f = Path("data/processed/hitters_features.csv")
    if f.exists():
        return pd.read_csv(f)   # <-- remove the len>=5 guard
    samp = Path("sample_data/hitters_features_sample.csv")
    if samp.exists():
        print("[INFO] Using sample_data/hitters_features_sample.csv for evaluation.")
        return pd.read_csv(samp)
    raise SystemExit("No hitters features found. Run build_features.py or ensure sample_data exists.")

def build_supervised_next_season(features: pd.DataFrame):
    _check_cols(features, REQ)
    f = features.dropna(subset=["woba"]).copy()
    merged = _season_shift(f, id_col="player_id", target_col="woba", season_col="season")
    if merged.empty:
        return None
    X_cols = ["pa_x","woba_x","xwoba_x","avg_ev_x","k_rate_x","bb_rate_x","barrel_rate_x"]
    have = [c for c in X_cols if c in merged.columns]
    if len(have) < 2:
        return None
    X = merged[have].fillna(0.0).values
    y = merged["woba_y"].values
    groups = merged["player_id"].values
    baseline = merged["woba_x"].values  # carry-forward baseline
    return {"mode": "next_season", "X": X, "y": y, "groups": groups, "X_cols": have, "baseline": baseline}

def build_supervised_within_season(features: pd.DataFrame):
    _check_cols(features, REQ)
    f = features.dropna(subset=["woba"]).copy()
    X_cols = ["xwoba","avg_ev","k_rate","bb_rate","barrel_rate","pa"]
    have = [c for c in X_cols if c in f.columns]
    if len(have) < 2:
        raise SystemExit("Not enough columns to run within-season evaluation.")
    X = f[have].fillna(0.0).values
    y = f["woba"].values
    groups = f["player_id"].values
    baseline = f["xwoba"].fillna(f["xwoba"].mean()).values if "xwoba" in f.columns else np.full_like(y, y.mean())
    return {"mode": "within_season", "X": X, "y": y, "groups": groups, "X_cols": have, "baseline": baseline}

def _make_cv(groups: np.ndarray, n_samples: int):
    unique_groups = np.unique(groups) if groups is not None else np.array([])
    if groups is not None and len(unique_groups) >= 2:
        n_splits = max(2, min(5, len(unique_groups), n_samples))
        return GroupKFold(n_splits=n_splits), f"GroupKFold(n_splits={n_splits})"
    # fallback to plain KFold when grouping isn’t viable
    n_splits = max(2, min(5, n_samples))
    return KFold(n_splits=n_splits, shuffle=True, random_state=42), f"KFold(n_splits={n_splits})"

def run_eval(dataset):
    n = len(dataset["y"])
    if n < 2:
        report = {
            "mode": dataset["mode"],
            "note": "Too few samples to train/evaluate a model. Showing baseline only.",
            "baseline": {
                "r2": float("nan"),
                "mae": float(mean_absolute_error(dataset["y"], dataset["baseline"])) if n else float("nan"),
            },
            "model": {"r2": float("nan"), "mae": float("nan")},
            "best_params": {},
            "n_samples": int(n),
            "x_features": dataset["X_cols"],
        }
        with open(OUT_DIR / "model_report.json", "w") as f:
            json.dump(report, f, indent=2)
        pd.DataFrame(columns=["feature","importance_mean","importance_std"]) \
            .to_csv(OUT_DIR / "feature_importance.csv", index=False)
        print("Wrote minimal report (insufficient samples).")
        return

    cv, cv_name = _make_cv(dataset["groups"], n)
    gbr = GradientBoostingRegressor(random_state=42)
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    gs = GridSearchCV(gbr, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)

    # ✅ pass groups for GroupKFold (safe for KFold too)
    gs.fit(dataset["X"], dataset["y"], groups=dataset["groups"])

    y_pred = gs.predict(dataset["X"])

    baseline_r2 = float(r2_score(dataset["y"], dataset["baseline"]))
    baseline_mae = float(mean_absolute_error(dataset["y"], dataset["baseline"]))
    model_r2 = float(r2_score(dataset["y"], y_pred))
    model_mae = float(mean_absolute_error(dataset["y"], y_pred))

    try:
        pi = permutation_importance(gs.best_estimator_, dataset["X"], dataset["y"],
                                    n_repeats=10, random_state=42, scoring="r2")
        importances = (
            pd.DataFrame({"feature": dataset["X_cols"], "importance_mean": pi.importances_mean,
                          "importance_std": pi.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        importances = pd.DataFrame({"feature": dataset["X_cols"], "importance_mean": [], "importance_std": []})

    importances.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    report = {
        "mode": dataset["mode"],
        "cv": cv_name,
        "baseline": {"r2": baseline_r2, "mae": baseline_mae},
        "model": {"r2": model_r2, "mae": model_mae},
        "best_params": gs.best_params_,
        "n_samples": int(n),
        "x_features": dataset["X_cols"],
    }
    with open(OUT_DIR / "model_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report:", report)
    print("Wrote:", OUT_DIR / "model_report.json", "and feature_importance.csv")

def main():
    features = _load_hitters_features()
    ds = build_supervised_next_season(features)
    if ds is None:
        print("[INFO] Not enough back-to-back seasons per player; falling back to within-season evaluation.")
        ds = build_supervised_within_season(features)
    run_eval(ds)

if __name__ == "__main__":
    main()
