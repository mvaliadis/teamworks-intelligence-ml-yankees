import argparse, os
import pandas as pd
from src.utils import load_config
from src.models import train_hitters_gbr

def main():
    parser = argparse.ArgumentParser(description="Train a simple hitters wOBA prediction model (t-1 -> t).")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # We train league-wide if available for better sample size (use data from multiple teams)
    fpath = "data/processed/hitters_features.csv"
    if not os.path.exists(fpath):
        raise SystemExit("Missing hitters_features.csv. Run build_features.py first.")

    features = pd.read_csv(fpath)
    model, r2 = train_hitters_gbr(features)
    print(f"5-fold CV R^2 (hitters wOBA t-1 -> t): {r2:.3f}")

    # Save a tiny model artifact (sklearn pickling for demo)
    import joblib
    joblib.dump(model, "data/processed/model_hitters_gbr.joblib")
    print("Saved model to data/processed/model_hitters_gbr.joblib")

if __name__ == "__main__":
    main()
