import argparse, os, glob
import pandas as pd
from src.utils import load_config, ensure_dir
from src.features import build_hitter_features, build_pitcher_features

def main():
    parser = argparse.ArgumentParser(description="Build season-level features for hitters/pitchers from raw Statcast CSVs.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    min_pa = int(cfg.get("min_pa", 50))
    min_bf = int(cfg.get("min_bf", 50))

    raw_files = sorted(glob.glob("data/raw/statcast_*.csv"))
    if not raw_files:
        raise SystemExit("No raw files found. Run pull_statcast.py first.")

    ensure_dir("data/processed")
    hitters_all, pitchers_all = [], []

    for f in raw_files:
        df = pd.read_csv(f, low_memory=False)
        hitters = build_hitter_features(df, min_pa=min_pa)
        pitchers = build_pitcher_features(df, min_bf=min_bf)
        hitters_all.append(hitters)
        pitchers_all.append(pitchers)

    hitters_out = pd.concat(hitters_all, ignore_index=True) if hitters_all else pd.DataFrame()
    pitchers_out = pd.concat(pitchers_all, ignore_index=True) if pitchers_all else pd.DataFrame()

    hitters_out.to_csv("data/processed/hitters_features.csv", index=False)
    pitchers_out.to_csv("data/processed/pitchers_features.csv", index=False)

    print("Wrote data/processed/hitters_features.csv", hitters_out.shape)
    print("Wrote data/processed/pitchers_features.csv", pitchers_out.shape)

if __name__ == "__main__":
    main()
