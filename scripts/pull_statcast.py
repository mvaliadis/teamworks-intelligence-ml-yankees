import argparse, os
import pandas as pd
from src.utils import load_config, month_ranges, ensure_dir
from src.data import pull_statcast_season

def main():
    parser = argparse.ArgumentParser(description="Pull MLB Statcast data by month and save to CSV.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    team = cfg.get("team", "NYY")
    years = cfg.get("years", [2023, 2024])
    start_month = int(cfg.get("start_month", 3))
    end_month = int(cfg.get("end_month", 10))
    cache_dir = cfg.get("cache_dir")

    out_dir = "data/raw"
    ensure_dir(out_dir)

    all_files = []
    for y in years:
        mr = list(month_ranges(y, start_month, end_month))
        df = pull_statcast_season(team=team, year=y, month_ranges=mr, cache_dir=cache_dir)
        out_path = os.path.join(out_dir, f"statcast_{team or 'ALL'}_{y}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} ({len(df):,} rows)")
        all_files.append(out_path)

    print("Done.", all_files)

if __name__ == "__main__":
    main()
