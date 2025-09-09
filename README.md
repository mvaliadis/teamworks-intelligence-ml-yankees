# Yankees (MLB) – End‑to‑End Sports Analytics (Statcast)

<p align="left">
  <!-- Replace USER/REPO after pushing -->
  <a href="https://github.com/USER/REPO/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
</p>


A compact, reproducible baseball analytics project you can run locally and publish as a portfolio repo.
It pulls MLB Statcast data (via [`pybaseball`](https://github.com/jldbc/pybaseball)), builds hitter/pitcher features,
and produces team‑ready leaderboards plus a simple Streamlit app.

> **Why this repo?** Shows practical, production‑minded DS for sports: data retrieval, feature engineering on noisy tracking‑adjacent pitch‑level data, evaluation, and a small app—cleanly organized and easy to extend to other teams/sports.

## Quickstart

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Configure (defaults to Yankees, 2023–2024)
cp config.example.yaml config.yaml  # edit as needed

# 3) Pull Statcast (chunks by month) -> data/raw/
python scripts/pull_statcast.py --config config.yaml

# 4) Build features -> data/processed/
python scripts/build_features.py --config config.yaml

# 5) (Optional) Train a simple model to predict season wOBA from last season features
python scripts/train_model.py --config config.yaml

# 6) Explore a small leaderboard app
streamlit run app.py
```

## Repo Layout

```
.
├─ app.py                      # Streamlit leaderboard for hitters/pitchers
├─ config.example.yaml         # Editable settings (team, years, filters)
├─ requirements.txt
├─ src/
│  ├─ data.py                  # Data retrieval utilities (Statcast)
│  ├─ features.py              # Robust feature engineering for hitters/pitchers
│  ├─ models.py                # Simple modeling helpers (GBR for wOBA)
│  └─ utils.py                 # IO, chunking, logging
├─ scripts/
│  ├─ pull_statcast.py         # CLI to fetch statcast by month and save raw CSVs
│  ├─ build_features.py        # CLI to build season-level features
│  └─ train_model.py           # CLI to train a basic predictive model
├─ data/
│  ├─ raw/                     # Raw statcast CSVs
│  └─ processed/               # Features per season: hitters.csv, pitchers.csv
└─ tests/
   └─ test_features.py         # Basic sanity checks on feature outputs
```

## Notes & Tips

- **API etiquette**: `pybaseball` scrapes public endpoints. Use the built‑in caching and chunk by month to avoid timeouts.
- **Reproducibility**: This repo prefers deterministic aggregations, explicit filters (e.g., minimum PA/BF), and simple, explainable metrics.
- **Extensibility**: You can swap `team: NYY` for any team (e.g., `LAD`) or set `team: null` to pull league‑wide data.
- **Attribution**: MIT License. Cite `pybaseball` in your README if posting results.
- **Disclaimer**: For demo/portfolio purposes; not affiliated with MLB or the New York Yankees.


## Showcase Notebooks
- `notebooks/yankees_2024_rolling_xwoba_and_comps.ipynb` — Rolling xwOBA for top NYY hitters in 2024 + comparable-player search (cosine similarity).
- `notebooks/contract_valuation_toy_model.ipynb` — Aging curve (empirical, proxy) → wOBA projection → toy WAR + $ valuation with risk bands.

> These are portfolio demos; in production you would join player bios for true ages and use team-grade run-to-win and $/WAR mappings.

## Instant Demo (No Downloads)
If you just want to click around, the app will **fallback to sample data** in `sample_data/` when `data/processed/*.csv` are missing.


## One-click Deploy (Streamlit Cloud)

1. Push this repo to GitHub.
2. Go to **https://streamlit.io/cloud** → New app → pick your repo and `app.py`.
3. Set **Python version 3.11** and add the following secret only if needed (optional cache path):
   - `PYBASEBALL_CACHE_DIR`: `.pybaseball-cache`

> Badge (replace `USER/REPO` after you deploy):  
> `[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)``
