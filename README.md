# Yankees (MLB) – End-to-End Sports Analytics (Statcast)

<p align="left">
  <!-- Replace USER/REPO after pushing -->
  <a href="https://github.com/USER/REPO/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
</p>

A compact, reproducible baseball analytics project.  
Pull MLB Statcast data (via [`pybaseball`](https://github.com/jldbc/pybaseball)), build hitter/pitcher features, evaluate a simple model, and explore leaderboards in a Streamlit app.

> **Why this repo?** Practical, production-minded DS for sports: robust ingest, vectorized features on noisy pitch-level data, a toy model with CV + feature importance, and a small app—cleanly organized and easy to extend.

---

## What’s inside

- **Ingest → Features → Model → App** pipeline driven by `config.yaml`.
- **Vectorized feature builders** (fast & robust):
  - Hitters: PA (AB/wOBA/event fallbacks), wOBA/xwOBA, EV (mean/max), K%, BB%, Barrel%.
  - Pitchers: BF, CSW%, K%, BB%, GB%, EV allowed.
- **Toy ML evaluation** (hitters): Gradient Boosting, grouped CV, permutation importance.
- **Interactive app**:
  - Sortable leaderboards
  - X/Y **axis selectors** (e.g., xwOBA vs wOBA), **season filter**, and **player name labels** toggle
  - **Model** tab: baseline vs model metrics + feature importance
- **Artifacts**: `data/processed/model_report.json`, `feature_importance.csv`
- **Deployment-friendly**: Dockerfile, CI stub, `.gitignore` for raw data.

**Example run** (league-wide 2022–2024, lenient thresholds)
- Hitters: **1,322** rows
- Pitchers: **2,435** rows
- Model (within-season): **R² 0.558** vs baseline **0.479**, **MAE 0.086** vs 0.093  
*(Your scores will vary by config.)*

---

## Quickstart

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Configure
cp config.example.yaml config.yaml   # edit as needed (team/years/thresholds)

# 3) Pull Statcast -> data/raw/
python scripts/pull_statcast.py --config config.yaml

# 4) Build features -> data/processed/
python scripts/build_features.py --config config.yaml

# 5) (Optional) Train a simple model (saves model + report)
# Windows:
set PYTHONPATH=.
python scripts\train_model.py --config config.yaml
# macOS/Linux:
PYTHONPATH=. python scripts/train_model.py --config config.yaml

# 6) Explore the app
streamlit run app.py
```

## Repo Layout

```
.
├─ app.py                      # Streamlit app (Hitters, Pitchers, Model tabs)
├─ config.example.yaml         # Template config (copy to config.yaml)
├─ requirements.txt
├─ src/
│  ├─ data.py                  # Statcast retrieval (month chunks, caching)
│  ├─ features.py              # Vectorized feature engineering (robust)
│  ├─ models.py                # Season-shift helper + utilities
│  └─ utils.py                 # IO, config, ensure_dir
├─ scripts/
│  ├─ pull_statcast.py         # Fetch statcast and write data/raw/*.csv
│  ├─ build_features.py        # Build hitters/pitchers features to data/processed/
│  ├─ evaluate_model.py        # Evaluate toy model, write report + importances
│  └─ train_model.py           # Train + save a model artifact (model.pkl, cols.json)
├─ data/
│  ├─ raw/                     # (gitignored) month/season CSVs
│  └─ processed/               # Features + model artifacts
├─ sample_data/                # Small fallback CSVs for instant demo
└─ tests/
   └─ test_features.py

```

## Notes & Tips

- **API etiquette**: `pybaseball` scrapes public endpoints. Use the built‑in caching and chunk by month to avoid timeouts.
- **Reproducibility**: This repo prefers deterministic aggregations, explicit filters (e.g., minimum PA/BF), and simple, explainable metrics.
- **Speed**: Feature building is vectorized; we read only needed columns with usecols.
- **R² warnings**: If dataset is tiny, CV may issue warnings; use league-wide multi-year and low thresholds while testing.
- **Windows import path**: set `PYTHONPATH=.` before running `evaluate_model.py/train_model.py`.
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
