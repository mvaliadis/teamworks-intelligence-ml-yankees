import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLB Statcast – Yankees / MLB", layout="wide")

DATA_DIR = Path("data/processed")
HIT_PATH = DATA_DIR / "hitters_features.csv"
PIT_PATH = DATA_DIR / "pitchers_features.csv"
REP_PATH = DATA_DIR / "model_report.json"
FI_PATH  = DATA_DIR / "feature_importance.csv"

@st.cache_data
def load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        samp = Path("sample_data") / p.name
        if samp.exists():
            df = pd.read_csv(samp)
        else:
            return pd.DataFrame()
    else:
        df = pd.read_csv(p)
    # ensure season is int (no 2,023 formatting)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df

hitters = load_csv(HIT_PATH)
pitchers = load_csv(PIT_PATH)

st.title("MLB Statcast – End-to-End Sports Analytics")

tabs = st.tabs(["Hitters", "Pitchers", "Model"])

# --------------------------- HIT TERS TAB --------------------------- #
with tabs[0]:
    st.subheader("Hitters")

    if hitters.empty:
        st.info("No hitters_features.csv found. Run the pipeline or use sample_data.")
    else:
        # Filters
        seasons = sorted([int(s) for s in hitters["season"].dropna().unique().tolist()])
        sel_seasons = st.multiselect("Season(s)", seasons, default=seasons)
        df = hitters[hitters["season"].isin(sel_seasons)] if sel_seasons else hitters

        left, right = st.columns([2, 1])

        with left:
            st.markdown("**Leaderboard**")
            st.dataframe(
                df.sort_values(["season","woba"], ascending=[True, False])
                  .reset_index(drop=True)
            )

        with right:
            st.markdown("**Download**")
            st.download_button("CSV (filtered)", df.to_csv(index=False).encode("utf-8"),
                               file_name="hitters_filtered.csv", mime="text/csv")

        st.divider()

        st.markdown("### Interactive scatter")
        # Axis options
        options = {
            "wOBA": "woba",
            "xwOBA": "xwoba",
            "Avg EV": "avg_ev",
            "Max EV": "max_ev",
            "K%": "k_rate",
            "BB%": "bb_rate",
            "Barrel%": "barrel_rate",
            "PA": "pa",
        }
        c1, c2, c3 = st.columns([1,1,1])
        x_key = c1.selectbox("X axis", list(options.keys()), index=1)   # default xwOBA
        y_key = c2.selectbox("Y axis", list(options.keys()), index=0)   # default wOBA
        label_on = c3.checkbox("Show player name labels", value=False)
        top_n = c3.number_input("Label top N by Y", min_value=1, max_value=100, value=20)

        plot_df = df.copy()
        # guard: ensure numeric
        for col in ["woba","xwoba","avg_ev","max_ev","k_rate","bb_rate","barrel_rate","pa"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        base = alt.Chart(plot_df).mark_circle(size=60).encode(
            x=alt.X(options[x_key], title=x_key),
            y=alt.Y(options[y_key], title=y_key),
            color=alt.Color("season:N", title="Season"),
            tooltip=[
                alt.Tooltip("player_name:N", title="Player"),
                alt.Tooltip("season:N", title="Season"),
                alt.Tooltip(options[x_key] + ":Q", title=x_key, format=".3f"),
                alt.Tooltip(options[y_key] + ":Q", title=y_key, format=".3f"),
                alt.Tooltip("pa:Q", title="PA", format="d"),
            ]
        )

        layers = [base]

        if label_on and len(plot_df):
            # label top N by Y
            ycol = options[y_key]
            top = plot_df.sort_values(ycol, ascending=False).head(int(top_n))
            text = alt.Chart(top).mark_text(align="left", dx=7, dy=3).encode(
                x=alt.X(options[x_key]),
                y=alt.Y(ycol),
                text="player_name:N",
                color=alt.Color("season:N", legend=None)
            )
            layers.append(text)

        st.altair_chart(alt.layer(*layers).interactive().properties(height=420), use_container_width=True)

# --------------------------- PITCHERS TAB --------------------------- #
with tabs[1]:
    st.subheader("Pitchers")

    if pitchers.empty:
        st.info("No pitchers_features.csv found. Run the pipeline or use sample_data.")
    else:
        seasons = sorted([int(s) for s in pitchers["season"].dropna().unique().tolist()])
        sel_seasons = st.multiselect("Season(s)", seasons, default=seasons, key="pit_seasons")
        df = pitchers[pitchers["season"].isin(sel_seasons)] if sel_seasons else pitchers

        left, right = st.columns([2, 1])

        with left:
            st.markdown("**Leaderboard**")
            st.dataframe(
                df.sort_values(["season","csw_rate"], ascending=[True, False])
                  .reset_index(drop=True)
            )

        with right:
            st.markdown("**Download**")
            st.download_button("CSV (filtered)", df.to_csv(index=False).encode("utf-8"),
                               file_name="pitchers_filtered.csv", mime="text/csv")

        st.divider()

        st.markdown("### Interactive scatter")
        options = {
            "CSW%": "csw_rate",
            "K%": "k_rate",
            "BB%": "bb_rate",
            "GB%": "gb_rate",
            "EV allowed": "ev_allowed",
            "BF": "bf",
        }
        c1, c2, c3 = st.columns([1,1,1])
        x_key = c1.selectbox("X axis", list(options.keys()), index=0)   # CSW%
        y_key = c2.selectbox("Y axis", list(options.keys()), index=1)   # K%
        label_on = c3.checkbox("Show player name labels", value=False, key="pit_labels")
        top_n = c3.number_input("Label top N by Y", min_value=1, max_value=100, value=20, key="pit_topn")

        plot_df = df.copy()
        for col in ["csw_rate","k_rate","bb_rate","gb_rate","ev_allowed","bf"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        base = alt.Chart(plot_df).mark_circle(size=60).encode(
            x=alt.X(options[x_key], title=x_key),
            y=alt.Y(options[y_key], title=y_key),
            color=alt.Color("season:N", title="Season"),
            tooltip=[
                alt.Tooltip("player_name:N", title="Player"),
                alt.Tooltip("season:N", title="Season"),
                alt.Tooltip(options[x_key] + ":Q", title=x_key, format=".3f"),
                alt.Tooltip(options[y_key] + ":Q", title=y_key, format=".3f"),
                alt.Tooltip("bf:Q", title="BF", format="d"),
            ]
        )

        layers = [base]
        if label_on and len(plot_df):
            ycol = options[y_key]
            top = plot_df.sort_values(ycol, ascending=False).head(int(top_n))
            text = alt.Chart(top).mark_text(align="left", dx=7, dy=3).encode(
                x=alt.X(options[x_key]),
                y=alt.Y(ycol),
                text="player_name:N",
                color=alt.Color("season:N", legend=None)
            )
            layers.append(text)

        st.altair_chart(alt.layer(*layers).interactive().properties(height=420), use_container_width=True)

# --------------------------- MODEL TAB --------------------------- #
with tabs[2]:
    st.subheader("Model Evaluation (Hitters)")
    if REP_PATH.exists() and FI_PATH.exists():
        with open(REP_PATH) as f:
            rep = json.load(f)

        # Defensive formatting (avoid NaN weirdness)
        def fmt(x, digits=3):
            try:
                return f"{float(x):.{digits}f}"
            except Exception:
                return "—"

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline R²", fmt(rep.get("baseline", {}).get("r2")))
        c2.metric("Model R²", fmt(rep.get("model", {}).get("r2")))
        c3.metric("Model MAE", fmt(rep.get("model", {}).get("mae")))

        st.caption(f"CV: {rep.get('cv','')} • mode: {rep.get('mode','')} • n={rep.get('n_samples','')}")

        fi = pd.read_csv(FI_PATH)
        st.markdown("#### Permutation Importance")
        fig, ax = plt.subplots(figsize=(6,3.5))
        if "feature" in fi.columns and "importance_mean" in fi.columns:
            ax.bar(fi["feature"], fi["importance_mean"])
            ax.set_ylabel("Mean importance")
            ax.set_xticklabels(fi["feature"], rotation=30, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
            ax.axis("off")
        st.pyplot(fig)

        st.markdown("Artifacts:")
        st.code(str(REP_PATH), language="text")
        st.code(str(FI_PATH),  language="text")
    else:
        st.info("No model artifacts found. Run: `python scripts/evaluate_model.py` or `python scripts/train_model.py`.")