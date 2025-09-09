import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Baseball Leaderboard", layout="wide")

st.title("MLB Leaderboards (Portfolio Demo)")
st.write("Load processed hitters/pitchers features (from Statcast) and explore simple leaderboards.")

# Paths
hitters_path = Path("data/processed/hitters_features.csv")
pitchers_path = Path("data/processed/pitchers_features.csv")
sample_hitters = Path("sample_data/hitters_features_sample.csv")
sample_pitchers = Path("sample_data/pitchers_features_sample.csv")

# Sidebar filters
st.sidebar.header("Filters")
_ = st.sidebar.radio("View", ["Hitters","Pitchers"], index=0)  # keeps sidebar tidy
name_query = st.sidebar.text_input("Search player name")
min_pa = st.sidebar.slider("Minimum PA (hitters)", min_value=0, max_value=650, value=50, step=10)
min_bf = st.sidebar.slider("Minimum BF (pitchers)", min_value=0, max_value=800, value=50, step=10)

# Tabs
tabs = st.tabs(["Hitters", "Pitchers"])

with tabs[0]:
    if hitters_path.exists() or sample_hitters.exists():
        hitters = pd.read_csv(hitters_path if hitters_path.exists() else sample_hitters)
        seasons = sorted(hitters["season"].dropna().unique().tolist())
        sel_season = st.selectbox("Season", seasons, index=len(seasons)-1 if seasons else 0)
        df = hitters[hitters["season"] == sel_season].copy()
        if name_query:
            df = df[df["player_name"].str.contains(name_query, case=False, na=False)]
        df = df[df["pa"] >= min_pa]
        metric = st.selectbox("Sort by", ["woba","xwoba","avg_ev","bb_rate","k_rate","barrel_rate"])
        st.dataframe(df.sort_values(metric, ascending=False).reset_index(drop=True))
        # Quick chart
        st.markdown("#### xwOBA vs. Average Exit Velocity")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(df["avg_ev"], df["xwoba"])
        ax.set_xlabel("Average Exit Velocity")
        ax.set_ylabel("xwOBA")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        # Download
        st.download_button("Download table as CSV",
                           data=df.to_csv(index=False),
                           file_name=f"hitters_{sel_season}.csv",
                           mime="text/csv")
    else:
        st.info("Run the pipeline to create data/processed/hitters_features.csv or use sample data (included)")

with tabs[1]:
    if pitchers_path.exists() or sample_pitchers.exists():
        pitchers = pd.read_csv(pitchers_path if pitchers_path.exists() else sample_pitchers)
        seasons = sorted(pitchers["season"].dropna().unique().tolist())
        sel_season = st.selectbox("Season ", seasons, index=len(seasons)-1 if seasons else 0, key="p_season")
        df = pitchers[pitchers["season"] == sel_season].copy()
        if name_query:
            df = df[df["player_name"].str.contains(name_query, case=False, na=False)]
        df = df[df["bf"] >= min_bf] if "bf" in df.columns else df
        metric = st.selectbox("Sort by ", ["csw_rate","k_rate","bb_rate","gb_rate","ev_allowed"], key="p_metric")
        ascending = st.checkbox("Ascending (lower is better)", value=False if metric!="ev_allowed" else True)
        st.dataframe(df.sort_values(metric, ascending=ascending).reset_index(drop=True))
        # Quick chart
        st.markdown("#### CSW% vs. EV Allowed")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(df["ev_allowed"], df["csw_rate"])
        ax.set_xlabel("EV Allowed")
        ax.set_ylabel("CSW%")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.download_button("Download table as CSV",
                           data=df.to_csv(index=False),
                           file_name=f"pitchers_{sel_season}.csv",
                           mime="text/csv")
    else:
        st.info("Run the pipeline to create data/processed/pitchers_features.csv or use sample data (included)")
