import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Baseball Leaderboard", layout="wide")

st.title("MLB Leaderboards (Portfolio Demo)")
st.write("Load processed hitters/pitchers features (from Statcast) and explore simple leaderboards.")

hitters_path = Path("data/processed/hitters_features.csv")
pitchers_path = Path("data/processed/pitchers_features.csv")
sample_hitters = Path("sample_data/hitters_features_sample.csv")
sample_pitchers = Path("sample_data/pitchers_features_sample.csv")

tabs = st.tabs(["Hitters", "Pitchers"])

with tabs[0]:
    if hitters_path.exists() or sample_hitters.exists():
        hitters = pd.read_csv(hitters_path if hitters_path.exists() else sample_hitters)
        sel_season = st.selectbox("Season", sorted(hitters["season"].unique()) if "season" in hitters.columns else [])
        df = hitters[hitters["season"] == sel_season].copy()
        metric = st.selectbox("Sort by", ["woba","xwoba","avg_ev","bb_rate","k_rate","barrel_rate"])
        st.dataframe(df.sort_values(metric, ascending=False).reset_index(drop=True))
    else:
        st.info("Run the pipeline to create data/processed/hitters_features.csv or use sample data (included)")

with tabs[1]:
    if pitchers_path.exists() or sample_pitchers.exists():
        pitchers = pd.read_csv(pitchers_path if pitchers_path.exists() else sample_pitchers)
        sel_season = st.selectbox("Season ", sorted(pitchers["season"].unique()) if "season" in pitchers.columns else [], key="p_season")
        df = pitchers[pitchers["season"] == sel_season].copy()
        metric = st.selectbox("Sort by ", ["csw_rate","k_rate","bb_rate","gb_rate","ev_allowed"], key="p_metric")
        ascending = st.checkbox("Ascending (lower is better)", value=False if metric!="ev_allowed" else True)
        st.dataframe(df.sort_values(metric, ascending=ascending).reset_index(drop=True))
    else:
        st.info("Run the pipeline to create data/processed/pitchers_features.csv or use sample data (included)")
