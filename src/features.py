# src/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

HITTER_ID = "batter"
PITCHER_ID = "pitcher"

# ----------------------------- utilities ------------------------------------ #

def _ensure_game_year(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'game_year' column derived from game_date if missing."""
    if "game_year" not in df.columns:
        df = df.copy()
        df["game_year"] = pd.to_datetime(df.get("game_date", pd.NaT), errors="coerce").dt.year
    return df

def _safe_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric safely (for wOBA fields etc.)."""
    return pd.to_numeric(s, errors="coerce")

# ---------------------------- hitter features -------------------------------- #

def build_hitter_features(statcast: pd.DataFrame, min_pa: int = 50) -> pd.DataFrame:
    """
    Vectorized hitter features per player-season.
    PA is max of three estimates:
      1) unique (game_pk, at_bat_number) with events
      2) sum(woba_denom)
      3) count of non-null events
    """
    cols = [
        "player_id","player_name","season","pa","woba","xwoba",
        "avg_ev","max_ev","k_rate","bb_rate","barrel_rate"
    ]
    if statcast is None or statcast.empty:
        return pd.DataFrame(columns=cols)

    sc = _ensure_game_year(statcast.copy())

    # Resolve batter id/name generously
    id_col = next((c for c in ["batter","batter_id","batter_pk","player_id"] if c in sc.columns), None)
    name_col = next((c for c in ["player_name","batter_name","batter_full_name"] if c in sc.columns), None)
    if name_col is None:
        if id_col is not None:
            sc["player_name"] = "Batter " + sc[id_col].astype(str)
        else:
            sc["player_name"] = "UNKNOWN"
        name_col = "player_name"
    else:
        sc[name_col] = sc[name_col].fillna("UNKNOWN")

    if id_col is None:
        sc["__pid__"] = sc[name_col].astype("category").cat.codes
        id_col = "__pid__"

    keys = [id_col, name_col, "game_year"]

    # Normalize
    if "woba_value" in sc.columns:
        sc["woba_value"] = _safe_numeric(sc["woba_value"]).fillna(0.0)
    if "woba_denom" in sc.columns:
        sc["woba_denom"] = _safe_numeric(sc["woba_denom"]).fillna(0.0)

    # ----- PA #1: AB-level unique (game_pk, at_bat_number) with events -----
    if {"game_pk","at_bat_number","events"}.issubset(sc.columns):
        ab = sc.loc[sc["events"].notna(), ["game_pk","at_bat_number","events"] + keys].drop_duplicates()
        pa1 = ab.groupby(keys, dropna=False).size().rename("pa_ab")
        k_cnt1  = ab.assign(_is_k=ab["events"].isin({"strikeout","strikeout_double_play"})) \
                   .groupby(keys, dropna=False)["_is_k"].sum().rename("K_cnt_ab")
        bb_cnt1 = ab.assign(_is_bb=ab["events"].isin({"walk","intent_walk","hit_by_pitch"})) \
                   .groupby(keys, dropna=False)["_is_bb"].sum().rename("BB_cnt_ab")
    else:
        pa1 = pd.Series(dtype="int64", name="pa_ab")
        k_cnt1 = pd.Series(dtype="int64", name="K_cnt_ab")
        bb_cnt1 = pd.Series(dtype="int64", name="BB_cnt_ab")

    # ----- PA #2: sum(woba_denom) -----
    if "woba_denom" in sc.columns:
        pa2 = sc.groupby(keys, dropna=False)["woba_denom"].sum().rename("pa_woba")
    else:
        pa2 = pd.Series(dtype="float64", name="pa_woba")

    # ----- PA #3: count of non-null events -----
    if "events" in sc.columns:
        pa3 = sc.groupby(keys, dropna=False)["events"].apply(lambda s: s.notna().sum()).rename("pa_evt")
        k_cnt3  = sc.groupby(keys, dropna=False)["events"].apply(
                    lambda s: s.isin({"strikeout","strikeout_double_play"}).sum()).rename("K_cnt_evt")
        bb_cnt3 = sc.groupby(keys, dropna=False)["events"].apply(
                    lambda s: s.isin({"walk","intent_walk","hit_by_pitch"}).sum()).rename("BB_cnt_evt")
    else:
        pa3 = pd.Series(dtype="int64", name="pa_evt")
        k_cnt3 = pd.Series(dtype="int64", name="K_cnt_evt")
        bb_cnt3 = pd.Series(dtype="int64", name="BB_cnt_evt")

    # Join PA estimates + choose max
    base = pd.concat([pa1, pa2, pa3], axis=1)
    base["pa"] = base.fillna(0)[["pa_ab","pa_woba","pa_evt"]].max(axis=1)

    # Aggregate wOBA bits, xwOBA/EV/Barrel
    if "woba_value" in sc.columns:
        base = base.join(sc.groupby(keys, dropna=False)["woba_value"].sum(), how="left")
    if "woba_denom" in sc.columns:
        base = base.join(sc.groupby(keys, dropna=False)["woba_denom"].sum(), how="left")
    # merge K/BB counts (prefer AB-based, fall back to event-based)
    base = base.join(k_cnt1, how="left").join(bb_cnt1, how="left")
    base = base.join(k_cnt3, how="left").join(bb_cnt3, how="left")
    base["K_cnt"]  = base[["K_cnt_ab","K_cnt_evt"]].fillna(0).max(axis=1)
    base["BB_cnt"] = base[["BB_cnt_ab","BB_cnt_evt"]].fillna(0).max(axis=1)

    if "estimated_woba_using_speedangle" in sc.columns:
        base = base.join(sc.groupby(keys, dropna=False)["estimated_woba_using_speedangle"].mean().rename("xwoba"), how="left")
    if "launch_speed" in sc.columns:
        g = sc.groupby(keys, dropna=False)["launch_speed"]
        base = base.join(g.mean().rename("avg_ev"), how="left")
        base = base.join(g.max().rename("max_ev"), how="left")
    if "barrel" in sc.columns:
        base = base.join(sc.groupby(keys, dropna=False)["barrel"].mean().rename("barrel_rate"), how="left")

    # Derived metrics
    denom = base["woba_denom"] if "woba_denom" in base.columns else pd.Series(0, index=base.index)
    wval  = base["woba_value"] if "woba_value" in base.columns else pd.Series(0, index=base.index)
    base["woba"] = np.where(denom.fillna(0) > 0, (wval.fillna(0) / denom.replace({0: np.nan})), np.nan)
    base["k_rate"]  = np.where(base["pa"].fillna(0) > 0, base["K_cnt"].fillna(0)  / base["pa"], np.nan)
    base["bb_rate"] = np.where(base["pa"].fillna(0) > 0, base["BB_cnt"].fillna(0) / base["pa"], np.nan)

    # Final tidy
    out = (
        base.reset_index()
            .rename(columns={"game_year":"season", id_col:"player_id", name_col:"player_name"})
    )
    for c in ["xwoba","avg_ev","max_ev","k_rate","bb_rate","barrel_rate"]:
        if c not in out.columns:
            out[c] = np.nan
    out = out[["player_id","player_name","season","pa","woba","xwoba","avg_ev","max_ev","k_rate","bb_rate","barrel_rate"]]

    # Filter + sort
    out = out[out["pa"].fillna(0).astype(int) >= int(min_pa)]
    if out.empty:
        return pd.DataFrame(columns=out.columns.tolist())
    return out.sort_values(["season","woba"], ascending=[True, False]).reset_index(drop=True)

# --------------------------- pitcher features -------------------------------- #

def _csw_components(desc: pd.Series) -> pd.Series:
    """Boolean mask for CSW events in pitch descriptions."""
    if desc is None:
        return pd.Series(dtype=bool)
    vals = {"called_strike", "swinging_strike", "swinging_strike_blocked"}
    return desc.isin(vals)

def build_pitcher_features(statcast: pd.DataFrame, min_bf: int = 50) -> pd.DataFrame:
    """
    Vectorized pitcher features per player-season.

    • BF: sum(woba_denom) when available; else count of rows with events.
    • CSW%: CSW events / total pitches.
    • K% / BB%: from 'events'.
    • GB%: grounders among BIP via bb_type.
    • EV allowed: mean launch_speed.
    """
    cols = ["player_id","player_name","season","bf","csw_rate","k_rate","bb_rate","gb_rate","ev_allowed"]
    if statcast is None or statcast.empty:
        return pd.DataFrame(columns=cols)

    sc = _ensure_game_year(statcast.copy())

    if "player_name" not in sc.columns:
        if "pitcher_name" in sc.columns:
            sc["player_name"] = sc["pitcher_name"]
        else:
            sc["player_name"] = "UNKNOWN"

    if "woba_denom" in sc.columns:
        sc["woba_denom"] = _safe_numeric(sc["woba_denom"]).fillna(0.0)

    keys = [PITCHER_ID, "player_name", "game_year"]
    g = sc.groupby(keys, dropna=False)

    # BF
    if "woba_denom" in sc.columns:
        bf = g["woba_denom"].sum().rename("bf")
    else:
        bf = g.apply(lambda d: d["events"].notna().sum() if "events" in d.columns else len(d)).rename("bf")

    # CSW%
    if "description" in sc.columns:
        csw = g["description"].apply(lambda s: _csw_components(s.fillna("")).sum()).rename("csw_cnt")
        tot_p = g.size().rename("pitch_cnt")
        csw_rate = (csw / tot_p.replace({0: np.nan})).rename("csw_rate")
    else:
        csw_rate = pd.Series(np.nan, index=bf.index, name="csw_rate")

    # K% / BB%
    def _group_event_rate(ev_set: set[str], name: str) -> pd.Series:
        if "events" not in sc.columns:
            return pd.Series(np.nan, index=bf.index, name=name)
        cnt = g["events"].apply(lambda s: s.isin(ev_set).sum())
        return (cnt / bf.replace({0: np.nan})).rename(name)

    k_rate  = _group_event_rate({"strikeout","strikeout_double_play"}, "k_rate")
    bb_rate = _group_event_rate({"walk","intent_walk","hit_by_pitch"}, "bb_rate")

    # GB% among BIP
    if "bb_type" in sc.columns:
        gb_cnt = g["bb_type"].apply(lambda s: (s == "ground_ball").sum())
        bip_cnt = g["bb_type"].apply(lambda s: s.notna().sum())
        gb_rate = (gb_cnt / bip_cnt.replace({0: np.nan})).rename("gb_rate")
    else:
        gb_rate = pd.Series(np.nan, index=bf.index, name="gb_rate")

    # EV allowed
    if "launch_speed" in sc.columns:
        ev_allowed = g["launch_speed"].mean().rename("ev_allowed")
    else:
        ev_allowed = pd.Series(np.nan, index=bf.index, name="ev_allowed")

    out = (
        bf.to_frame()
          .join([csw_rate, k_rate, bb_rate, gb_rate, ev_allowed], how="left")
          .reset_index()
          .rename(columns={"game_year":"season", PITCHER_ID:"player_id"})
    )

    out = out[["player_id","player_name","season","bf","csw_rate","k_rate","bb_rate","gb_rate","ev_allowed"]]
    out = out[out["bf"].fillna(0).astype(int) >= int(min_bf)]
    if out.empty:
        return pd.DataFrame(columns=out.columns.tolist())
    return out.sort_values(["season","csw_rate"], ascending=[True, False]).reset_index(drop=True)
