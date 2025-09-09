from typing import Tuple, Dict
import numpy as np
import pandas as pd

HITTER_ID = "batter"
PITCHER_ID = "pitcher"

# Helper: compute wOBA from statcast columns if present
def _compute_woba(df: pd.DataFrame) -> pd.Series:
    if "woba_value" in df.columns and "woba_denom" in df.columns:
        denom = df["woba_denom"].replace(0, np.nan)
        woba = df["woba_value"].sum() / denom.sum()
        return pd.Series({"woba": woba})
    return pd.Series({"woba": np.nan})

def _events_rate(df: pd.DataFrame, events: set, denom: float) -> float:
    if denom == 0:
        return np.nan
    mask = df["events"].isin(events) if "events" in df.columns else pd.Series(False, index=df.index)
    return mask.sum() / denom

def _csw_rate(df: pd.DataFrame) -> float:
    # CSW% = (called_strike + swinging_strike + swinging_strike_blocked) / total_pitches
    if "description" not in df.columns:
        return np.nan
    desc = df["description"].fillna("")
    csw = desc.isin(["called_strike", "swinging_strike", "swinging_strike_blocked"]).sum()
    total = len(df)
    return csw / total if total else np.nan

def _gb_rate(df: pd.DataFrame) -> float:
    # ground ball rate among balls in play
    if "bb_type" not in df.columns:
        return np.nan
    bip = df["bb_type"].notna().sum()
    if bip == 0:
        return np.nan
    return (df["bb_type"] == "ground_ball").sum() / bip

def build_hitter_features(statcast: pd.DataFrame, min_pa: int = 50) -> pd.DataFrame:
    if statcast.empty:
        return pd.DataFrame(columns=["player_id","player_name","season","pa","woba","xwoba","avg_ev","max_ev","k_rate","bb_rate","barrel_rate"])
    # Statcast often has 'player_name' for batters; ensure a stable name column
    if "player_name" not in statcast.columns and "batter_name" in statcast.columns:
        statcast["player_name"] = statcast["batter_name"]

    # Plate appearances approximated by sum of woba_denom if present
    statcast["woba_denom"] = statcast.get("woba_denom", 0)
    grouped = statcast.groupby([HITTER_ID, "player_name", "game_year"], dropna=False)

    rows = []
    for (pid, name, season), df in grouped:
        pa = df["woba_denom"].sum() if "woba_denom" in df.columns else len(df.dropna(subset=["events"]))  # fallback
        if pa < min_pa:
            continue

        # Core quality/contact metrics
        woba_s = _compute_woba(df)
        xwoba = df["estimated_woba_using_speedangle"].mean() if "estimated_woba_using_speedangle" in df.columns else np.nan
        avg_ev = df["launch_speed"].mean() if "launch_speed" in df.columns else np.nan
        max_ev = df["launch_speed"].max() if "launch_speed" in df.columns else np.nan

        # K% / BB% from 'events'
        k_rate = _events_rate(df, {"strikeout", "strikeout_double_play"}, denom=pa)
        bb_rate = _events_rate(df, {"walk", "hit_by_pitch", "intent_walk"}, denom=pa)

        # Barrel rate if available
        barrel_rate = df["barrel"].mean() if "barrel" in df.columns else np.nan

        rows.append({
            "player_id": pid,
            "player_name": name,
            "season": season,
            "pa": pa,
            "woba": woba_s["woba"],
            "xwoba": xwoba,
            "avg_ev": avg_ev,
            "max_ev": max_ev,
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            "barrel_rate": barrel_rate
        })

    return pd.DataFrame(rows).sort_values(["season","woba"], ascending=[True, False]).reset_index(drop=True)

def build_pitcher_features(statcast: pd.DataFrame, min_bf: int = 50) -> pd.DataFrame:
    if statcast.empty:
        return pd.DataFrame(columns=["player_id","player_name","season","bf","csw_rate","k_rate","bb_rate","gb_rate","ev_allowed"])
    if "player_name" not in statcast.columns and "pitcher_name" in statcast.columns:
        statcast["player_name"] = statcast["pitcher_name"]

    # Batter faced approximated by plate appearances against if woba_denom available; else fallback to events count
    statcast["woba_denom"] = statcast.get("woba_denom", 0)
    grouped = statcast.groupby([PITCHER_ID, "player_name", "game_year"], dropna=False)

    rows = []
    for (pid, name, season), df in grouped:
        bf = df["woba_denom"].sum() if "woba_denom" in df.columns else len(df.dropna(subset=["events"]))
        if bf < min_bf:
            continue

        csw = _csw_rate(df)
        k_rate = _events_rate(df, {"strikeout", "strikeout_double_play"}, denom=bf)
        bb_rate = _events_rate(df, {"walk", "intent_walk", "hit_by_pitch"}, denom=bf)
        gb = _gb_rate(df)
        ev_allowed = df["launch_speed"].mean() if "launch_speed" in df.columns else np.nan

        rows.append({
            "player_id": pid,
            "player_name": name,
            "season": season,
            "bf": bf,
            "csw_rate": csw,
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            "gb_rate": gb,
            "ev_allowed": ev_allowed
        })

    return pd.DataFrame(rows).sort_values(["season","csw_rate"], ascending=[True, False]).reset_index(drop=True)
