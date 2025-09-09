import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_FEATURES = ["woba","xwoba","avg_ev","k_rate","bb_rate","barrel_rate"]

def top_comps(df: pd.DataFrame, player_name: str, season: int, k: int = 10, features = None) -> pd.DataFrame:
    """
    Return top k most similar hitters to `player_name` in `season` based on selected features (cosine similarity).
    """
    if features is None:
        features = DEFAULT_FEATURES
    f = df.dropna(subset=features + ["player_name","season"]).copy()
    f = f[f["season"] == season].copy()
    if f.empty:
        raise ValueError("No rows for the requested season.")
    X = f[features].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    sims = cosine_similarity(Xz)
    # Anchor index for the player
    mask = f["player_name"].str.lower() == player_name.lower()
    if not mask.any():
        raise ValueError(f"Player '{player_name}' not found in season {season}.")
    idx = np.where(mask.values)[0][0]
    scores = sims[idx]
    f = f.assign(similarity=scores)
    # Exclude the player themselves, sort by similarity
    out = f.loc[~mask, ["player_name","season","pa","woba","xwoba","avg_ev","k_rate","bb_rate","barrel_rate","similarity"]]\
           .sort_values("similarity", ascending=False).head(k).reset_index(drop=True)
    return out
