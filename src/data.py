import os
from pathlib import Path
from typing import Optional, Tuple, Iterable

import pandas as pd
from tqdm import tqdm

# pybaseball imports kept inside functions so the file can be imported without the package pre-installed
def _import_pybaseball(cache_dir: Optional[str] = None):
    from pybaseball import statcast
    from pybaseball import cache
    if cache_dir:
        cache.enable()
        cache.cache_config['cache_dir'] = cache_dir
    return statcast

def pull_statcast_month(team: Optional[str], start_date: str, end_date: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    statcast = _import_pybaseball(cache_dir)
    # team can be None for league-wide
    df = statcast(start_dt=start_date, end_dt=end_date, team=team)
    # Normalize column names a bit
    return df

def pull_statcast_season(team: Optional[str], year: int, month_ranges: Iterable[Tuple[str,str]], cache_dir: Optional[str] = None) -> pd.DataFrame:
    parts = []
    for start, end in tqdm(list(month_ranges), desc=f"Pulling {year}"):
        df = pull_statcast_month(team=team, start_date=start, end_date=end, cache_dir=cache_dir)
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Drop duplicated pitch rows if any
    out = out.drop_duplicates(subset=[c for c in out.columns if c != "release_spin_rate"]).reset_index(drop=True)
    return out
