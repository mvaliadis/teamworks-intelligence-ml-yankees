import os, sys, yaml, itertools, calendar, datetime as dt
from pathlib import Path

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def month_ranges(year: int, start_month: int, end_month: int):
    for m in range(start_month, end_month + 1):
        start = dt.date(year, m, 1)
        end_day = calendar.monthrange(year, m)[1]
        end = dt.date(year, m, end_day)
        yield start.isoformat(), end.isoformat()

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None
