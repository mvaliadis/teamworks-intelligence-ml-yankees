import pandas as pd
from src.features import build_hitter_features, build_pitcher_features

def test_empty_inputs():
    import pandas as pd
    empty = pd.DataFrame()
    h = build_hitter_features(empty, min_pa=50)
    p = build_pitcher_features(empty, min_bf=50)
    assert h.empty and p.empty
