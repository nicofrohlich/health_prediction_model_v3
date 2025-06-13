import pandas as pd
import numpy as np

def extract_features(row):
    scores = [row[f'ews_w_day_{i + 1}'] for i in range(7)]
    return pd.Series({
        "media": np.mean(scores),
        "mediana": np.median(scores),
        "desvio": np.std(scores),
        "trend": scores[-1] - scores[0],
        "escore_max": max(scores),
        "escore_min": min(scores)
    })