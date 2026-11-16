import os
import pandas as pd

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
RAW_DIR = os.path.join("data", "raw")


def load_data(seasons=SEASONS):
    """Download each season (or read the saved copy) and combine them."""
    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for s in seasons:
        path = os.path.join(RAW_DIR, f"E0_{s}.csv")
        if os.path.exists(path):
            raw = pd.read_csv(path)
        else:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
            raw = pd.read_csv(url)
            raw.to_csv(path, index=False)
        frames.append(raw[COLS].assign(season=s))

    df = pd.concat(frames).dropna()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)
