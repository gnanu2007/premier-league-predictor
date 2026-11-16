from collections import defaultdict, deque
import numpy as np
import pandas as pd

K, HOME_ADV, FORM_N = 20, 60, 5


def build_features(df):
    """Create Elo and recent-form features using only pre-match information."""
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=FORM_N))  # (points, gf, ga)

    def form(team):
        h = hist[team]
        if not h:
            return 1.3, 1.4, 1.4  # league-average defaults
        a = np.array(h)
        return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

    rows = []
    for r in df.itertuples():
        h, a = r.HomeTeam, r.AwayTeam
        hp, hgf, hga = form(h)
        ap, agf, aga = form(a)
        rows.append({
            "elo_diff": elo[h] - elo[a] + HOME_ADV,
            "home_pts": hp, "home_gf": hgf, "home_ga": hga,
            "away_pts": ap, "away_gf": agf, "away_ga": aga,
        })

        # update AFTER recording features
        exp_home = 1 / (1 + 10 ** ((elo[a] - elo[h] - HOME_ADV) / 400))
        score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[r.FTR]
        elo[h] += K * (score_home - exp_home)
        elo[a] -= K * (score_home - exp_home)
        hist[h].append(({"H": 3, "D": 1, "A": 0}[r.FTR], r.FTHG, r.FTAG))
        hist[a].append(({"H": 0, "D": 1, "A": 3}[r.FTR], r.FTAG, r.FTHG))

    return pd.DataFrame(rows), df["FTR"]

def build_match_features(df, home_team, away_team):
    """Build features for a future match using historical data."""

    if home_team == away_team:
        raise ValueError("Select two different teams.")

    teams = set(df["HomeTeam"]) | set(df["AwayTeam"])

    if home_team not in teams or away_team not in teams:
        raise ValueError("One or both teams were not found in the dataset.")

    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=FORM_N))

    def form(team):
        h = hist[team]
        if not h:
            return 1.3, 1.4, 1.4
        a = np.array(h)
        return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

    # Process historical matches in date order
    for r in df.sort_values("Date").itertuples():
        h, a = r.HomeTeam, r.AwayTeam

        exp_home = 1 / (
            1 + 10 ** ((elo[a] - elo[h] - HOME_ADV) / 400)
        )

        score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[r.FTR]

        elo[h] += K * (score_home - exp_home)
        elo[a] -= K * (score_home - exp_home)

        hist[h].append(
            ({"H": 3, "D": 1, "A": 0}[r.FTR], r.FTHG, r.FTAG)
        )
        hist[a].append(
            ({"H": 0, "D": 1, "A": 3}[r.FTR], r.FTAG, r.FTHG)
        )

    hp, hgf, hga = form(home_team)
    ap, agf, aga = form(away_team)

    return pd.DataFrame([{
        "elo_diff": elo[home_team] - elo[away_team] + HOME_ADV,
        "home_pts": hp,
        "home_gf": hgf,
        "home_ga": hga,
        "away_pts": ap,
        "away_gf": agf,
        "away_ga": aga,
    }])
