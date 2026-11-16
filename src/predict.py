
import joblib

from src.data_loader import load_data
from src.features import build_match_features


def predict_match(home_team, away_team):
    df = load_data()

    # Get available team names
    teams = sorted(
        set(df["HomeTeam"]) | set(df["AwayTeam"])
    )

    # Match names without case sensitivity
    team_lookup = {
        team.lower(): team for team in teams
    }

    home_team = team_lookup.get(home_team.strip().lower())
    away_team = team_lookup.get(away_team.strip().lower())

    if not home_team or not away_team:
        print("\nInvalid team name!")
        print("Available teams:")
        print(", ".join(teams))
        return

    if home_team == away_team:
        print("Please select two different teams.")
        return

    # Load trained model
    model = joblib.load("models/model.pkl")

    # Build match features
    X = build_match_features(df, home_team, away_team)

    # Predict result
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    result_names = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    }

    print("\n===== PREMIER LEAGUE PREDICTION =====")
    print(f"Home Team: {home_team}")
    print(f"Away Team: {away_team}")
    print(f"Predicted Result: {result_names[prediction]}")

    print("\nPrediction Probabilities:")

    for result, probability in zip(model.classes_, probabilities):
        print(
            f"{result_names[result]}: {probability * 100:.2f}%"
        )


if __name__ == "__main__":
    home = input("Enter Home Team: ")
    away = input("Enter Away Team: ")

    predict_match(home, away)
