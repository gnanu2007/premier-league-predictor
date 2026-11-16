import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.features import build_features
from src.evaluate import evaluate


def main():
    df = load_data()
    X, y = build_features(df)
    X.assign(FTR=y).to_csv("data/processed/features.csv", index=False)

    # Time-based split: train on older seasons, test on the latest
    test_season = df["season"].max()
    train = (df["season"] != test_season).values
    X_train, X_test, y_train, y_test = X[train], X[~train], y[train], y[~train]

    baseline = y_train.mode()[0]
    print(f"Baseline (always '{baseline}'): "
          f"{accuracy_score(y_test, [baseline] * len(y_test)):.3f}")

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05),
    }

    best_name, best_ll, best_model = None, float("inf"), None
    for name, model in models.items():
        model.fit(X_train, y_train)
        _, ll = evaluate(name, model, X_test, y_test)
        if ll < best_ll:
            best_name, best_ll, best_model = name, ll, model

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    print(f"\nSaved best model ({best_name}) to models/model.pkl")


if __name__ == "__main__":
    main()
