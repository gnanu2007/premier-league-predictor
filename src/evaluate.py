import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, log_loss,
                             confusion_matrix, ConfusionMatrixDisplay)

LABELS = ["A", "D", "H"]  # Away, Draw, Home


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=model.classes_)
    print(f"\n{name}\n  Accuracy: {acc:.3f}\n  Log loss: {ll:.3f}")

    cm = confusion_matrix(y_test, pred, labels=LABELS)
    print("  Confusion matrix (rows=true, cols=pred, order A/D/H):")
    print(cm)

    os.makedirs("results", exist_ok=True)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot()
    plt.title(name)
    plt.savefig(os.path.join("results", f"cm_{name.replace(' ', '_')}.png"))
    plt.close()
    return acc, ll
