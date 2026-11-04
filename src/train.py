"""Train and compare classifiers for air quality prediction (UCI Air Quality dataset)."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "AirQualityUCI.xlsx"
MODEL_PATH = ROOT / "models" / "best_air_quality.pkl"
FIG_DIR = ROOT / "outputs" / "figures"
CO_THRESHOLD = 2.0  # CO(GT) >= 2.0 mg/m3 is labelled "bad air"


def load_and_clean() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    print("Rows before cleaning:", len(df))
    df = df.replace(-200, np.nan)          # -200 marks a failed sensor reading
    df = df.drop(columns=["NMHC(GT)"])     # mostly missing, so drop the column, not the rows
    df = df.dropna()
    print("Rows after cleaning:", len(df))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Hour"] = pd.to_datetime(df["Time"].astype(str), format="%H:%M:%S").dt.hour
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Bad_Air"] = (df["CO(GT)"] >= CO_THRESHOLD).astype(int)
    return df


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = add_features(load_and_clean())
    print(df["Bad_Air"].value_counts(normalize=True))

    X = df.drop(columns=["Date", "Time", "CO(GT)", "Bad_Air"])
    y = df["Bad_Air"]
    print(X.describe())

    plt.figure(figsize=(10, 8))
    sns.heatmap(X.assign(Bad_Air=y).corr(), cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png")
    plt.close()

    # Split first, then scale inside the pipeline (no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "CV_F1": cv_f1,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).round(4).sort_values("CV_F1", ascending=False)
    print(results)
    results.to_csv(ROOT / "outputs" / "model_comparison.csv", index=False)

    # Best model chosen by cross-validation, not by the test set
    best_name = results.iloc[0]["Model"]
    best = fitted[best_name]
    print("Best model:", best_name)

    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(FIG_DIR / "confusion_matrix.png")
    plt.close()

    inner = best.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=X.columns).sort_values()
        imp.plot(kind="barh", figsize=(8, 6), title="Feature Importance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance.png")
        plt.close()

    joblib.dump(best, MODEL_PATH)  # pipeline includes the scaler
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()