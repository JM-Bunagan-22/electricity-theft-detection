"""
Compare a supervised classifier (which uses the ground-truth theft labels)
against an unsupervised anomaly detector (which doesn't) on the SGCC
dataset, and report how well each recovers the known theft cases.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_IN = os.path.join(DATA_DIR, "customer_features.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "scored_customers.csv")
METRICS_OUT = os.path.join(DATA_DIR, "metrics.json")
MODEL_OUT = os.path.join(DATA_DIR, "theft_model.joblib")

FEATURE_COLS = [
    "mean_kwh",
    "std_kwh",
    "coeff_variation",
    "missing_rate",
    "zero_rate",
    "longest_zero_run",
    "max_single_day_drop",
    "pct_large_drop_days",
    "weekend_weekday_ratio",
    "trend_slope",
    "recency_ratio",
]


def run():
    df = pd.read_csv(FEATURES_IN)
    X = df[FEATURE_COLS]
    y = df["flag"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, stratify=y, random_state=42
    )

    # --- Supervised: Random Forest trained on the labels ---
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    rf_scores_test = clf.predict_proba(X_test)[:, 1]

    rf_roc_auc = roc_auc_score(y_test, rf_scores_test)
    rf_pr_auc = average_precision_score(y_test, rf_scores_test)
    precision, recall, thresholds = precision_recall_curve(y_test, rf_scores_test)
    f1_scores = 2 * precision * recall / np.where(precision + recall == 0, 1, precision + recall)
    best_idx = f1_scores[:-1].argmax()
    best_threshold = thresholds[best_idx]
    rf_preds_at_best = (rf_scores_test >= best_threshold).astype(int)
    report = classification_report(y_test, rf_preds_at_best, target_names=["normal", "theft"], output_dict=True)

    # --- Unsupervised baseline: Isolation Forest, never sees y ---
    contamination = y_train.mean()
    iso = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1
    )
    iso.fit(X_train)
    # more negative score = more anomalous; flip sign so higher = riskier
    iso_scores_test = -iso.score_samples(X_test)
    iso_roc_auc = roc_auc_score(y_test, iso_scores_test)
    iso_pr_auc = average_precision_score(y_test, iso_scores_test)

    metrics = {
        "n_customers": int(len(df)),
        "n_theft": int(y.sum()),
        "theft_rate": float(y.mean()),
        "test_size": int(len(y_test)),
        "random_forest": {
            "roc_auc": float(rf_roc_auc),
            "pr_auc": float(rf_pr_auc),
            "best_threshold": float(best_threshold),
            "precision_theft": float(report["theft"]["precision"]),
            "recall_theft": float(report["theft"]["recall"]),
            "f1_theft": float(report["theft"]["f1-score"]),
        },
        "isolation_forest": {
            "roc_auc": float(iso_roc_auc),
            "pr_auc": float(iso_pr_auc),
        },
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLS, clf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Score every customer with the trained RF for the dashboard, keeping
    # track of which rows were held out of training (test set).
    all_scores = clf.predict_proba(X)[:, 1]
    scored = df[["customer_id", "flag"] + FEATURE_COLS].copy()
    scored["risk_score"] = all_scores
    scored["in_test_set"] = df.index.isin(idx_test)
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored.to_csv(RESULTS_OUT, index=False)

    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved scored customers to {RESULTS_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    return metrics


if __name__ == "__main__":
    run()
