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
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier

def _unsupervised_score_features(X_train: pd.DataFrame, X_all: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """Fit unsupervised outlier detectors on the training rows only, then score every
    customer with each. These scores become extra input features for XGBOD below --
    the ADBench benchmark run (see adbench-electricity-theft) found this combination
    of unsupervised outlier scores + gradient boosting beats a plain supervised model."""
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_all)

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination, n_jobs=-1)
    lof.fit(X_train)
    lof_scores = -lof.score_samples(X_all)

    return pd.DataFrame({"iso_outlier_score": iso_scores, "lof_outlier_score": lof_scores}, index=X_all.index)


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

    # --- XGBOD-style: unsupervised outlier scores as extra features for XGBoost ---
    # (Zhao & Hryniewicki 2018; the winning approach in the ADBench benchmark run
    # against this same dataset -- see adbench-electricity-theft.)
    unsup_scores_all = _unsupervised_score_features(X_train, X, contamination)
    X_train_aug = pd.concat([X_train, unsup_scores_all.loc[X_train.index]], axis=1)
    X_test_aug = pd.concat([X_test, unsup_scores_all.loc[X_test.index]], axis=1)
    X_all_aug = pd.concat([X, unsup_scores_all], axis=1)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgbod = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgbod.fit(X_train_aug, y_train)
    xgbod_scores_test = xgbod.predict_proba(X_test_aug)[:, 1]
    xgbod_roc_auc = roc_auc_score(y_test, xgbod_scores_test)
    xgbod_pr_auc = average_precision_score(y_test, xgbod_scores_test)

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
        "xgbod": {
            "roc_auc": float(xgbod_roc_auc),
            "pr_auc": float(xgbod_pr_auc),
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

    # Score every customer with both models for the dashboard, keeping track
    # of which rows were held out of training (test set).
    all_scores = clf.predict_proba(X)[:, 1]
    xgbod_all_scores = xgbod.predict_proba(X_all_aug)[:, 1]
    scored = df[["customer_id", "flag"] + FEATURE_COLS].copy()
    scored["risk_score"] = all_scores
    scored["xgbod_risk_score"] = xgbod_all_scores
    scored["in_test_set"] = df.index.isin(idx_test)
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored.to_csv(RESULTS_OUT, index=False)

    joblib.dump(clf, MODEL_OUT)
    joblib.dump(xgbod, os.path.join(DATA_DIR, "xgbod_model.joblib"))
    print(f"\nSaved scored customers to {RESULTS_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    return metrics


if __name__ == "__main__":
    run()
