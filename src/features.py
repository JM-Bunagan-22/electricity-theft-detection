"""
Build a per-customer feature matrix from the long-format consumption data.

Electricity theft tends to show up as: consumption that suddenly drops
(a bypassed or tampered meter under-reports), stretches of exact zeros,
unusually erratic day-to-day swings, or a mismatch between weekday and
weekend usage. These features are designed to surface those patterns
without hard-coding the label.
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LONG_IN = os.path.join(DATA_DIR, "consumption_long.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "customer_features.csv")


def _longest_zero_streak(values: np.ndarray) -> int:
    is_zero = values == 0
    if not is_zero.any():
        return 0
    # index of each run's start via cumulative-sum trick
    change = np.diff(np.concatenate(([0], is_zero.astype(int), [0])))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    return int((ends - starts).max())


def _customer_features(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("date")
    values = g["kwh"].to_numpy(dtype=float)
    n = len(values)
    missing_mask = np.isnan(values)
    missing_rate = missing_mask.mean()

    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    filled = np.nan_to_num(filled, nan=0.0)

    mean = filled.mean()
    std = filled.std()
    cv = std / mean if mean > 0 else 0.0

    day_over_day = np.diff(filled)
    max_drop = -day_over_day.min() if n > 1 else 0.0
    pct_drop_days = (day_over_day < -0.5 * mean).mean() if mean > 0 and n > 1 else 0.0

    zero_rate = (filled == 0).mean()
    longest_zero_run = _longest_zero_streak(filled)

    is_weekend = g["date"].dt.dayofweek.to_numpy() >= 5
    weekday_mean = filled[~is_weekend].mean() if (~is_weekend).any() else 0.0
    weekend_mean = filled[is_weekend].mean() if is_weekend.any() else 0.0
    weekend_ratio = weekend_mean / weekday_mean if weekday_mean > 0 else 1.0

    t = np.arange(n)
    trend_slope = np.polyfit(t, filled, 1)[0] if n > 1 and filled.std() > 0 else 0.0

    half = n // 2
    first_half_mean = filled[:half].mean() if half > 0 else mean
    second_half_mean = filled[half:].mean() if half > 0 else mean
    recency_ratio = second_half_mean / first_half_mean if first_half_mean > 0 else 1.0

    return pd.Series(
        {
            "flag": g["flag"].iloc[0],
            "mean_kwh": mean,
            "std_kwh": std,
            "coeff_variation": cv,
            "missing_rate": missing_rate,
            "zero_rate": zero_rate,
            "longest_zero_run": longest_zero_run,
            "max_single_day_drop": max_drop,
            "pct_large_drop_days": pct_drop_days,
            "weekend_weekday_ratio": weekend_ratio,
            "trend_slope": trend_slope,
            "recency_ratio": recency_ratio,
        }
    )


def build_features() -> pd.DataFrame:
    print("Loading long-format consumption data...")
    long_df = pd.read_csv(LONG_IN, parse_dates=["date"])

    print(f"Engineering features for {long_df['customer_id'].nunique()} customers...")
    features = (
        long_df.groupby("customer_id", sort=False)
        .apply(_customer_features, include_groups=False)
        .reset_index()
    )
    features["flag"] = features["flag"].astype(int)

    features.to_csv(FEATURES_OUT, index=False)
    print(f"Saved feature matrix to {FEATURES_OUT} ({features.shape[0]} rows, {features.shape[1]} cols)")
    return features


if __name__ == "__main__":
    build_features()
