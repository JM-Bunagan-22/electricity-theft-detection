"""
Download and prepare the SGCC (State Grid Corporation of China) electricity
theft detection dataset: daily consumption for 42,372 customers over
2014-01-01 to 2016-10-31, labeled 0 (normal) / 1 (theft).

Source: https://github.com/henryRDlab/ElectricityTheftDetection
(companion data for Zheng et al., "Wide and Deep Convolutional Neural
Networks for Electricity-Theft Detection to Secure Smart Grids", IEEE
Trans. Industrial Informatics, 2018.)

The dataset ships as a 3-part split zip (data.zip, data.z01, data.z02)
that must be joined before extraction. Joining requires 7z (p7zip-full)
since Python's zipfile and Info-ZIP's `zip -s0` both reject this
archive's local headers as a false-positive "overlapping entries" zip
bomb check.
"""

import os
import shutil
import subprocess

import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/henryRDlab/ElectricityTheftDetection/master"
PARTS = ["data.z01", "data.z02", "data.zip"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV = os.path.join(DATA_DIR, "sgcc_raw.csv")
LONG_OUT = os.path.join(DATA_DIR, "consumption_long.csv")


def _check_7z():
    if shutil.which("7z") is None:
        raise RuntimeError(
            "7z is required to extract the multi-part archive. "
            "Install it with: apt-get install -y p7zip-full"
        )


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        print("Raw data already present, skipping download.")
        return

    _check_7z()
    print("Downloading dataset parts (~52 MB total)...")
    for part in PARTS:
        dest = os.path.join(DATA_DIR, part)
        resp = requests.get(f"{RAW_BASE}/{part}", timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"  downloaded {part} ({len(resp.content) / 1e6:.1f} MB)")

    print("Extracting joined archive with 7z...")
    subprocess.run(
        ["7z", "x", "-y", os.path.join(DATA_DIR, "data.zip")],
        cwd=DATA_DIR,
        check=True,
    )
    os.rename(os.path.join(DATA_DIR, "data.csv"), RAW_CSV)
    for part in PARTS:
        os.remove(os.path.join(DATA_DIR, part))
    print(f"Saved raw dataset to {RAW_CSV}")


def load_and_clean():
    """Load the wide raw file (one row per customer, one column per date)
    and reshape to a long (customer, date, consumption) table with dates
    in chronological order."""
    print("Loading raw data (this file is large, may take a minute)...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df = df.rename(columns={"CONS_NO": "customer_id", "FLAG": "flag"})

    date_cols = [c for c in df.columns if c not in ("customer_id", "flag")]

    long_df = df.melt(
        id_vars=["customer_id", "flag"],
        value_vars=date_cols,
        var_name="date",
        value_name="kwh",
    )
    long_df["date"] = pd.to_datetime(long_df["date"], format="%Y/%m/%d")
    long_df = long_df.sort_values(["customer_id", "date"]).reset_index(drop=True)

    long_df.to_csv(LONG_OUT, index=False)
    n_customers = long_df["customer_id"].nunique()
    n_theft = df["flag"].sum()
    print(
        f"Saved long-format consumption data to {LONG_OUT} "
        f"({n_customers} customers, {n_theft} flagged theft = "
        f"{n_theft / n_customers:.1%})"
    )
    return long_df


if __name__ == "__main__":
    download_data()
    load_and_clean()
