"""Feature pipeline for the video game price prediction dataset.

Loads the raw `video_game_reviews.csv`, drops identifier columns,
imputes missing values, removes outliers via IQR, and one-hot
encodes categoricals. Returns a model-ready (X, y) pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DROP_COLS = ["Game Title", "Release Year", "Developer", "Publisher", "User Review Text"]
GRAPHICS_MAP = {"Low": 0, "Medium": 1, "High": 2, "Ultra": 3}
RATING_MAP = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
ORDINAL_MAPS = {
    "Graphics Quality": GRAPHICS_MAP,
    "Soundtrack Quality": RATING_MAP,
    "Story Quality": RATING_MAP,
}
ONE_HOT_COLS = ["Game Mode", "Multiplayer", "Age Group Targeted", "Requires Special Device", "Platform", "Genre"]
NUMERIC_COLS = ["User Rating", "Game Length (Hours)", "Min Number of Players"]
TARGET = "Price"


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median for numerics, mode for categoricals. No-op on a clean frame."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: Iterable[str], k: float = 3.0) -> pd.DataFrame:
    """Drop rows where any column in `cols` is outside [Q1 - k*IQR, Q3 + k*IQR].

    Uses k=3 by default (Tukey's "far out" threshold) so a uniformly
    distributed feature is not aggressively pruned.
    """
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask &= df[col].between(lo, hi)
    return df.loc[mask].reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        mapped = df[col].map(mapping)
        if mapped.isna().any():
            unknown = sorted(set(df.loc[mapped.isna(), col].unique()))
            raise ValueError(f"Unknown values in {col!r}: {unknown}. Update ORDINAL_MAPS.")
        df[col] = mapped.astype(int)
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)
    return df


def build_features(
    path: str | Path,
    *,
    drop_outliers: bool = True,
    outlier_cols: Iterable[str] = (TARGET, *NUMERIC_COLS),
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """End-to-end feature pipeline. Returns (X, y, stats)."""
    df = load_raw(path)
    n0 = len(df)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = impute_missing(df)

    n_imputed = n0  # placeholder; full audit could log per-col counts
    if drop_outliers:
        df = remove_outliers_iqr(df, outlier_cols)
    n1 = len(df)

    df = encode(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    stats = {"rows_in": n0, "rows_out": n1, "rows_dropped_as_outliers": n0 - n1, "n_features": X.shape[1]}
    return X, y, stats
