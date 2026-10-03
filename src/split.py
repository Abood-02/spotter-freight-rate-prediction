"""
Data Splitting Module
Implements time-based train/validation splitting to avoid temporal lookahead leakage.
"""

from __future__ import annotations
import pandas as pd


def get_temporal_split(
    df: pd.DataFrame, split_date: str = "2025-09-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split development data by date threshold to simulate future out-of-time forecasting.
    
    Default split_date '2025-09-01' produces:
      - Train Fold: Months 1-8 (Jan 01, 2025 - Aug 31, 2025)
      - Validation Fold: Months 9-10 (Sep 01, 2025 - Oct 31, 2025)
    """
    dt_series = pd.to_datetime(df["date"])
    split_dt = pd.to_datetime(split_date)

    train_mask = dt_series < split_dt
    val_mask = dt_series >= split_dt

    train_split = df.loc[train_mask].copy().reset_index(drop=True)
    val_split = df.loc[val_mask].copy().reset_index(drop=True)

    return train_split, val_split
