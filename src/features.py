"""
Feature Engineering Module
Constructs geospatial, temporal, interaction, and categorical signals.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate the great-circle distance between two points on Earth in miles."""
    r_miles = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r_miles * c


class FeatureEngineer:
    """Extracts domain-driven features for freight pricing models."""

    def __init__(self) -> None:
        self.pickup_encoder: dict[str, int] = {}
        self.delivery_encoder: dict[str, int] = {}
        self.equipment_encoder: dict[str, int] = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
        self.feature_columns: list[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn categorical vocabularies from training data."""
        unique_pickups = sorted(df["pickup"].astype(str).unique())
        self.pickup_encoder = {city: i for i, city in enumerate(unique_pickups)}

        unique_deliveries = sorted(df["delivery"].astype(str).unique())
        self.delivery_encoder = {city: i for i, city in enumerate(unique_deliveries)}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer geospatial, temporal, and interaction features."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted on training data before transform.")

        out = df.copy()
        date = pd.to_datetime(out["date"])

        # 1. Domain Baseline Quote ($)
        out["base_quote"] = out["quote_signal"] * out["distance"]

        # 2. Geospatial Features
        p_lat = out["pickup_lat"].to_numpy(dtype=float)
        p_lon = out["pickup_lon"].to_numpy(dtype=float)
        d_lat = out["delivery_lat"].to_numpy(dtype=float)
        d_lon = out["delivery_lon"].to_numpy(dtype=float)

        hav_dist = haversine_np(p_lat, p_lon, d_lat, d_lon)
        out["haversine_dist"] = hav_dist
        out["tortuosity"] = out["distance"] / (hav_dist + 1.0)
        out["lat_diff"] = np.abs(d_lat - p_lat)
        out["lon_diff"] = np.abs(d_lon - p_lon)

        # 3. Calendar & Seasonality Features
        out["dayofweek"] = date.dt.dayofweek
        out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
        out["day"] = date.dt.day
        out["month"] = date.dt.month
        out["dayofyear"] = date.dt.dayofyear
        out["is_month_end"] = (out["day"] >= 27).astype(int)
        out["is_q4"] = out["month"].isin([10, 11, 12]).astype(int)
        
        # Freight December Holiday Crunch (Dec 15 - Dec 24)
        out["is_holiday_rush"] = ((out["month"] == 12) & (out["day"].between(15, 24))).astype(int)
        # Post-Christmas Holiday Freeze (Dec 25 - Dec 31)
        out["is_post_holiday"] = ((out["month"] == 12) & (out["day"] >= 25)).astype(int)

        # 4. Weight Dynamics
        out["weight_per_mile"] = out["weight"] / (out["distance"] + 1.0)
        out["is_heavy_load"] = (out["weight"] >= 38000.0).astype(int)

        # 5. Categorical Encoding
        out["equipment_code"] = out["equipment"].map(self.equipment_encoder).fillna(-1).astype(int)
        out["pickup_code"] = out["pickup"].map(self.pickup_encoder).fillna(-1).astype(int)
        out["delivery_code"] = out["delivery"].map(self.delivery_encoder).fillna(-1).astype(int)

        # Keep record of feature columns
        self.feature_columns = [
            "distance", "weight", "market_index", "quote_signal", "base_quote",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "haversine_dist", "tortuosity", "lat_diff", "lon_diff",
            "dayofweek", "is_weekend", "day", "month", "dayofyear",
            "is_month_end", "is_q4", "is_holiday_rush", "is_post_holiday",
            "weight_per_mile", "is_heavy_load",
            "equipment_code", "pickup_code", "delivery_code"
        ]

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
