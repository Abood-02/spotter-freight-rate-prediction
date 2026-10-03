"""
Data Preprocessing Module
Handles data cleaning, negative weight rectification, and robust imputation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


class DataPreprocessor:
    """Preprocesses freight shipment datasets, fitting state strictly on training data."""

    def __init__(self) -> None:
        self.equipment_weight_medians: dict[str, float] = {}
        self.global_weight_median: float = 32000.0
        self.global_market_index_median: float = 1.0
        self.global_quote_signal_median: float = 2.0
        self.city_coordinates: dict[str, tuple[float, float]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Learn imputation values and city coordinates from training data."""
        clean_weights = df["weight"].abs()
        
        # 1. Learn median weight by equipment
        equipment_medians = clean_weights.groupby(df["equipment"]).median().to_dict()
        self.equipment_weight_medians = equipment_medians
        self.global_weight_median = float(clean_weights.median())

        # 2. Learn market index and quote signal medians
        if "market_index" in df.columns:
            self.global_market_index_median = float(df["market_index"].dropna().median())
        if "quote_signal" in df.columns:
            self.global_quote_signal_median = float(df["quote_signal"].dropna().median())

        # 3. Build city-to-coordinate lookup table
        pickup_coords = df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
        for city, row in pickup_coords.iterrows():
            self.city_coordinates[str(city)] = (float(row["pickup_lat"]), float(row["pickup_lon"]))

        delivery_coords = df.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
        for city, row in delivery_coords.iterrows():
            if str(city) not in self.city_coordinates:
                self.city_coordinates[str(city)] = (float(row["delivery_lat"]), float(row["delivery_lon"]))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to clean and impute dataframe."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transform.")

        out = df.copy()

        # Standardize date format to ISO YYYY-MM-DD
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

        # 1. Fix negative weights (sign inversion bug) and ensure float dtype
        if "weight" in out.columns:
            out["weight"] = out["weight"].astype(float).abs()
            # Impute missing weights by equipment median
            for eq, med in self.equipment_weight_medians.items():
                mask = out["weight"].isna() & (out["equipment"] == eq)
                out.loc[mask, "weight"] = float(med)
            out["weight"] = out["weight"].fillna(self.global_weight_median)

        # 2. Impute market index
        if "market_index" in out.columns:
            out["market_index"] = out["market_index"].fillna(self.global_market_index_median)
        else:
            out["market_index"] = self.global_market_index_median

        # 3. Handle quote signal (if missing, e.g. december-chart-inputs)
        if "quote_signal" in out.columns:
            out["quote_signal"] = out["quote_signal"].fillna(self.global_quote_signal_median)
        else:
            out["quote_signal"] = self.global_quote_signal_median

        # 4. Impute missing coordinates (for december-chart-inputs)
        if "pickup_lat" not in out.columns or out["pickup_lat"].isna().any():
            if "pickup_lat" not in out.columns:
                out["pickup_lat"] = np.nan
                out["pickup_lon"] = np.nan
            for idx, row in out[out["pickup_lat"].isna()].iterrows():
                city = row.get("pickup")
                if city in self.city_coordinates:
                    lat, lon = self.city_coordinates[city]
                    out.at[idx, "pickup_lat"] = lat
                    out.at[idx, "pickup_lon"] = lon

        if "delivery_lat" not in out.columns or out["delivery_lat"].isna().any():
            if "delivery_lat" not in out.columns:
                out["delivery_lat"] = np.nan
                out["delivery_lon"] = np.nan
            for idx, row in out[out["delivery_lat"].isna()].iterrows():
                city = row.get("delivery")
                if city in self.city_coordinates:
                    lat, lon = self.city_coordinates[city]
                    out.at[idx, "delivery_lat"] = lat
                    out.at[idx, "delivery_lon"] = lon

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
