"""
Model Definitions & Benchmarking Architecture
Implements 3-tier hierarchy: Heuristic Baseline, Ridge Regression, and LightGBM.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import lightgbm as lgb


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE metrics."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100.0)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }


class HeuristicBaseline:
    """Tier 1: Domain formula estimate (quote_signal * distance)."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HeuristicBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "base_quote" in X.columns:
            return X["base_quote"].to_numpy(dtype=float)
        return (X["quote_signal"] * X["distance"]).to_numpy(dtype=float)


class RidgeBaseline:
    """Tier 2: Regularized linear model with feature standardization."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=42))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, a_min=10.0, a_max=None)


class LightGBMProductionModel:
    """Tier 3: Gradient Boosted Decision Tree with L1 regression objective."""

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.04,
        num_leaves: int = 31,
        max_depth: int = -1,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
    ) -> None:
        self.params: dict[str, Any] = {
            "objective": "regression_l1",
            "metric": "l1",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMProductionModel":
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        
        # Suppress deprecation warning for eval_set across LightGBM versions
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks if callbacks else None,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, a_min=10.0, a_max=None)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
