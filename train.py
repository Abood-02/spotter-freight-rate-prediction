"""
Model Training & Benchmarking Pipeline
Trains 3 model tiers on temporal split, prints leaderboard, and saves production model.
"""

from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer
from src.split import get_temporal_split
from src.models import (
    HeuristicBaseline,
    RidgeBaseline,
    LightGBMProductionModel,
    compute_metrics,
)


def run_training() -> None:
    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - MODEL BENCHMARKING & TRAINING")
    print("=" * 70)

    # 1. Load Data
    raw_df = pd.read_csv("train-test.csv")
    print(f"Loaded {len(raw_df):,} raw development loads.")

    # 2. Preprocess & Feature Engineer
    preprocessor = DataPreprocessor().fit(raw_df)
    clean_df = preprocessor.transform(raw_df)

    engineer = FeatureEngineer().fit(clean_df)
    features_df = engineer.transform(clean_df)

    feature_cols = engineer.feature_columns
    target_col = "posted_rate"
    print(f"Engineered {len(feature_cols)} features.")

    # 3. Temporal Split
    train_fold, val_fold = get_temporal_split(features_df, split_date="2025-09-01")
    print(f"Temporal Split:")
    print(f"  • Training Fold   (Jan 01 - Aug 31): {len(train_fold):,} rows")
    print(f"  • Validation Fold (Sep 01 - Oct 31): {len(val_fold):,} rows")

    X_train, y_train = train_fold[feature_cols], train_fold[target_col]
    X_val, y_val = val_fold[feature_cols], val_fold[target_col]

    # 4. Benchmarking Tier 1: Heuristic Baseline
    print("\nTraining & Evaluating Models on Holdout (Sep-Oct 2025)...")
    m_heuristic = HeuristicBaseline().fit(X_train, y_train)
    p_heuristic = m_heuristic.predict(X_val)
    metrics_heuristic = compute_metrics(y_val, p_heuristic)

    # 5. Benchmarking Tier 2: Ridge Regression
    m_ridge = RidgeBaseline(alpha=10.0).fit(X_train, y_train)
    p_ridge = m_ridge.predict(X_val)
    metrics_ridge = compute_metrics(y_val, p_ridge)

    # 6. Benchmarking Tier 3: LightGBM Regressor
    m_lgbm = LightGBMProductionModel(n_estimators=1000, learning_rate=0.03, num_leaves=31)
    m_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=40)
    p_lgbm = m_lgbm.predict(X_val)
    metrics_lgbm = compute_metrics(y_val, p_lgbm)

    # 7. Print Official Comparison Leaderboard
    leaderboard = pd.DataFrame([
        {"Tier": "Tier 1: Domain Heuristic", "Model": "quote_signal * distance", **metrics_heuristic},
        {"Tier": "Tier 2: Linear Baseline", "Model": "Ridge Regression", **metrics_ridge},
        {"Tier": "Tier 3: Production Model", "Model": "LightGBM Regressor (L1)", **metrics_lgbm},
    ])

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE LEADERBOARD (TEMPORAL HOLDOUT SET)")
    print("=" * 70)
    print(leaderboard.to_string(index=False))

    mae_reduction = ((metrics_heuristic["MAE"] - metrics_lgbm["MAE"]) / metrics_heuristic["MAE"]) * 100
    print(f"\n=> Production LightGBM reduces MAE by {mae_reduction:.1f}% over the domain baseline.")

    # 8. Feature Importance
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": m_lgbm.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nTop 8 Most Important Features (LightGBM):")
    for idx, row in importance_df.head(8).iterrows():
        print(f"  {idx+1}. {row['Feature']:20s} (splits: {row['Importance']:,})")

    # 9. Retrain on 100% of Labeled Data for Final Inference
    print("\nRetraining LightGBM on 100% of labeled data (Jan-Oct 2025)...")
    full_preprocessor = DataPreprocessor().fit(raw_df)
    full_clean = full_preprocessor.transform(raw_df)
    full_engineer = FeatureEngineer().fit(full_clean)
    full_features = full_engineer.transform(full_clean)

    X_full = full_features[feature_cols]
    y_full = full_features[target_col]

    final_model = LightGBMProductionModel(n_estimators=750, learning_rate=0.03, num_leaves=31)
    final_model.fit(X_full, y_full)

    # 10. Save Pipeline Artifacts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "production_pipeline.joblib"

    artifact_bundle = {
        "preprocessor": full_preprocessor,
        "engineer": full_engineer,
        "model": final_model,
        "feature_cols": feature_cols,
        "leaderboard": leaderboard,
    }
    joblib.dump(artifact_bundle, bundle_path)
    print(f"Production pipeline bundle saved to: {bundle_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()
