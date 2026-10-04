"""
Inference & Submission Pipeline (Phase 4)
Loads production model, generates predictions for validation and December inputs
"""

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def generate_predictions(
    model_path: str = "models/production_pipeline.joblib",
    val_input_path: str = "validation.csv",
    dec_input_path: str = "december-chart-inputs.csv",
    val_output_path: str = "validation_predictions.csv",
    dec_output_path: str = "december-chart-inputs.csv",
) -> None:
    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - INFERENCE PIPELINE (PHASE 4)")
    print("=" * 70)

    # 1. Load Serialized Pipeline
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model bundle not found at {model_path}. Run train.py first.")

    bundle = joblib.load(model_file)
    preprocessor = bundle["preprocessor"]
    engineer = bundle["engineer"]
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"Loaded production pipeline bundle from: {model_path}")

    # 2. Predict on validation.csv (12,000 loads)
    print(f"\nProcessing {val_input_path}...")
    raw_val = pd.read_csv(val_input_path)
    clean_val = preprocessor.transform(raw_val)
    feat_val = engineer.transform(clean_val)

    X_val = feat_val[feature_cols]
    val_preds = model.predict(X_val)
    val_preds = np.round(val_preds, 2)

    val_submission = pd.DataFrame({
        "load_id": raw_val["load_id"],
        "predicted_rate": val_preds
    })

    # Sanity checks on validation predictions
    assert len(val_submission) == 12000, f"Expected 12,000 rows, got {len(val_submission)}"
    assert list(val_submission.columns) == ["load_id", "predicted_rate"], "Invalid columns"
    assert (val_submission["predicted_rate"] > 0).all(), "Predictions contain non-positive rates"
    assert not val_submission["predicted_rate"].isna().any(), "Predictions contain NaNs"

    val_submission.to_csv(val_output_path, index=False)
    print(f"Saved {len(val_submission):,} validation predictions to: {val_output_path}")
    print(f"  • Rate stats: min=${val_preds.min():.2f}, mean=${val_preds.mean():.2f}, median=${np.median(val_preds):.2f}, max=${val_preds.max():.2f}")

    # 3. Predict on december-chart-inputs.csv (31 loads)
    print(f"\nProcessing {dec_input_path}...")
    raw_dec = pd.read_csv(dec_input_path)
    clean_dec = preprocessor.transform(raw_dec)
    feat_dec = engineer.transform(clean_dec)

    X_dec = feat_dec[feature_cols]
    dec_preds = model.predict(X_dec)
    dec_preds = np.round(dec_preds, 2)

    # Preserve exact original 7 columns
    out_dec = raw_dec.copy()
    out_dec["predicted_rate"] = dec_preds

    assert len(out_dec) == 31, f"Expected 31 rows, got {len(out_dec)}"
    expected_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert list(out_dec.columns) == expected_cols, "December columns altered"
    assert (out_dec["predicted_rate"] > 0).all(), "December predictions contain non-positive rates"

    out_dec.to_csv(dec_output_path, index=False)
    print(f"Saved completed December inputs to: {dec_output_path}")
    print(f"  • December rate stats: min=${dec_preds.min():.2f}, mean=${dec_preds.mean():.2f}, max=${dec_preds.max():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    generate_predictions()
