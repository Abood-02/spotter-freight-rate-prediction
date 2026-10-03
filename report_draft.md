# Freight Rate Prediction Assessment Report
Machine Learning Engineer Candidate | Spotter | October 2026

## 1. Approach to Validate Test/Train Data and Split

### Validating the Train and Test Data
Before modeling, I conducted a forensic data quality audit across both train-test.csv (48,000 development loads) and validation.csv (12,000 evaluation loads) to ensure dataset integrity and identify potential distribution shifts:

1. Schema and Identifier Parity: Verified that load IDs are unique and non-overlapping (TR-000001 through TR-048000 for training, TE-000001 through TE-012000 for validation), confirming clean dataset isolation.
2. Anomaly Detection and Physical Sanity: Audited physical boundaries and uncovered negative weights in both datasets (292 rows in train, 145 rows in validation, down to -47,500 lbs). Comparing the distributions of positive weights and the absolute values of negative weights confirmed an upstream sign-inversion bug, which was resolved using absolute values.
3. Missingness Audit: Confirmed that missing values exist in the exact same features across both datasets: weight (approximately 1%) and market_index (approximately 1.5%). Missing weights were imputed using medians stratified by equipment type to preserve vehicle payload physics.
4. Generalization and Shift Check: Cross-referencing origin-destination pairs revealed that 736 lanes in the validation set (17.5%) never appeared in training data. This validated that the model could not rely on memorizing city names and required continuous spatial features (lat/lon coordinates and Haversine geodesic distance) to price unseen routes.

### Splitting Strategy
Because freight rates are time-dependent, using standard random K-Fold cross-validation would shuffle future records into training folds, causing temporal data leakage where future market indices and holiday rate spikes leak backward into past predictions.

To replicate the real-world challenge of predicting future November and December loads, I implemented a chronological out-of-time split on the development data:
- Training Fold: January 1 to August 31, 2025 (Months 1 to 8, containing 38,477 loads, 80.2%)
- Validation Fold: September 1 to October 31, 2025 (Months 9 to 10, containing 9,523 loads, 19.8%)

All preprocessors, coordinate lookups, and imputation medians were fitted strictly on the Training Fold and applied immutably to the Validation Fold to ensure zero data leakage.

Evaluating on the unseen September-October holdout confirmed the chosen architecture:
- Broker Baseline (quote_signal * distance): MAE 246.02 USD, MAPE 12.11%
- Ridge Regression (L2 Linear Baseline): MAE 153.79 USD, MAPE 7.93%
- Production LightGBM (L1 Objective): MAE 107.43 USD, MAPE 4.63% (56.3% error reduction)


## 2. Fixed December Prediction Chart

Using the final model, daily December rates were predicted for the fixed route specified in december-chart-inputs.csv (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs). The chart below was produced directly by Spotter's official score.py validator:

![December 2025 Predicted Load Rate](scorer_results/candidate_december.png)

Key Observations:
- Baseline Calibration: Forecasted rates average 834.51 USD (ranging from 828.87 to 837.59 USD), cleanly matching the historical training average of 856.59 USD on this exact lane.
- Calendar Dynamics: Rather than outputting an insensitive flat line, the model reflects freight seasonality, starting at baseline levels (around 828 USD) and rising to approximately 837 USD during the late-December holiday shipping push.
- Compliance: Both validation_predictions.csv and december-chart-inputs.csv passed score.py with zero errors.
