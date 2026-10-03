"""
Forensic Exploratory Data Analysis (EDA) Script
"""
import pandas as pd
import numpy as np


def print_stage_header(title: str, stage: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"STAGE {stage}: {title.upper()}")
    print(f"{'=' * 70}")


def stage_1_structural_reconnaissance(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print_stage_header("Structural Reconnaissance", 1)
    print(f"Train dimensions:      {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"Validation dimensions: {val.shape[0]:,} rows × {val.shape[1]} columns")
    print(f"Train unique IDs:      {train['load_id'].nunique():,}")
    print(f"Val unique IDs:        {val['load_id'].nunique():,}")
    print(f"ID overlap:            {len(set(train['load_id']).intersection(set(val['load_id'])))}")

    schema_df = pd.DataFrame({
        "Train Dtype": train.dtypes,
        "Val Dtype": val.dtypes,
        "Train Nulls": train.isnull().sum(),
        "Val Nulls": val.isnull().sum()
    })
    print("\nFeature Schema:")
    print(schema_df.to_string())


def stage_2_target_forensics(train: pd.DataFrame) -> None:
    print_stage_header("Target Variable Forensics", 2)
    y = train["posted_rate"]
    rpm = y / train["distance"]

    stats_df = pd.DataFrame({
        "Metric": ["Count", "Mean", "Std", "Min", "25%", "Median (50%)", "75%", "99%", "Max", "Skewness"],
        "posted_rate ($)": [
            f"{len(y):,}", f"${y.mean():,.2f}", f"${y.std():,.2f}", f"${y.min():,.2f}",
            f"${y.quantile(0.25):,.2f}", f"${y.median():,.2f}", f"${y.quantile(0.75):,.2f}",
            f"${y.quantile(0.99):,.2f}", f"${y.max():,.2f}", f"{y.skew():.2f}"
        ],
        "Rate Per Mile ($/mi)": [
            f"{len(rpm):,}", f"${rpm.mean():.2f}", f"${rpm.std():.2f}", f"${rpm.min():.2f}",
            f"${rpm.quantile(0.25):,.2f}", f"${rpm.median():.2f}", f"${rpm.quantile(0.75):,.2f}",
            f"${rpm.quantile(0.99):,.2f}", f"${rpm.max():.2f}", f"{rpm.skew():.2f}"
        ]
    })
    print(stats_df.to_string(index=False))
    print(f"\nNon-positive rates (<= 0): {(y <= 0).sum()}")


def stage_3_physical_reality_checks(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print_stage_header("Physical Reality & Anomaly Audit", 3)

    train_neg_wt = (train["weight"] < 0).sum()
    val_neg_wt = (val["weight"] < 0).sum()
    pos_weights = train[train["weight"] > 0]["weight"]
    neg_weights_abs = train[train["weight"] < 0]["weight"].abs()

    print(f"Weight (lbs) Audit:")
    print(f"  Train negative weights: {train_neg_wt} rows ({train_neg_wt/len(train)*100:.2f}%)")
    print(f"  Val negative weights:   {val_neg_wt} rows ({val_neg_wt/len(val)*100:.2f}%)")
    print(f"  Positive weight mean:   {pos_weights.mean():.1f} lbs (std: {pos_weights.std():.1f})")
    print(f"  Abs(Negative) mean:     {neg_weights_abs.mean():.1f} lbs (std: {neg_weights_abs.std():.1f})")

    print(f"\nDistance (miles) Audit:")
    print(f"  Train: [{train['distance'].min():.1f}, {train['distance'].max():.1f}], mean = {train['distance'].mean():.1f}")
    print(f"  Val:   [{val['distance'].min():.1f}, {val['distance'].max():.1f}], mean = {val['distance'].mean():.1f}")

    print(f"\nGeographic Bounds Audit:")
    print(f"  Pickup Lat/Lon:   [{train['pickup_lat'].min():.2f}, {train['pickup_lat'].max():.2f}] / [{train['pickup_lon'].min():.2f}, {train['pickup_lon'].max():.2f}]")
    print(f"  Delivery Lat/Lon: [{train['delivery_lat'].min():.2f}, {train['delivery_lat'].max():.2f}] / [{train['delivery_lon'].min():.2f}, {train['delivery_lon'].max():.2f}]")


def stage_4_missingness_audit(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print_stage_header("Missingness Audit", 4)
    missing = []
    for col in train.columns:
        if col == "posted_rate":
            continue
        t_null = train[col].isnull().sum()
        v_null = val[col].isnull().sum()
        if t_null > 0 or v_null > 0:
            missing.append({
                "Column": col,
                "Train Nulls": f"{t_null:,} ({t_null/len(train)*100:.2f}%)",
                "Val Nulls": f"{v_null:,} ({v_null/len(val)*100:.2f}%)"
            })
    print(pd.DataFrame(missing).to_string(index=False))


def stage_5_feature_dynamics(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print_stage_header("Feature Correlations & Cardinality", 5)
    y = train["posted_rate"]
    base_calc = train["quote_signal"] * train["distance"]

    corr_df = pd.DataFrame([
        {"Feature": "distance", "Correlation with posted_rate": train["distance"].corr(y)},
        {"Feature": "quote_signal * distance", "Correlation with posted_rate": base_calc.corr(y)},
        {"Feature": "quote_signal (alone)", "Correlation with posted_rate": train["quote_signal"].corr(y)},
        {"Feature": "market_index", "Correlation with posted_rate": train["market_index"].corr(y)},
        {"Feature": "abs(weight)", "Correlation with posted_rate": train["weight"].abs().corr(y)}
    ])
    print("Correlations:")
    print(corr_df.to_string(index=False))

    train_lanes = set(train["pickup"] + " -> " + train["delivery"])
    val_lanes = set(val["pickup"] + " -> " + val["delivery"])
    unseen_lanes = val_lanes - train_lanes

    print(f"\nLane Cardinality:")
    print(f"  Unique lanes in Train:      {len(train_lanes):,}")
    print(f"  Unique lanes in Validation: {len(val_lanes):,}")
    print(f"  Unseen lanes in Validation: {len(unseen_lanes):,} ({len(unseen_lanes)/len(val_lanes)*100:.1f}%)")


def stage_6_temporal_split_strategy(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print_stage_header("Temporal Horizon & Split Setup", 6)
    print(f"Train date span:      {train['date'].min()} to {train['date'].max()} ({train['date'].nunique()} days)")
    print(f"Validation date span: {val['date'].min()} to {val['date'].max()} ({val['date'].nunique()} days)")

    train_copy = train.copy()
    train_copy["date"] = pd.to_datetime(train_copy["date"])
    train_copy["month"] = train_copy["date"].dt.to_period("M")
    monthly = train_copy.groupby("month").agg(
        loads=("posted_rate", "count"),
        avg_rate=("posted_rate", "mean"),
        avg_rpm=("posted_rate", lambda s: (s / train_copy.loc[s.index, "distance"]).mean())
    )
    print("\nMonthly Overview:")
    print(monthly.to_string())


def main() -> None:
    train = pd.read_csv("train-test.csv")
    val = pd.read_csv("validation.csv")

    stage_1_structural_reconnaissance(train, val)
    stage_2_target_forensics(train)
    stage_3_physical_reality_checks(train, val)
    stage_4_missingness_audit(train, val)
    stage_5_feature_dynamics(train, val)
    stage_6_temporal_split_strategy(train, val)


if __name__ == "__main__":
    main()
