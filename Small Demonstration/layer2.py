"""Layer 2: Isolation Forest plus a >3-SD-from-median extreme-deviation check."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = {"claim_id", "package_id", "bill_amount", "surgery_start", "surgery_end"}
MIN_REFERENCE_GROUP_SIZE = 5


def apply_layer2(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Layer 2 cannot run; missing required columns: {sorted(missing)}")
    df = df.copy()
    starts = pd.to_datetime(df["surgery_start"], errors="coerce")
    ends = pd.to_datetime(df["surgery_end"], errors="coerce")
    duration = (ends - starts).dt.total_seconds() / 60.0
    duration = duration.where(duration >= 0)
    df["procedure_duration_minutes"] = duration.round(2)

    amount = pd.to_numeric(df["bill_amount"], errors="coerce")
    df["layer2_reference_group"] = df["package_id"].fillna("UNKNOWN_PACKAGE").astype(str)
    if "region" in df.columns:
        df["layer2_reference_group"] = df["region"].fillna("UNKNOWN_REGION").astype(str) + "::" + df["layer2_reference_group"]
        region_note = "Region and package reference groups were used where data was available."
    else:
        region_note = "REGION FIELD MISSING: package-only reference groups used; regional comparison is not possible with the supplied CSV."

    df["layer2_extreme_deviation_flag"] = "NOT_EVALUATED_SMALL_GROUP"
    df["layer2_bill_median_deviation_sd"] = np.nan
    df["layer2_duration_median_deviation_sd"] = np.nan
    df["layer2_extreme_deviation_reason"] = f"Reference groups need at least {MIN_REFERENCE_GROUP_SIZE} records; {region_note}"
    group_col = df["layer2_reference_group"]
    for group_name, index_values in df.groupby(group_col, dropna=False).groups.items():
        indexes = list(index_values)
        if len(indexes) < MIN_REFERENCE_GROUP_SIZE:
            continue
        for values, output_column in [
            (amount, "layer2_bill_median_deviation_sd"),
            (duration, "layer2_duration_median_deviation_sd"),
        ]:
            subset = values.loc[indexes].dropna()
            if len(subset) < MIN_REFERENCE_GROUP_SIZE:
                continue
            median = float(subset.median())
            # This is the requested operational definition: distance from median / group SD.
            # It is distinct from a standard z-score, which is centered on the mean.
            standard_deviation = float(subset.std(ddof=0))
            if standard_deviation > 0:
                df.loc[subset.index, output_column] = (subset - median).abs() / standard_deviation
        eligible = df.loc[indexes, ["layer2_bill_median_deviation_sd", "layer2_duration_median_deviation_sd"]]
        exceeded = (eligible > 3.0).any(axis=1)
        df.loc[eligible.index[exceeded], "layer2_extreme_deviation_flag"] = "YES"
        df.loc[eligible.index[~exceeded], "layer2_extreme_deviation_flag"] = "NO"
        for idx in eligible.index[exceeded]:
            df.loc[idx, "layer2_extreme_deviation_reason"] = "Bill amount or procedure duration exceeded 3 group SDs from its group median. " + region_note
        for idx in eligible.index[~exceeded]:
            df.loc[idx, "layer2_extreme_deviation_reason"] = "No >3-SD-from-median deviation detected in an eligible reference group. " + region_note

    # Isolation Forest is fit on numeric features from the complete demonstration dataset.
    valid = amount.notna() & duration.notna() & (duration >= 0)
    df["layer2_iforest_flag"] = "NOT_EVALUATED_INVALID_FEATURES"
    df["layer2_iforest_score"] = np.nan
    if valid.sum() >= 10:
        features = pd.DataFrame({
            "log_bill_amount": np.log1p(amount.loc[valid].clip(lower=0)),
            "procedure_duration_minutes": duration.loc[valid],
        }, index=df.index[valid])
        package_dummies = pd.get_dummies(df.loc[valid, "package_id"].fillna("UNKNOWN_PACKAGE").astype(str), prefix="package", dtype=float)
        features = pd.concat([features, package_dummies], axis=1)
        scaled = StandardScaler().fit_transform(features)
        contamination = min(0.15, max(0.05, 2 / int(valid.sum())))
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        predictions = model.fit_predict(scaled)
        scores = model.decision_function(scaled)  # Lower values indicate more unusual observations.
        df.loc[valid, "layer2_iforest_flag"] = np.where(predictions == -1, "ANOMALY", "INLIER")
        df.loc[valid, "layer2_iforest_score"] = scores

    flags, reasons = [], []
    for idx, row in df.iterrows():
        hits = []
        if row["layer2_iforest_flag"] == "ANOMALY":
            hits.append("Isolation Forest marked this record as unusual relative to this small dataset")
        if row["layer2_extreme_deviation_flag"] == "YES":
            hits.append("Bill amount or procedure duration exceeded the configured >3-SD-from-median threshold")
        flags.append("YES" if hits else "NO")
        if hits:
            reasons.append("; ".join(hits) + ". Not evidence of fraud by itself. " + region_note)
        else:
            reasons.append("No Layer 2 flag from the implemented checks. " + region_note)
    df["layer2_flag"] = flags
    df["layer2_reason"] = reasons
    return df


def run_layer2(csv_path):
    path = Path(csv_path)
    updated = apply_layer2(pd.read_csv(path))
    updated.to_csv(path, index=False)
    print(f"Layer 2 updated: {path.name}")
    return updated


if __name__ == "__main__":
    run_layer2(Path(__file__).resolve().parent / "fraud_results.csv")
