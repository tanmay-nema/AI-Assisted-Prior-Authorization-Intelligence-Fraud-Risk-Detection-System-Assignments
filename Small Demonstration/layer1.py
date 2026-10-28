"""Layer 1: rule-based hard gates for the Tri-Fold Fraud Engine.

This is a small feasibility demonstration, not a production fraud adjudicator.
The input file is updated in place; app.py first creates fraud_results.csv from
claims.csv so the original input remains unchanged.
"""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "claim_id", "patient_id", "hospital_id", "procedure",
    "admission_date", "discharge_date"
}


def _split_items(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    # For this demonstration, package/billed item lists may be separated by | or ;.
    return {part.strip().casefold() for part in str(value).replace(";", "|").split("|") if part.strip()}


def apply_layer1(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Layer 1 cannot run; missing required columns: {sorted(missing)}")

    df = df.copy().reset_index(drop=True)
    n = len(df)
    bio_status = ["NOT_APPLICABLE"] * n
    bio_reason = ["This procedure-consistency rule does not apply to the recorded procedure."] * n
    package_status = ["NOT_EVALUABLE_MISSING_FIELDS"] * n
    package_reason = ["Not evaluated: itemized billed services and package-inclusion rules are not present in the supplied CSV."] * n
    overlap_status = ["NO_OVERLAP_DETECTED"] * n
    overlap_reason = [""] * n

    # Clinical consistency check. Sex alone is not treated as proof of impossibility.
    for idx, row in df.iterrows():
        procedure = str(row.get("procedure", "")).casefold()
        if "hysterectomy" in procedure:
            bio_status[idx] = "NOT_EVALUABLE_MISSING_FIELDS"
            bio_reason[idx] = "Not evaluated: verified patient anatomy/clinical context is not present in the supplied CSV."
            anatomy = str(row.get("verified_anatomical_context", "")).strip().upper()
            patient_sex = str(row.get("patient_sex", "")).strip().casefold()
            if anatomy in {"UTERUS_ABSENT", "NO_UTERUS"}:
                bio_status[idx] = "FLAG_FOR_CLINICAL_REVIEW"
                bio_reason[idx] = "Hysterectomy claim conflicts with the recorded verified anatomical context; verify records and exceptions."
            elif patient_sex == "male":
                bio_status[idx] = "REVIEW_REQUIRED"
                bio_reason[idx] = "Hysterectomy claim with patient_sex=Male; verify anatomy and clinical documentation. Sex alone does not establish fraud."
            elif anatomy in {"UTERUS_PRESENT", "PRESENT"}:
                bio_status[idx] = "NO_RULE_TRIGGERED"
                bio_reason[idx] = "Recorded anatomy does not trigger this consistency rule."
            elif "patient_sex" in df.columns or "verified_anatomical_context" in df.columns:
                bio_status[idx] = "INSUFFICIENT_DATA"
                bio_reason[idx] = "Hysterectomy claim needs verified anatomy/context; available fields are incomplete."

    # Unbundled billing check requires item-level billing and package reference data.
    package_fields = {"billed_items", "package_included_items"}
    if package_fields.issubset(df.columns):
        for idx, row in df.iterrows():
            billed = _split_items(row.get("billed_items"))
            included = _split_items(row.get("package_included_items"))
            allowed = _split_items(row.get("allowed_separate_billing_items", ""))
            if not billed or not included:
                package_status[idx] = "INSUFFICIENT_DATA"
                package_reason[idx] = "Billed items or package-inclusion reference is empty."
            else:
                suspect_items = sorted((billed & included) - allowed)
                if suspect_items:
                    package_status[idx] = "FLAG_FOR_REVIEW"
                    package_reason[idx] = "Potential duplicate package billing for item(s): " + ", ".join(suspect_items)
                else:
                    package_status[idx] = "NO_RULE_TRIGGERED"
                    package_reason[idx] = "No included item was found separately billed under the configured rule."

    # Flag overlapping admission periods for the same patient at different hospitals.
    admissions = df.copy()
    admissions["_admit"] = pd.to_datetime(admissions["admission_date"], errors="coerce")
    admissions["_discharge"] = pd.to_datetime(admissions["discharge_date"], errors="coerce")
    for i in range(n):
        for j in range(i + 1, n):
            a, b = admissions.iloc[i], admissions.iloc[j]
            if pd.isna(a["_admit"]) or pd.isna(a["_discharge"]) or pd.isna(b["_admit"]) or pd.isna(b["_discharge"]):
                continue
            if str(a["patient_id"]) != str(b["patient_id"]):
                continue
            if str(a["hospital_id"]) == str(b["hospital_id"]):
                continue
            overlaps = max(a["_admit"], b["_admit"]) <= min(a["_discharge"], b["_discharge"])
            if overlaps:
                message_i = f"Admission overlaps with claim {b['claim_id']} at hospital {b['hospital_id']}."
                message_j = f"Admission overlaps with claim {a['claim_id']} at hospital {a['hospital_id']}."
                overlap_status[i] = overlap_status[j] = "FLAG_FOR_REVIEW"
                overlap_reason[i] = (overlap_reason[i] + " " + message_i).strip()
                overlap_reason[j] = (overlap_reason[j] + " " + message_j).strip()

    df["layer1_biological_status"] = bio_status
    df["layer1_biological_reason"] = bio_reason
    df["layer1_unbundled_status"] = package_status
    df["layer1_unbundled_reason"] = package_reason
    df["layer1_admission_overlap_status"] = overlap_status
    df["layer1_admission_overlap_reason"] = overlap_reason

    hard_gate_statuses = {"FLAG_FOR_CLINICAL_REVIEW", "REVIEW_REQUIRED", "FLAG_FOR_REVIEW"}
    flags, reasons = [], []
    for i in range(n):
        triggered = []
        if bio_status[i] in hard_gate_statuses:
            triggered.append("Biological/clinical consistency: " + bio_reason[i])
        if package_status[i] in hard_gate_statuses:
            triggered.append("Unbundled billing: " + package_reason[i])
        if overlap_status[i] == "FLAG_FOR_REVIEW":
            triggered.append("Overlapping admission: " + overlap_reason[i])
        flags.append("YES" if triggered else "NO")
        reasons.append(" | ".join(triggered) if triggered else "No applicable Layer 1 rule triggered. Some checks may be unavailable because source fields are missing.")
    df["layer1_flag"] = flags
    df["layer1_reason"] = reasons
    return df


def run_layer1(csv_path):
    path = Path(csv_path)
    frame = pd.read_csv(path)
    updated = apply_layer1(frame)
    updated.to_csv(path, index=False)
    print(f"Layer 1 updated: {path.name}")
    return updated


if __name__ == "__main__":
    run_layer1(Path(__file__).resolve().parent / "fraud_results.csv")
