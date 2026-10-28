"""Optional controlled examples proving each detection mechanism can trigger.
These scenarios are held in memory and do not modify claims.csv or fraud_results.csv.
"""
from pathlib import Path
import pandas as pd
from layer1 import apply_layer1
from layer2 import apply_layer2
from layer3 import apply_layer3, find_collisions, build_relationship_graph, draw_graph

FOLDER = Path(__file__).resolve().parent


def main():
    base = pd.read_csv(FOLDER / "claims.csv")
    examples = []

    # Layer 1a: anatomy inconsistency is reviewable when verified clinical context exists.
    bio = base.iloc[[0]].copy()
    bio["claim_id"] = "DEMO-BIO"
    bio["procedure"] = "Hysterectomy"
    bio["verified_anatomical_context"] = "UTERUS_ABSENT"
    bio_result = apply_layer1(bio).iloc[0]
    examples.append(["Layer 1 - clinical consistency", "Injected hysterectomy/anatomy inconsistency", bio_result["layer1_biological_status"], bio_result["layer1_biological_status"] == "FLAG_FOR_CLINICAL_REVIEW"])

    # Layer 1b: package inclusion plus separate billing triggers the bundled-item rule.
    bundled = base.iloc[[1]].copy()
    bundled["claim_id"] = "DEMO-BUNDLE"
    bundled["billed_items"] = "surgeon_fee|package_service_x"
    bundled["package_included_items"] = "package_service_x|room_fee"
    bundled["allowed_separate_billing_items"] = "surgeon_fee"
    bundle_result = apply_layer1(bundled).iloc[0]
    examples.append(["Layer 1 - unbundled billing", "Injected included item billed separately", bundle_result["layer1_unbundled_status"], bundle_result["layer1_unbundled_status"] == "FLAG_FOR_REVIEW"])

    # Layer 1c: same patient appears at different hospitals during overlapping admission dates.
    overlap_a = base.iloc[[0]].copy()
    overlap_a["claim_id"] = "DEMO-OVERLAP-A"
    overlap_a["patient_id"] = "PT-DEMO"
    overlap_a["admission_date"] = "2026-09-01"
    overlap_a["discharge_date"] = "2026-09-05"
    overlap_b = base.iloc[[1]].copy()
    overlap_b["claim_id"] = "DEMO-OVERLAP-B"
    overlap_b["patient_id"] = "PT-DEMO"
    overlap_b["hospital_id"] = "H-DEMO-2"
    overlap_b["admission_date"] = "2026-09-03"
    overlap_b["discharge_date"] = "2026-09-06"
    overlap_result = apply_layer1(pd.concat([overlap_a, overlap_b], ignore_index=True))
    overlap_detected = (overlap_result["layer1_admission_overlap_status"] == "FLAG_FOR_REVIEW").all()
    examples.append(["Layer 1 - overlapping admissions", "Injected overlapping dates across hospitals", "FLAG_FOR_REVIEW" if overlap_detected else "NOT_DETECTED", bool(overlap_detected)])

    # Layer 2: create a controlled unusually high bill and long surgery in a temporary in-memory dataset.
    outlier_data = base.copy()
    outlier_data.loc[0, "bill_amount"] = 1_500_000
    outlier_data.loc[0, "surgery_start"] = "2026-09-01 01:00"
    outlier_data.loc[0, "surgery_end"] = "2026-09-01 20:00"
    outlier_result = apply_layer2(outlier_data)
    outlier_row = outlier_result.loc[outlier_result["claim_id"] == "CL001"].iloc[0]
    detected = (outlier_row["layer2_iforest_flag"] == "ANOMALY") or (outlier_row["layer2_extreme_deviation_flag"] == "YES")
    examples.append(["Layer 2 - statistical anomaly", "Injected extreme bill and duration in test copy", f"IF={outlier_row['layer2_iforest_flag']}; >3SD={outlier_row['layer2_extreme_deviation_flag']}", bool(detected)])

    # Layer 3: a doctor is assigned overlapping surgeries at two separate hospitals.
    collision_a = base.iloc[[0]].copy()
    collision_a["claim_id"] = "DEMO-COLLISION-A"
    collision_a["doctor_id"] = "DR-DEMO"
    collision_a["hospital_id"] = "H-DEMO-1"
    collision_a["surgery_start"] = "2026-09-01 09:00"
    collision_a["surgery_end"] = "2026-09-01 11:00"
    collision_b = base.iloc[[1]].copy()
    collision_b["claim_id"] = "DEMO-COLLISION-B"
    collision_b["doctor_id"] = "DR-DEMO"
    collision_b["hospital_id"] = "H-DEMO-2"
    collision_b["surgery_start"] = "2026-09-01 10:00"
    collision_b["surgery_end"] = "2026-09-01 12:00"
    collision_data = pd.concat([collision_a, collision_b], ignore_index=True)
    collisions = find_collisions(collision_data)
    collision_graph = build_relationship_graph(collision_data, collisions)
    draw_graph(collision_graph, FOLDER / "demo_collision_graph.png", "Controlled Example: Same Doctor Scheduled at Two Hospitals Simultaneously")
    examples.append(["Layer 3 - NetworkX collision", "Injected overlapping surgery schedules across hospitals", f"{len(collisions)} collision pair(s)", len(collisions) > 0])

    result = pd.DataFrame(examples, columns=["layer_or_rule", "controlled_scenario", "observed_result", "passed"])
    result.to_csv(FOLDER / "demo_scenario_results.csv", index=False)
    print(result.to_string(index=False))
    print("\nWrote demo_scenario_results.csv and demo_collision_graph.png.")
    print("All injected scenarios are isolated from the original input and main fraud_results.csv.")


if __name__ == "__main__":
    main()
