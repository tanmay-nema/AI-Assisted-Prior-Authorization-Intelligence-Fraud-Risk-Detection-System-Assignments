"""Run the three Tri-Fold layers in sequence on a working copy of claims.csv."""
from pathlib import Path
import shutil
import pandas as pd
from layer1 import run_layer1
from layer2 import run_layer2
from layer3 import run_layer3

FOLDER = Path(__file__).resolve().parent
INPUT_FILE = FOLDER / "claims.csv"
OUTPUT_FILE = FOLDER / "fraud_results.csv"
GRAPH_FILE = FOLDER / "layer3_relationship_graph.png"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Copy the untouched source before the first layer runs.
    shutil.copy2(INPUT_FILE, OUTPUT_FILE)
    print(f"Created working copy: {OUTPUT_FILE.name}")

    run_layer1(OUTPUT_FILE)
    run_layer2(OUTPUT_FILE)
    run_layer3(OUTPUT_FILE, GRAPH_FILE)

    result = pd.read_csv(OUTPUT_FILE)
    layer_cols = ["layer1_flag", "layer2_flag", "layer3_flag"]
    result["_layers_flagged"] = result[layer_cols].eq("YES").sum(axis=1)
    result["overall_status"] = result["_layers_flagged"].map(
        lambda count: "PRIORITY_REVIEW" if count >= 2 else ("MANUAL_REVIEW" if count == 1 else "NO_FLAGS_IN_EVALUATED_CHECKS")
    )
    limitations = []
    if not {"patient_sex", "verified_anatomical_context"}.intersection(result.columns):
        limitations.append("biological/anatomical procedure check unavailable: no patient anatomy/sex fields")
    if not {"billed_items", "package_included_items"}.issubset(result.columns):
        limitations.append("unbundled-billing check unavailable: no itemized bills/package inclusion rules")
    if "region" not in result.columns:
        limitations.append("regional outlier comparison unavailable: no region field")
    result["overall_status_note"] = "; ".join(limitations) if limitations else "Check input coverage before interpreting results."
    result = result.drop(columns=["_layers_flagged"])
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nTri-Fold run complete.")
    print(f"Final output: {OUTPUT_FILE.name}")
    print(f"Graph: {GRAPH_FILE.name}")
    print("Flags by layer:")
    for col in layer_cols:
        print(f"  {col}: {(result[col] == 'YES').sum()} claim(s)")
    print("Overall statuses:")
    print(result["overall_status"].value_counts().to_string())
    print("\nLimitations are recorded in overall_status_note and layer-specific reason columns.")


if __name__ == "__main__":
    main()
