import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# ==============================================================================
# 1. LOAD KNOWLEDGE BASE & TEST CASES FROM JSON FILES
# ==============================================================================
def load_json_data() -> Tuple[Dict[str, Dict[str, Any]], list]:
    with open("policy_knowledge_base.json", "r", encoding="utf-8") as f:
        policy_kb = json.load(f)
    
    with open("synthetic_cases.json", "r", encoding="utf-8") as f:
        synthetic_cases = json.load(f)
        
    return policy_kb, synthetic_cases


# ==============================================================================
# 2. DECISION ENGINES: SYSTEM A (NAIVE) VS SYSTEM B (EXCLUSION-FIRST)
# ==============================================================================

def evaluate_system_a_naive(case: Dict[str, Any], kb: Dict[str, Any]) -> Tuple[str, str]:
    """
    System A (Naive Package Matcher):
    Checks only if the procedure code exists in the general package list.
    Ignores exclusion riders, waiting periods, and missing evidence checks.
    """
    default_policy = kb["PMJAY-2026-V1"]
    proc = case["procedure_code"]
    if proc in default_policy["covered_packages"]:
        clause = default_policy["covered_packages"][proc]["clause_id"]
        return "COVERED", f"Matched general package {proc} ({clause}); ignored riders/missing facts"
    return "NOT COVERED", f"Procedure {proc} not found in general package catalogue"


def evaluate_system_b_exclusion_first(case: Dict[str, Any], kb: Dict[str, Any]) -> Tuple[str, str]:
    """
    System B (Exclusion-First Deterministic Validator):
    1. Gate 1: Verify policy existence and completeness of clinical/tenure evidence.
       If missing -> Abstain and route to Intelligent RFI (MANUAL REVIEW).
    2. Gate 2: Check general package existence.
    3. Gate 3: Hard Exclusion Rider & Waiting Period Gate (Negative-First Precedence).
    4. Gate 4: Approve with explicit Policy Clause citation.
    """
    policy_id = case["policy_id"]
    if policy_id not in kb:
        return "MANUAL REVIEW", f"Abstained: Policy '{policy_id}' missing from Knowledge Base (Trigger RFI)"

    policy = kb[policy_id]
    proc = case["procedure_code"]

    # Check if procedure is in the closed-set package catalogue
    if proc not in policy["covered_packages"]:
        return "NOT COVERED", f"Rejected: Procedure '{proc}' not in covered HBP catalogue"

    pkg_rule = policy["covered_packages"][proc]

    # Gate 1: Missing Evidence / Abstention Check
    if (
        not case["evidence_complete"]
        or case["diagnosis_icd10"] is None
        or case["policy_tenure_months"] is None
    ):
        return "MANUAL REVIEW", "Abstained: Missing clinical ICD-10 or policy tenure (Trigger RFI)"

    # Gate 2: Hard Exclusion Rider Evaluation (Negative-First Precedence)
    for rider in policy["exclusion_riders"]:
        if (
            rider["target_package"] == proc
            and rider["excluded_condition_icd10"] == case["diagnosis_icd10"]
        ):
            return "NOT COVERED", f"Blocked by Exclusion Rider [{rider['rider_id']}]: {rider['description']}"

    # Gate 3: Waiting Period Gate
    req_months = pkg_rule["waiting_period_months"]
    if case["policy_tenure_months"] < req_months:
        return (
            "NOT COVERED",
            f"Blocked by Waiting Period ({case['policy_tenure_months']}m < {req_months}m required) [{pkg_rule['clause_id']}]"
        )

    # Gate 4: All negative gates cleared -> Return Covered with exact citation
    return "COVERED", f"Approved: Satisfies package [{pkg_rule['clause_id']}]; 0 exclusion riders triggered"


# ==============================================================================
# 3. BENCHMARK EXECUTION, CSV EXPORT & MATPLOTLIB VISUALIZATION
# ==============================================================================

def run_experiment() -> None:
    policy_kb, synthetic_cases = load_json_data()
    records = []
    
    for case in synthetic_cases:
        pred_a, reason_a = evaluate_system_a_naive(case, policy_kb)
        pred_b, reason_b = evaluate_system_b_exclusion_first(case, policy_kb)

        records.append({
            "case_id": case["case_id"],
            "scenario": case["scenario"],
            "procedure_code": case["procedure_code"],
            "diagnosis_icd10": case["diagnosis_icd10"],
            "policy_tenure_months": case["policy_tenure_months"],
            "ground_truth": case["ground_truth"],
            "is_ground_truth_excluded": case["is_ground_truth_excluded"],
            "naive_decision": pred_a,
            "naive_Correct": pred_a == case["ground_truth"],
            "naive_false_approval": case["is_ground_truth_excluded"] and (pred_a == "COVERED"),
            "exclusion_first_decision": pred_b,
            "exclusion_first_correct": pred_b == case["ground_truth"],
            "exclusion_first_false_approval": case["is_ground_truth_excluded"] and (pred_b == "COVERED"),
            "citation_and_audit_log": reason_b
        })

    df = pd.DataFrame(records)
    csv_filename = "policy_exclusion_case_results.csv"
    df.to_csv(csv_filename, index=False)

    # Compute Benchmark Metrics
    total_cases = len(df)
    total_excluded_cases = df["is_ground_truth_excluded"].sum()

    naive_acc = (df["naive_Correct"].sum() / total_cases) * 100
    naive_far = (df["naive_false_approval"].sum() / total_excluded_cases) * 100
    naive_mrr = ((df["naive_decision"] == "MANUAL REVIEW").sum() / total_cases) * 100

    excl_acc = (df["exclusion_first_correct"].sum() / total_cases) * 100
    excl_far = (df["exclusion_first_false_approval"].sum() / total_excluded_cases) * 100
    excl_mrr = ((df["exclusion_first_decision"] == "MANUAL REVIEW").sum() / total_cases) * 100

    print("=" * 72)
    print(" SPRINT BENCHMARK: NAIVE MATCHER VS. EXCLUSION-FIRST VALIDATOR")
    print("=" * 72)
    print(f"Total Synthetic Test Cases : {total_cases}")
    print(f"Ground-Truth Excluded Cases: {total_excluded_cases}\n")
    print(f"{'Metric':<28} | {'Naive Baseline':<18} | {'Exclusion-First (Ours)':<22}")
    print("-" * 72)
    print(f"{'Accuracy':<28} | {naive_acc:>16.2f}% | {excl_acc:>20.2f}%")
    print(f"{'False Approval Rate (FAR)':<28} | {naive_far:>16.2f}% | {excl_far:>20.2f}%")
    print(f"{'Manual-Review / RFI Rate':<28} | {naive_mrr:>16.2f}% | {excl_mrr:>20.2f}%")
    print("=" * 72)
    print(f"\n[INFO] Case-by-case audit log saved to: {csv_filename}")

    # Plot Comparison Graph
    metrics = ["Accuracy (%)", "False Approval Rate (%)", "Manual-Review / RFI Rate (%)"]
    naive_scores = [naive_acc, naive_far, naive_mrr]
    excl_scores = [excl_acc, excl_far, excl_mrr]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    rects1 = ax.bar([i - width/2 for i in x], naive_scores, width, label="System A: Naive Package Matcher", color="#e74c3c")
    rects2 = ax.bar([i + width/2 for i in x], excl_scores, width, label="System B: Exclusion-First Validator", color="#2ecc71")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Evidence-to-Policy Verification: Preventing Exclusion Hallucinations", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    # Annotate bar values
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    plot_filename = "policy_exclusion_results.png"
    plt.savefig(plot_filename)
    print(f"[INFO] Comparison chart saved to      : {plot_filename}")


if __name__ == "__main__":
    run_experiment()