# Tri-Fold Fraud Engine — Small Demonstration

This is a small, explainable feasibility experiment—not a production fraud detector or a system that confirms fraud.

## Files

- `claims.csv` — the original 20-row input CSV, kept unmodified.
- `app.py` — orchestration script. It first copies `claims.csv` to `fraud_results.csv`, then runs Layer 1, Layer 2, and Layer 3 in sequence.
- `layer1.py` — rule-based checks for overlapping admission periods, clinical/anatomical consistency (when relevant evidence fields are supplied), and possible unbundled billing (when item-level/package reference fields are supplied).
- `layer2.py` — Isolation Forest on bill amount, surgery duration, and package category; also checks whether bill amount or duration is more than three group standard deviations from the group median when the reference group is large enough.
- `layer3.py` — NetworkX claim–doctor–hospital graph and overlap detection for a doctor's surgeries at different hospitals.
- `demo_tests.py` — optional, isolated test scenarios showing each mechanism can trigger. It does not modify `claims.csv` or `fraud_results.csv`.
- `layer3_relationship_graph.png` — graph generated from the actual supplied input after running `app.py`.
- `demo_collision_graph.png` — graph generated from a clearly labelled synthetic collision test, not from the original 20 claims.
- `fraud_results.csv` — final working copy, created and updated by the three layers when `app.py` runs.

## Run

From this folder, install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python app.py
```

To run the controlled test scenarios separately:

```bash
python demo_tests.py
```

## Important data limitations

The supplied CSV has 11 columns only. It does not include patient sex/anatomical context, billed line items, package inclusion rules, or region. Therefore, Layer 1 reports the biological/package checks as unavailable unless those extra fields are supplied in a future dataset. It does not invent values for missing fields.

The supplied CSV uses distinct patient and doctor IDs and does not itself include overlapping admissions for one patient or overlapping surgeries for the same doctor across hospitals. Accordingly, the main run may not find a Layer 1 admission overlap or Layer 3 collision. The separate `demo_tests.py` uses temporary in-memory rows to demonstrate those paths without changing the original or final output CSV.

Layer 2's package groups are only three or four records each. A >3-SD-from-median group test is therefore marked not evaluated for those groups. Since the CSV also lacks a region field, it cannot perform true regional comparison. Isolation Forest still runs as an exploratory model across the small demonstration dataset; its anomaly labels are relative to this dataset and must not be described as confirmed fraud or measured real-world accuracy.

A doctor scheduling conflict is an investigation lead, not proof of collusion. Clinical rules must be reviewed by qualified domain experts and allow legitimate exceptions.
