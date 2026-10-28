"""Layer 3: NetworkX relationship graph and simultaneous cross-hospital surgery check."""
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {"claim_id", "doctor_id", "hospital_id", "surgery_start", "surgery_end"}


def find_collisions(df: pd.DataFrame):
    """Return a list of (index_a, index_b, explanation) for overlapping doctor schedules."""
    events = df.copy()
    events["_start"] = pd.to_datetime(events["surgery_start"], errors="coerce")
    events["_end"] = pd.to_datetime(events["surgery_end"], errors="coerce")
    collisions = []
    for doctor_id, group in events.groupby("doctor_id", dropna=True):
        records = list(group.iterrows())
        for pos in range(len(records)):
            idx_a, a = records[pos]
            if pd.isna(a["_start"]) or pd.isna(a["_end"]) or a["_end"] <= a["_start"]:
                continue
            for next_pos in range(pos + 1, len(records)):
                idx_b, b = records[next_pos]
                if pd.isna(b["_start"]) or pd.isna(b["_end"]) or b["_end"] <= b["_start"]:
                    continue
                if str(a["hospital_id"]) == str(b["hospital_id"]):
                    continue
                overlap = max(a["_start"], b["_start"]) < min(a["_end"], b["_end"])
                if overlap:
                    explanation = (
                        f"Doctor {doctor_id} has overlapping surgery intervals across hospitals "
                        f"{a['hospital_id']} ({a['claim_id']}) and {b['hospital_id']} ({b['claim_id']})."
                    )
                    collisions.append((idx_a, idx_b, explanation))
    return collisions


def build_relationship_graph(df: pd.DataFrame, collisions=None):
    graph = nx.Graph()
    for _, row in df.iterrows():
        claim = f"claim:{row['claim_id']}"
        doctor = f"doctor:{row['doctor_id']}"
        hospital = f"hospital:{row['hospital_id']}"
        graph.add_node(claim, kind="claim")
        graph.add_node(doctor, kind="doctor")
        graph.add_node(hospital, kind="hospital")
        graph.add_edge(doctor, claim, relation="billed_for")
        graph.add_edge(claim, hospital, relation="submitted_by")
    for idx_a, idx_b, _ in (collisions or []):
        claim_a = f"claim:{df.loc[idx_a, 'claim_id']}"
        claim_b = f"claim:{df.loc[idx_b, 'claim_id']}"
        graph.add_edge(claim_a, claim_b, relation="SIMULTANEOUS_CROSS_HOSPITAL_SURGERY", collision=True)
    return graph


def draw_graph(graph, output_path, title):
    output_path = Path(output_path)
    plt.figure(figsize=(15, 10))
    positions = nx.spring_layout(graph, seed=42, k=0.75)
    node_colors = []
    for node in graph.nodes:
        kind = graph.nodes[node].get("kind")
        node_colors.append({"doctor": "#4C78A8", "hospital": "#72B7B2", "claim": "#F2CF5B"}.get(kind, "#999999"))
    nx.draw_networkx_nodes(graph, positions, node_size=420, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(graph, positions, alpha=0.6, width=1.0)
    collision_edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get("collision")]
    if collision_edges:
        nx.draw_networkx_edges(graph, positions, edgelist=collision_edges, edge_color="#D62728", width=3.0)
    labels = {node: node.replace("doctor:", "DR:").replace("hospital:", "H:").replace("claim:", "CL:") for node in graph.nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def apply_layer3(df: pd.DataFrame, graph_path=None) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Layer 3 cannot run; missing required columns: {sorted(missing)}")
    df = df.copy()
    collisions = find_collisions(df)
    df["layer3_flag"] = "NO"
    df["layer3_collision_count"] = 0
    df["layer3_reason"] = "No simultaneous cross-hospital surgery collision detected for this doctor in the supplied data."
    for idx_a, idx_b, explanation in collisions:
        for idx in (idx_a, idx_b):
            df.loc[idx, "layer3_flag"] = "YES"
            df.loc[idx, "layer3_collision_count"] += 1
            existing = str(df.loc[idx, "layer3_reason"])
            if existing.startswith("No simultaneous"):
                existing = ""
            df.loc[idx, "layer3_reason"] = (existing + " " + explanation).strip()
    if graph_path:
        graph = build_relationship_graph(df, collisions)
        title = "Claim–Doctor–Hospital Relationship Graph (red edges = detected schedule collisions)"
        draw_graph(graph, graph_path, title)
    return df


def run_layer3(csv_path, graph_path=None):
    path = Path(csv_path)
    updated = apply_layer3(pd.read_csv(path), graph_path=graph_path)
    updated.to_csv(path, index=False)
    print(f"Layer 3 updated: {path.name}")
    return updated


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    run_layer3(folder / "fraud_results.csv", folder / "layer3_relationship_graph.png")
