import pandas as pd
from tqdm import tqdm
from .config import FEATURE_COLS, OUT_PATH
import os

def create_edge_features(row):
    return [float(row[col]) for col in FEATURE_COLS]

def build_graph(df):
    edges = []
    edges_file = os.path.join(OUT_PATH, "edges_df.csv")
    if os.path.exists(edges_file):
        print(f"Loading cached edges: {edges_file}")
        return pd.read_csv(edges_file)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating edges"):
        tx_node = f"tx_{row.TransactionID}"
        edge_feat = create_edge_features(row)
        ts = int(row.TransactionDT)
        label = int(row.isFraud)

        # Primary edges
        if row.card1 != -999:
            edges.append({
                "src": f"card1_{int(row.card1)}",
                "dst": tx_node,
                "timestamp": ts,
                "edge_feat": edge_feat,
                "label": label,
                "edge_type": 0
            })

        # Context edges
        if row.addr1 != -999:
            edges.append({
                "src": tx_node,
                "dst": f"addr_{int(row.addr1)}",
                "timestamp": ts,
                "edge_feat": edge_feat,
                "label": -1,
                "edge_type": 1
            })

        if row.card1 != -999 and row.P_emaildomain != "Unknown":
            edges.append({
                "src": f"card1_{int(row.card1)}",
                "dst": f"email_{row.P_emaildomain}",
                "timestamp": ts,
                "edge_feat": edge_feat,
                "label": -1,
                "edge_type": 2
            })

        if row.card1 != -999:
            edges.append({
                "src": f"card1_{int(row.card1)}",
                "dst": row.device_node,
                "timestamp": ts,
                "edge_feat": edge_feat,
                "label": -1,
                "edge_type": 3
            })

        if row.P_emaildomain != "Unknown" and row.addr1 != -999:
            edges.append({
                "src": f"email_{row.P_emaildomain}",
                "dst": f"addr_{int(row.addr1)}",
                "timestamp": ts,
                "edge_feat": edge_feat,
                "label": -1,
                "edge_type": 4
            })

    edges_df = pd.DataFrame(edges)
    edges_df.sort_values("timestamp", inplace=True)
    edges_df.reset_index(drop=True, inplace=True)
    edges_df.to_csv(edges_file, index=False)
    cleanup_memory()
    print(f"Created {len(edges_df):,} edges, Labeled: {(edges_df['label'] != -1).sum():,}")
    return edges_df
