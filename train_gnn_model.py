"""
train_gnn_model.py — Graph Neural Network over network topology for topology-aware root cause.

Uses same node features as master_features.pkl + graph structure (topology_links).
Model: 2-layer GCN (Graph Convolutional Network) for node-level root cause prediction.
Identifies root cause at a node (e.g. fiber cut at NE_A causing cascades on neighbours).

Inference: GNN requires graph structure; Spring Boot can call Python microservice (REST)
or use pre-computed node embeddings. We save PyTorch state_dict + config for inference.
"""

import os
import sys
import json
import logging
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch

from config import (
    configure_logging, OUTPUT_DIR, MODELS_DIR,
    GNN_HIDDEN_DIM, GNN_N_EPOCHS, GNN_BATCH_SIZE, GNN_VAL_FRAC, GNN_RANDOM_STATE,
    GNN_MAX_NODES,
)

log = configure_logging("train_gnn_model")

DATA_DIR     = OUTPUT_DIR
HIDDEN_DIM   = GNN_HIDDEN_DIM
N_EPOCHS     = GNN_N_EPOCHS
BATCH_SIZE   = GNN_BATCH_SIZE
RANDOM_STATE = GNN_RANDOM_STATE

def _load_root_cause_labels() -> list:
    """
    Load root cause labels from root_cause_labels.json (written by
    train_root_cause.py).  No hardcoded fallback — if the file is
    missing the caller gets a clear error telling them which step to run.
    """
    from config import load_root_cause_labels
    return load_root_cause_labels()


def _load_feature_cols() -> list | None:
    """
    Load the feature column list from root_cause_features.json (written by
    train_root_cause.py after XGBoost training).

    Fallback: if the file does not exist yet, return None so that
    load_graph_and_features() will derive columns directly from the
    numeric columns in master_features.pkl.  No hardcoded list.
    """
    path = os.path.join(MODELS_DIR, "root_cause_features.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    log.warning(
        "root_cause_features.json not found — GNN will derive feature "
        "columns from numeric columns in master_features.pkl.  Run "
        "train_root_cause.py first for best results."
    )
    return None


def load_graph_and_features():
    """Load master_features, topology links, build node index and edge list."""
    master_path = os.path.join(DATA_DIR, "master_features.pkl")
    links_path = os.path.join(DATA_DIR, "topology_links.pkl")
    ne_path = os.path.join(DATA_DIR, "network_elements.pkl")
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Missing {master_path}. Run feature_builder / pipeline first.")
    master = pd.read_pickle(master_path)
    ne_df = pd.read_pickle(ne_path) if os.path.exists(ne_path) else master
    links_df = pd.read_pickle(links_path) if os.path.exists(links_path) else pd.DataFrame(columns=["src", "dst"])

    # Cap node count to avoid O(n²) adjacency matrix blowing up RAM
    if GNN_MAX_NODES > 0 and len(master) > GNN_MAX_NODES:
        log.info("Capping GNN nodes: %d → %d (GNN_MAX_NODES=%d)", len(master), GNN_MAX_NODES, GNN_MAX_NODES)
        master = master.sample(GNN_MAX_NODES, random_state=42).reset_index(drop=True)

    # Node index: 0 .. n-1 by master row order (ID column)
    if "ID" not in master.columns:
        master["ID"] = np.arange(len(master))
    id_to_idx = {int(row["ID"]): i for i, row in master.iterrows()}
    n_nodes = len(master)

    # Feature matrix — prefer saved feature list from XGBoost training;
    # fall back to numeric columns in master_features.pkl.
    saved_cols = _load_feature_cols()
    if saved_cols:
        use_cols = [c for c in saved_cols if c in master.columns]
    else:
        use_cols = []
    if len(use_cols) < 5:
        use_cols = [c for c in master.select_dtypes(include=[np.number]).columns if c != "ID"][:40]
    X = master[use_cols].fillna(0).astype(np.float32).values
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # Edge index (bidirectional)
    edge_list = []
    for _, row in links_df.iterrows():
        s, d = row.get("src"), row.get("dst")
        if pd.isna(s) or pd.isna(d):
            continue
        try:
            si, di = id_to_idx.get(int(s)), id_to_idx.get(int(d))
            if si is not None and di is not None and si != di:
                edge_list.append((si, di))
                edge_list.append((di, si))
        except (TypeError, ValueError):
            continue
    edge_list = list(set(edge_list))
    if not edge_list:
        log.warning("No edges in topology — GNN will behave like MLP on nodes.")

    return {
        "X": X,
        "id_to_idx": id_to_idx,
        "idx_to_id": {v: k for k, v in id_to_idx.items()},
        "edge_list": edge_list,
        "n_nodes": n_nodes,
        "feature_cols": use_cols,
        "master": master,
    }


def build_node_labels(alarms_path: str, id_to_idx: dict, ne_df: pd.DataFrame, label_list: list):
    """From alarms_raw.pkl, assign each NE a root cause label (mode over its alarms)."""
    if not os.path.exists(alarms_path):
        log.warning("No alarms_raw.pkl — using UNKNOWN for all nodes.")
        n = len(id_to_idx)
        return np.full(n, label_list.index("UNKNOWN") if "UNKNOWN" in label_list else 0, dtype=np.int64)
    alarms = pd.read_pickle(alarms_path)
    ne_key = "NE_ID_FK" if "NE_ID_FK" in alarms.columns else "entity_id"
    rc_col = "ROOT_CAUSE" if "ROOT_CAUSE" in alarms.columns else None
    if rc_col is None:
        n = len(id_to_idx)
        return np.full(n, label_list.index("UNKNOWN") if "UNKNOWN" in label_list else 0, dtype=np.int64)
    alarms[ne_key] = alarms[ne_key].astype(str)
    # NE_ID in master may be string; id_to_idx is keyed by integer ID. We need NE_ID -> label.
    # master has ID (int) and often NE_ID (str). id_to_idx is ID -> idx. So we need NE_ID -> ID from ne_df.
    ne_id_to_internal_id = {}
    if "ID" in ne_df.columns and "NE_ID" in ne_df.columns:
        ne_id_to_internal_id = ne_df.set_index("NE_ID")["ID"].astype(int).to_dict()
    agg = alarms.groupby(ne_key)[rc_col].apply(lambda s: s.mode().iloc[0] if len(s) else "UNKNOWN").to_dict()
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
    default_idx = label_to_idx.get("UNKNOWN", 0)
    n = len(id_to_idx)
    y = np.full(n, default_idx, dtype=np.int64)
    for ne_id, rc in agg.items():
        internal_id = ne_id_to_internal_id.get(ne_id) or ne_id_to_internal_id.get(str(ne_id))
        if internal_id is None:
            try:
                internal_id = int(ne_id)
            except (TypeError, ValueError):
                continue
        idx = id_to_idx.get(internal_id)
        if idx is not None and rc in label_to_idx:
            y[idx] = label_to_idx[rc]
    return y


def build_adjacency(n_nodes: int, edge_list: list):
    """Normalized adjacency D^{-1/2}(A+I)D^{-1/2} as a sparse COO tensor.

    Sparse representation avoids allocating an n×n dense matrix — critical
    when n is in the thousands but the graph has only a few hundred edges.
    """
    # Include self-loops
    all_edges = list(edge_list) + [(i, i) for i in range(n_nodes)]
    rows = np.array([e[0] for e in all_edges], dtype=np.int64)
    cols = np.array([e[1] for e in all_edges], dtype=np.int64)

    # Degree = number of non-zero entries per row
    degree = np.bincount(rows, minlength=n_nodes).astype(np.float32)
    d_inv_sqrt = np.power(degree + 1e-6, -0.5)
    vals = (d_inv_sqrt[rows] * d_inv_sqrt[cols]).astype(np.float32)

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values  = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = torch.nn.Linear(in_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        # Simplified GCN: H1 = relu(adj @ x @ W1), H2 = adj @ H1 @ W2
        h = torch.relu(torch.sparse.mm(adj, self.conv1(x)))
        out = torch.sparse.mm(adj, self.conv2(h))
        return out


def train_and_save(data: dict, y: np.ndarray, label_list: list):
    """Train 2-layer GCN and save state_dict + config.

    A random node-level train/val split is created so that the reported
    accuracy is on held-out nodes, not the training set.
    """
    X = data["X"]
    edge_list = data["edge_list"]
    n_nodes = data["n_nodes"]

    # ── Train / validation split (node-level mask)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(n_nodes)
    rng.shuffle(idx)
    n_val   = max(1, int(n_nodes * GNN_VAL_FRAC))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    log.info(f"GNN split: {train_mask.sum()} train nodes, {val_mask.sum()} val nodes")

    adj = build_adjacency(n_nodes, edge_list)
    # Sparse tensors don't support MPS yet — always use CUDA or CPU for GNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t   = torch.from_numpy(X).to(device)
    adj   = adj.to(device)
    y_t   = torch.from_numpy(y).long().to(device)
    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)

    n_classes = len(label_list)
    in_dim    = X.shape[1]
    model     = GCN(in_dim, HIDDEN_DIM, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_acc = 0.0
    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        logits = model(x_t, adj)
        # Loss computed on training nodes only
        loss = torch.nn.functional.cross_entropy(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                all_logits = model(x_t, adj)
                train_acc = (all_logits[train_mask].argmax(dim=1) == y_t[train_mask]).float().mean().item()
                val_acc   = (all_logits[val_mask].argmax(dim=1)   == y_t[val_mask]).float().mean().item()
                best_val_acc = max(best_val_acc, val_acc)
            log.info(
                f"Epoch {epoch+1}/{N_EPOCHS} loss={loss.item():.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )
            model.train()

    model.eval()
    with torch.no_grad():
        logits = model(x_t, adj)
        val_acc = (logits[val_mask].argmax(dim=1) == y_t[val_mask]).float().mean().item()
    log.info(f"Final validation accuracy: {val_acc:.4f}  (best={best_val_acc:.4f})")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    state_path = os.path.join(MODELS_DIR, "gnn_root_cause.pt")
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "in_dim": in_dim,
        "hidden_dim": HIDDEN_DIM,
        "n_classes": n_classes,
    }, state_path)
    log.info(f"Saved model: {state_path}")

    config_path = os.path.join(MODELS_DIR, "gnn_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "id_to_idx": {str(k): v for k, v in data["id_to_idx"].items()},
            "feature_cols": data["feature_cols"],
            "label_list": label_list,
            "n_nodes": n_nodes,
            "edge_list": data["edge_list"],
        }, f, indent=2)
    log.info(f"Saved config: {config_path}")

    labels_path = os.path.join(MODELS_DIR, "gnn_labels.json")
    with open(labels_path, "w") as f:
        json.dump(label_list, f)
    return model


def run():
    label_list = _load_root_cause_labels()
    data = load_graph_and_features()
    alarms_path = os.path.join(DATA_DIR, "alarms_raw.pkl")
    ne_path = os.path.join(DATA_DIR, "network_elements.pkl")
    ne_df = pd.read_pickle(ne_path) if os.path.exists(ne_path) else data.get("master", pd.DataFrame())
    y = build_node_labels(alarms_path, data["id_to_idx"], ne_df, label_list)
    log.info(f"GNN: {data['n_nodes']} nodes, {len(data['edge_list'])} edges, {len(data['feature_cols'])} features, {len(label_list)} classes")
    train_and_save(data, y, label_list)


if __name__ == "__main__":
    run()
