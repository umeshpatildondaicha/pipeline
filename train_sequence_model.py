"""
train_sequence_model.py — LSTM/Transformer on alarm sequences for root cause prediction.

Input: incident alarm_sequence (alarm codes in time order), padded/truncated to max_len (default 20).
Output: root cause class (same 9 classes as root_cause_classifier).

Exports to ONNX for Spring Boot NocInferenceService.
"""

import os
import sys
import json
import logging
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from config import (
    configure_logging, OUTPUT_DIR, MODELS_DIR, DEPLOY_DIR,
    SEQ_MAX_LEN, SEQ_EMBED_DIM, SEQ_HIDDEN_SIZE, SEQ_BATCH_SIZE,
    SEQ_VALID_FRAC, SEQ_N_EPOCHS, XGB_RANDOM_STATE,
)

log = configure_logging("train_sequence_model")

DATA_DIR = OUTPUT_DIR
MAX_SEQ_LEN  = SEQ_MAX_LEN
EMBED_DIM    = SEQ_EMBED_DIM
HIDDEN_SIZE  = SEQ_HIDDEN_SIZE
N_EPOCHS     = SEQ_N_EPOCHS
BATCH_SIZE   = SEQ_BATCH_SIZE
VALID_FRAC   = SEQ_VALID_FRAC
RANDOM_STATE = XGB_RANDOM_STATE

# PAD=0, UNK=1, then alarm codes
PAD_IDX = 0
UNK_IDX = 1


def _load_incidents():
    path = os.path.join(DATA_DIR, "incidents.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run alarm_correlator.py first.")
    return pd.read_pickle(path)


def _load_root_cause_labels():
    """Load root cause labels from the canonical JSON written by train_root_cause.py."""
    from config import load_root_cause_labels
    try:
        return load_root_cause_labels()
    except FileNotFoundError as exc:
        log.warning(
            "%s  — run train_root_cause.py (or run_pipeline.py --mode full) first "
            "to generate root_cause_labels.json from real data.",
            exc,
        )
        # Return an empty list so the caller fails clearly rather than silently
        # using a stale hardcoded set.
        raise


def build_vocab(incidents_df: pd.DataFrame):
    """Build alarm_code -> index. 0=pad, 1=unk, then all seen codes."""
    codes = set()
    for seq in incidents_df["alarm_sequence"]:
        if isinstance(seq, (list, tuple)):
            for c in seq:
                codes.add(str(c).strip())
        elif pd.notna(seq):
            codes.add(str(seq).strip())
    codes.discard("")
    code_list = sorted(codes)
    code_to_idx = {"__PAD__": PAD_IDX, "__UNK__": UNK_IDX}
    for i, c in enumerate(code_list):
        code_to_idx[c] = i + 2
    return code_to_idx


def sequence_to_indices(seq, code_to_idx, max_len: int):
    """Convert list of alarm codes to list of ints, pad/truncate to max_len."""
    if not seq:
        return [PAD_IDX] * max_len
    indices = [code_to_idx.get(str(c).strip(), UNK_IDX) for c in seq]
    if len(indices) > max_len:
        indices = indices[-max_len:]
    elif len(indices) < max_len:
        indices = [PAD_IDX] * (max_len - len(indices)) + indices
    return indices


def prepare_data(incidents_df: pd.DataFrame, code_to_idx: dict, label_list: list):
    """Return X (np int64, shape (n, max_len)), y (np int64), and mask of valid rows."""
    max_len = MAX_SEQ_LEN
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
    X_list, y_list = [], []
    for _, row in incidents_df.iterrows():
        seq = row.get("alarm_sequence") or []
        rc = str(row.get("dominant_root_cause", "UNKNOWN")).strip()
        if rc not in label_to_idx:
            rc = "UNKNOWN" if "UNKNOWN" in label_to_idx else label_list[0]
        y_list.append(label_to_idx[rc])
        X_list.append(sequence_to_indices(seq, code_to_idx, max_len))
    X = np.array(X_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.int64)
    return X, y, code_to_idx, label_list


def train_and_export(
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    n_classes: int,
    code_to_idx: dict,
    label_list: list,
):
    """Train LSTM, save artifacts and ONNX."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split

    class SequenceRootCauseLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, n_classes, pad_idx=0):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            # x: (batch, seq_len)
            emb = self.embed(x)
            _, (h_n, _) = self.lstm(emb)
            out = self.fc(h_n.squeeze(0))
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceRootCauseLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        n_classes=n_classes,
        pad_idx=PAD_IDX,
    ).to(device)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)
    n_val = max(1, int(len(dataset) * VALID_FRAC))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_STATE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log.info("Training LSTM (%d epochs, batch=%d)...", N_EPOCHS, BATCH_SIZE)
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                pred = logits.argmax(dim=1)
                total += by.size(0)
                correct += (pred == by).sum().item()
        acc = correct / total if total else 0
        log.info(f"Epoch {epoch+1}/{N_EPOCHS} train_loss={train_loss/len(train_loader):.4f} val_acc={acc:.4f}")

    # Save vocab and labels for inference
    vocab_path = os.path.join(MODELS_DIR, "sequence_model_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(code_to_idx, f, indent=0)
    log.info(f"Saved vocab: {vocab_path}")

    labels_path = os.path.join(MODELS_DIR, "sequence_model_labels.json")
    with open(labels_path, "w") as f:
        json.dump(label_list, f)
    log.info(f"Saved labels: {labels_path}")

    # ONNX export: input (batch, seq_len) int64
    model.eval()
    dummy = torch.randint(0, vocab_size, (1, MAX_SEQ_LEN), device=device)
    onnx_path = os.path.join(MODELS_DIR, "root_cause_sequence.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["alarm_sequence"],
        output_names=["logits"],
        dynamic_axes={"alarm_sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    log.info(f"ONNX saved: {onnx_path}")

    # Copy to deploy dir for Spring Boot
    if os.path.isdir(DEPLOY_DIR):
        import shutil
        for name in ["root_cause_sequence.onnx", "sequence_model_vocab.json", "sequence_model_labels.json"]:
            src = os.path.join(MODELS_DIR, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(DEPLOY_DIR, name))
        log.info(f"Copied sequence artifacts to {DEPLOY_DIR}")

    return model, onnx_path


def run():
    incidents_df = _load_incidents()
    if incidents_df.empty or "alarm_sequence" not in incidents_df.columns:
        log.error("No incidents or missing alarm_sequence. Run alarm_correlator.py first.")
        return

    label_list = _load_root_cause_labels()
    code_to_idx = build_vocab(incidents_df)
    vocab_size = len(code_to_idx)
    n_classes = len(label_list)
    log.info(f"Vocab size={vocab_size}, classes={n_classes}, incidents={len(incidents_df)}")

    X, y, _, _ = prepare_data(incidents_df, code_to_idx, label_list)
    train_and_export(X, y, vocab_size, n_classes, code_to_idx, label_list)


if __name__ == "__main__":
    run()
