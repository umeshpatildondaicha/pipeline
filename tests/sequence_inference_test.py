#!/usr/bin/env python3
"""
Sequence inference test — run root_cause_sequence.onnx on sample alarm sequences.

Loads sequence_model_vocab.json, sequence_model_labels.json, and root_cause_sequence.onnx
from deploy/, runs inference on a few sample sequences, and prints predicted root cause + confidence.

Run from pipeline directory (after training sequence model):
  .venv/bin/python tests/sequence_inference_test.py
  .venv/bin/python tests/sequence_inference_test.py --sequence 7459 8588 7460
"""

import argparse
import json
import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)

from config import DEPLOY_DIR, MODELS_DIR

MAX_SEQ_LEN = 20
PAD_IDX = 0
UNK_IDX = 1


def _path(name):
    # Prefer MODELS_DIR for ONNX (single file); deploy may have external .data refs
    dirs = (MODELS_DIR, DEPLOY_DIR) if name.endswith(".onnx") else (DEPLOY_DIR, MODELS_DIR)
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def load_artifacts():
    onnx_path = _path("root_cause_sequence.onnx")
    vocab_path = _path("sequence_model_vocab.json")
    labels_path = _path("sequence_model_labels.json")
    if not onnx_path or not vocab_path or not labels_path:
        raise FileNotFoundError(
            "Missing sequence model artifacts. Run train_sequence_model.py (or run_pipeline.py --mode test) first."
        )
    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(labels_path) as f:
        labels = json.load(f)
    return onnx_path, vocab, labels


def sequence_to_indices(alarm_codes, vocab, max_len=MAX_SEQ_LEN):
    n = len(alarm_codes)
    take = min(n, max_len)
    pad = max_len - take
    start = n - take
    out = [PAD_IDX] * pad
    for i in range(take):
        code = str(alarm_codes[start + i]).strip()
        out.append(vocab.get(code, UNK_IDX))
    return out


def run_inference(onnx_path, vocab, labels, sequences):
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    results = []
    for seq in sequences:
        indices = sequence_to_indices(seq, vocab)
        arr = np.array([indices], dtype=np.int64)
        out = sess.run(None, {input_name: arr})
        logits = out[0][0]
        pred_idx = int(logits.argmax())
        confidence = 1.0 / (1.0 + np.exp(-logits[pred_idx])) if logits.size else 0.5
        if pred_idx < len(labels):
            root_cause = labels[pred_idx]
        else:
            root_cause = "UNKNOWN"
        results.append((root_cause, float(confidence), seq))
    return results


def main():
    ap = argparse.ArgumentParser(description="Test sequence model inference")
    ap.add_argument("--sequence", nargs="+", default=["7459", "8588", "7460", "7461"],
                    help="Alarm code sequence (e.g. 7459 8588 7460)")
    args = ap.parse_args()

    onnx_path, vocab, labels = load_artifacts()
    sequences = [args.sequence]
    results = run_inference(onnx_path, vocab, labels, sequences)
    print("Sequence model inference test")
    print("  ONNX:", onnx_path)
    print("  Vocab size:", len(vocab), "  Labels:", len(labels))
    for root_cause, conf, seq in results:
        print(f"  Sequence {seq[:5]}{'...' if len(seq) > 5 else ''} -> {root_cause} (conf={conf:.3f})")
    print("OK")


if __name__ == "__main__":
    main()
