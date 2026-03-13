#!/usr/bin/env python3
"""
Stream Inference Test — stream alarms from ALARM table and run model inference.

Reads alarms in batches (simulating a stream), joins with NE features,
runs root-cause ONNX model and propagation lookup, and writes results to CSV.

Run from pipeline directory:
  .venv/bin/python tests/stream_inference_test.py
  .venv/bin/python tests/stream_inference_test.py --limit 2000 --batch-size 500
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add pipeline root so we can import config and use pipeline paths
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)

import numpy as np
import pandas as pd

# Optional: reduce noise from missing optional deps
import logging
logging.getLogger("onnxruntime").setLevel(logging.WARNING)


def load_pipeline_artifacts():
    """Load feature data and model artifacts from the pipeline output dirs."""
    from config import OUTPUT_DIR, DEPLOY_DIR, MODELS_DIR

    processed = OUTPUT_DIR

    def _path(name):
        """Resolve artifact path: try deploy first, then MODELS_DIR."""
        for d in (DEPLOY_DIR, MODELS_DIR):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return os.path.join(DEPLOY_DIR, name)

    master_path = os.path.join(processed, "master_features.pkl")
    ne_path = os.path.join(processed, "network_elements.pkl")

    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"Run the pipeline first to create {master_path}\n"
            "  cd pipeline && .venv/bin/python run_pipeline.py --mode full"
        )
    if not os.path.exists(ne_path):
        raise FileNotFoundError(f"Missing {ne_path}. Run pipeline first.")

    master_df = pd.read_pickle(master_path)
    ne_df = pd.read_pickle(ne_path)

    # NE lookup: ENTITY_ID (from alarm) matches NE_ID (string) in network_elements
    # Keep ID and NE_ID for join
    ne_lookup = ne_df[["ID", "NE_ID"]].drop_duplicates("NE_ID")
    ne_lookup = ne_lookup.rename(columns={"ID": "ne_pk_id"})

    with open(_path("root_cause_features.json")) as f:
        feature_cols = json.load(f)
    with open(_path("root_cause_labels.json")) as f:
        label_list = json.load(f)
    with open(_path("alarm_propagation_rules.json")) as f:
        propagation_rules = json.load(f)

    onnx_path = _path("root_cause_classifier.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Missing root_cause_classifier.onnx. Run pipeline first.")

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    return {
        "master_df": master_df,
        "ne_lookup": ne_lookup,
        "feature_cols": feature_cols,
        "label_list": label_list,
        "propagation_rules": propagation_rules,
        "session": session,
        "input_name": input_name,
    }


def stream_alarms(engine, batch_size: int, limit: int, order_col: str = "OPEN_TIME"):
    """Yield batches of alarms from ALARM table ordered by time."""
    query = f"""
        SELECT ALARM_ID_PK, OPEN_TIME, ENTITY_ID, ALARM_CODE, SEVERITY, ALARM_NAME, ALARM_STATUS
        FROM ALARM
        ORDER BY {order_col}
    """
    offset = 0
    while True:
        batch_query = query + f" LIMIT {batch_size} OFFSET {offset}"
        df = pd.read_sql(batch_query, engine)
        if df.empty:
            break
        yield df
        offset += len(df)
        if limit and offset >= limit:
            break
        if len(df) < batch_size:
            break


def build_feature_row(master_row, feature_cols: list) -> np.ndarray:
    """Build one float32 feature vector in the order expected by the ONNX model."""
    vec = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in master_row.index:
            val = master_row[col]
            if pd.isna(val):
                vec[i] = 0.0
            else:
                try:
                    vec[i] = float(val)
                except (TypeError, ValueError):
                    vec[i] = 0.0
        else:
            vec[i] = 0.0
    return vec


def run_inference(session, input_name: str, features_batch: np.ndarray, n_classes: int):
    """
    Run ONNX root-cause classifier on a batch of feature vectors.
    Prediction is always derived from the model output (no hardcoding):
    - If the model returns probabilities (shape [N, n_classes]), use argmax for class and max for confidence.
    - If the model returns (label, prob), use them; confidence is the probability of the predicted class.
    """
    if len(features_batch) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    outputs = session.run(None, {input_name: features_batch.astype(np.float32)})
    # Find probability output: shape (N, n_classes) float array (always derive class from this)
    prob_out = None
    for o in outputs:
        if hasattr(o, "shape") and len(o.shape) == 2 and o.shape[1] == n_classes:
            prob_out = np.asarray(o, dtype=np.float64)
            break
    if prob_out is None and len(outputs) == 1:
        prob_out = np.asarray(outputs[0], dtype=np.float64)
    if prob_out is not None and len(prob_out.shape) == 2:
        pred_indices = np.argmax(prob_out, axis=1)
        confidences = np.max(prob_out, axis=1)
        return pred_indices, confidences
    # Fallback: first output = label index, second = probabilities
    pred_indices = np.asarray(outputs[0]).flatten().astype(np.int64)
    probs = outputs[1] if len(outputs) > 1 else outputs[0]
    if isinstance(probs, np.ndarray) and len(probs.shape) >= 2:
        confidences = np.max(probs, axis=1)
    elif isinstance(probs, np.ndarray):
        confidences = np.array([float(np.max(probs))] * len(pred_indices))
    else:
        confidences = np.array([max(p.values()) if isinstance(p, dict) else float(p) for p in probs])
    return pred_indices, confidences


def get_propagation(propagation_rules: dict, alarm_code: str, max_consequents: int = 5):
    """Return list of likely consequent alarm codes for this alarm code."""
    code_str = str(alarm_code)
    if code_str not in propagation_rules:
        return []
    rules = propagation_rules[code_str]
    out = []
    for r in rules[:max_consequents]:
        if isinstance(r, dict) and "consequent" in r:
            out.append(r["consequent"])
        else:
            out.append(str(r))
    return out


def main():
    parser = argparse.ArgumentParser(description="Stream alarms from DB and run model inference")
    parser.add_argument("--batch-size", type=int, default=1000, help="Alarms per batch")
    parser.add_argument("--limit", type=int, default=5000, help="Max alarms to process (0 = all)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: tests/output)")
    parser.add_argument("--order", type=str, default="OPEN_TIME", help="Order column for streaming")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PIPELINE_DIR, "tests", "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading pipeline artifacts (master features, NE lookup, ONNX model, rules)...")
    artifacts = load_pipeline_artifacts()
    master_df = artifacts["master_df"]
    ne_lookup = artifacts["ne_lookup"]
    feature_cols = artifacts["feature_cols"]
    label_list = artifacts["label_list"]
    propagation_rules = artifacts["propagation_rules"]
    session = artifacts["session"]
    input_name = artifacts["input_name"]

    from config import get_engine
    engine = get_engine()

    print(f"Streaming alarms from ALARM (batch_size={args.batch_size}, limit={args.limit or 'all'})...")
    results = []
    total_processed = 0
    total_skipped_no_ne = 0

    for batch_df in stream_alarms(engine, args.batch_size, args.limit or 0, args.order):
        # Respect limit: take only up to (limit - total_processed) rows
        if args.limit and total_processed + len(batch_df) > args.limit:
            batch_df = batch_df.head(args.limit - total_processed)
        # Join alarms with NE lookup: ENTITY_ID (alarm) = NE_ID (ne_lookup)
        batch_df = batch_df.merge(
            ne_lookup,
            left_on="ENTITY_ID",
            right_on="NE_ID",
            how="left",
        )
        # Drop NE_ID to avoid duplicate column when merging with master
        if "NE_ID" in batch_df.columns:
            batch_df = batch_df.drop(columns=["NE_ID"])

        # Alarms with no matching NE get ne_pk_id NaN — skip them for feature lookup
        has_ne = batch_df["ne_pk_id"].notna()
        batch_with_ne = batch_df[has_ne].copy()
        skipped = (~has_ne).sum()
        total_skipped_no_ne += skipped

        if batch_with_ne.empty:
            total_processed += len(batch_df)
            continue

        # Merge with master features on ne_pk_id = ID
        merged = batch_with_ne.merge(
            master_df,
            left_on="ne_pk_id",
            right_on="ID",
            how="left",
            suffixes=("", "_mf"),
        )

        # Build feature matrix
        feature_rows = []
        for _, row in merged.iterrows():
            vec = build_feature_row(row, feature_cols)
            feature_rows.append(vec)
        feature_matrix = np.vstack(feature_rows)

        # Run ONNX (prediction from model only; label_list order must match training)
        pred_indices, confidences = run_inference(
            session, input_name, feature_matrix, n_classes=len(label_list)
        )

        # Build result rows (predicted_root_cause = model output only, from argmax of ONNX probabilities)
        for i, (_, row) in enumerate(merged.iterrows()):
            idx = int(pred_indices[i])
            label = label_list[idx] if 0 <= idx < len(label_list) else f"UNKNOWN_{idx}"
            conf = float(confidences[i])
            alarm_code = row["ALARM_CODE"]
            prop_list = get_propagation(propagation_rules, alarm_code)
            results.append({
                "alarm_id": row["ALARM_ID_PK"],
                "open_time": row["OPEN_TIME"],
                "entity_id": row["ENTITY_ID"],
                "alarm_code": alarm_code,
                "alarm_name": row.get("ALARM_NAME", ""),
                "severity": row.get("SEVERITY", ""),
                "predicted_root_cause": label,
                "confidence": round(conf, 4),
                "propagation_count": len(prop_list),
                "propagation_alarms": json.dumps(prop_list),
            })

        total_processed += len(batch_df)
        print(f"  Processed {total_processed} alarms, results so far: {len(results)}")

        if args.limit and total_processed >= args.limit:
            break

    # Write CSV
    out_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "stream_inference_results.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path} ({len(out_df)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("STREAM INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Alarms processed (with NE match): {len(results)}")
    print(f"  Alarms skipped (no NE match):     {total_skipped_no_ne}")
    print(f"  Output CSV:                       {csv_path}")
    if not out_df.empty:
        print("\n  Top 10 predicted root causes:")
        top = out_df["predicted_root_cause"].value_counts().head(10)
        for cause, count in top.items():
            print(f"    {cause}: {count}")
        print("\n  Sample rows (first 5):")
        print(out_df.head().to_string())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
