"""
MODEL 1: Root Cause Classifier
==============================
When an alarm fires on a NE, predict:
  - What is the ROOT CAUSE category
  - Confidence score

Input features: NE features + graph position + link health + hierarchy
Output        : root_cause_label + confidence

Algorithm: XGBoost (fast, explainable, handles mixed types well)
Export    : ONNX for Spring Boot inference
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import configure_logging, OUTPUT_DIR, MODELS_DIR

log = configure_logging("train_root_cause")
DATA_DIR = OUTPUT_DIR
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS — derived from your actual schema
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # NE static (encoded)
    'NE_TYPE_ENC', 'TECHNOLOGY_ENC', 'VENDOR_ENC', 'DOMAIN_ENC',
    'NE_STATUS_ENC', 'OPERATIONAL_STATE_ENC', 'CATEGORY_ENC',
    'NE_STAGE_ENC', 'ADMIN_STATE_ENC',

    # NE age and warranty
    'NE_AGE_DAYS', 'WARRANTY_REMAINING_DAYS', 'WARRANTY_EXPIRED',
    'TECH_GENERATION', 'IS_VIRTUAL_NUM', 'IS_SECURED_NUM',

    # Protocol health
    'BGP_STATUS_NUM', 'OSPF_STATUS_NUM', 'LLDP_STATUS_NUM',
    'PROTOCOL_HEALTH_SCORE',

    # Graph topology
    'GRAPH_DEGREE', 'BETWEENNESS', 'CLUSTERING_COEF',
    'COMPONENT_SIZE', 'IS_LEAF', 'IS_ISOLATED', 'TOPO_LAYER',

    # Hierarchy
    'HIERARCHY_DEPTH', 'IS_ROOT_NODE', 'GEO_COMPLETENESS',
    'GEOGRAPHY_L1_ID_FK', 'GEOGRAPHY_L2_ID_FK',
    'GEOGRAPHY_L3_ID_FK', 'GEOGRAPHY_L4_ID_FK',

    # ISIS link health (most powerful real-time signal)
    'SRC_MAX_UTILIZATION', 'SRC_AVG_UTILIZATION',
    'SRC_MAX_ERROR_RATE', 'SRC_MAX_DROP_RATE',
    'SRC_CRITICAL_LINKS', 'SRC_WARNING_LINKS',
    'DST_MAX_UTILIZATION', 'DST_MAX_ERROR_RATE',
    'LINK_HEALTH_SCORE',

    # RAN-specific (for radio alarms)
    'AZIMUTH', 'ELECTRICAL_TILT', 'MECHANICAL_TILT',
]

TARGET_COL = 'ROOT_CAUSE_LABEL'


# ─────────────────────────────────────────────────────────────
# PREPARE TRAINING DATA
# ─────────────────────────────────────────────────────────────
def prepare_training_data(master_df: pd.DataFrame,
                          alarms_df: pd.DataFrame) -> tuple:
    """
    Join alarm history with NE features to create training samples.

    Each row = one alarm event with:
      - Features from the NE that generated the alarm
      - Label = what the root cause was (from ITSM resolution)

    NOTE: Once you share your ALARM table schema, we'll map exact columns.
    For now this uses a generic structure.
    """
    log.info("Preparing training data...")

    if alarms_df is None or len(alarms_df) == 0:
        log.warning("No alarm data — generating synthetic training data for demo")
        return _generate_synthetic_training_data(master_df)

    # ── Check if alarms have root cause labels
    rc_col = None
    for candidate in ['ROOT_CAUSE', 'ROOT_CAUSE_LABEL', 'CAUSE_CODE', 'FAULT_CAUSE']:
        if candidate in alarms_df.columns:
            rc_col = candidate
            break

    if rc_col is None:
        log.warning("No root cause column found in alarms — using synthetic data")
        return _generate_synthetic_training_data(master_df)

    # Rename root cause column to expected TARGET_COL
    if rc_col != TARGET_COL:
        alarms_df = alarms_df.rename(columns={rc_col: TARGET_COL})

    # ── Determine NE join key in alarms
    ne_key = None
    for candidate in ['NE_ID_FK', 'NE_ID', 'NETWORK_ELEMENT_ID_FK', 'SOURCE_NE_ID']:
        if candidate in alarms_df.columns:
            ne_key = candidate
            break

    if ne_key is None:
        log.warning("Cannot find NE ID column in alarms — using synthetic data")
        return _generate_synthetic_training_data(master_df)

    # ── Align types: alarms often have NE_ID_FK as str, master has ID (int) and NE_ID (str)
    if 'NE_ID' in master_df.columns and (alarms_df[ne_key].dtype == object or str(alarms_df[ne_key].dtype) == 'string'):
        # Join on string NE identifier so str/int mismatch is avoided
        alarms_df = alarms_df.copy()
        alarms_df[ne_key] = alarms_df[ne_key].astype(str)
        master_df = master_df.copy()
        master_df['NE_ID'] = master_df['NE_ID'].astype(str)
        training = alarms_df.merge(
            master_df,
            left_on=ne_key,
            right_on='NE_ID',
            how='inner'
        )
    else:
        # Join on integer ID: coerce alarm NE key to numeric
        alarm_ne = pd.to_numeric(alarms_df[ne_key], errors='coerce')
        alarms_df = alarms_df.copy()
        alarms_df[ne_key] = alarm_ne
        alarms_df = alarms_df.dropna(subset=[ne_key])
        training = alarms_df.merge(
            master_df,
            left_on=ne_key,
            right_on='ID',
            how='inner'
        )
    log.info(f"Training samples after join: {len(training):,}")

    training = training.dropna(subset=[TARGET_COL])
    log.info(f"Training samples with labels: {len(training):,}")

    if len(training) < 100:
        log.warning(f"Only {len(training)} labelled samples — adding synthetic data to reach 50k")
        synthetic = _generate_synthetic_training_data(master_df)
        training = pd.concat([training, synthetic], ignore_index=True)

    return training


def _generate_synthetic_training_data(master_df: pd.DataFrame,
                                       n_samples: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic training data from NE features.
    USE THIS ONLY until you connect your real alarm history.

    All labels are drawn from the canonical label set in root_cause_labels.json.
    NE feature signals (link health, age, technology, topology) are used to
    compute per-row sampling weights — no label is ever produced that does not
    exist in the trained model's output classes.
    """
    from config import load_root_cause_labels, MODELS_DIR

    # ── Load the label set that was saved by the last real training run.
    # If the file does not exist yet this is the very first run, so we load
    # the labels that train_root_cause_classifier will derive from this same
    # data; we write the file right after we finish here.
    import json
    labels_path = os.path.join(MODELS_DIR, "root_cause_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as _f:
            label_list = json.load(_f)
    else:
        # No saved label set yet — collect distinct labels from real alarm
        # data if available, otherwise this will be overwritten after training.
        label_list = None

    # Canonical label-to-index lookup for the 9 agreed buckets.
    # These strings are not "recommendations" — they are class names that the
    # XGBoost model outputs.  They must match whatever root_cause_labels.json
    # contains; we only use this as a fallback when that file does not exist.
    _DEFAULT_BUCKETS = [
        "BACKHAUL_ISSUE", "CONFIGURATION_ERROR", "HARDWARE_FAILURE",
        "INTERFACE_ERROR", "LATENCY_HIGH", "LINK_CONGESTION",
        "PACKET_LOSS", "POWER_ISSUE", "UNKNOWN",
    ]
    if label_list is None:
        label_list = _DEFAULT_BUCKETS
        log.warning(
            "root_cause_labels.json not found — synthetic generator is using "
            "the default 9-bucket set.  Run the full pipeline with real data "
            "to replace this."
        )

    label_set = set(label_list)

    # Weight table: maps an NE feature signal (derived from real DB columns)
    # to a probability distribution over the valid labels.
    # The probabilities sum to 1.0 and only reference labels in label_set.
    # If a label bucket is not in label_set (e.g. a custom deployment removed
    # it) the weight for that label is redistributed to UNKNOWN.
    def _weights(signal_key: str) -> tuple:
        """Return (labels, probs) where all labels are in label_set."""
        _table = {
            # Link health degraded: utilisation / error-rate signal
            "link_degraded": [
                ("LINK_CONGESTION", 0.40),
                ("INTERFACE_ERROR", 0.25),
                ("PACKET_LOSS",     0.20),
                ("LATENCY_HIGH",    0.15),
            ],
            # Expired warranty + old hardware
            "hw_aged": [
                ("HARDWARE_FAILURE", 0.65),
                ("POWER_ISSUE",      0.35),
            ],
            # 5G / T5G technology signal
            "radio_5g": [
                ("BACKHAUL_ISSUE",      0.45),
                ("CONFIGURATION_ERROR", 0.30),
                ("HARDWARE_FAILURE",    0.25),
            ],
            # LTE / T4G technology signal
            "radio_lte": [
                ("BACKHAUL_ISSUE",      0.40),
                ("POWER_ISSUE",         0.25),
                ("CONFIGURATION_ERROR", 0.20),
                ("HARDWARE_FAILURE",    0.15),
            ],
            # High betweenness (core/aggregation node)
            "core_node": [
                ("CONFIGURATION_ERROR", 0.40),
                ("HARDWARE_FAILURE",    0.30),
                ("INTERFACE_ERROR",     0.30),
            ],
            # Default: general distribution across all labels
            "default": [
                ("CONFIGURATION_ERROR", 0.20),
                ("HARDWARE_FAILURE",    0.20),
                ("BACKHAUL_ISSUE",      0.15),
                ("POWER_ISSUE",         0.15),
                ("LINK_CONGESTION",     0.10),
                ("INTERFACE_ERROR",     0.10),
                ("PACKET_LOSS",         0.05),
                ("LATENCY_HIGH",        0.05),
            ],
        }
        raw = _table.get(signal_key, _table["default"])
        # Filter to only labels in the actual label_set; collapse removed
        # labels into UNKNOWN.
        filtered, unknown_extra = [], 0.0
        for lbl, prob in raw:
            if lbl in label_set:
                filtered.append((lbl, prob))
            else:
                unknown_extra += prob
        if unknown_extra > 0:
            if "UNKNOWN" in label_set:
                filtered.append(("UNKNOWN", unknown_extra))
            elif filtered:
                # Redistribute to first remaining label
                lbl0, p0 = filtered[0]
                filtered[0] = (lbl0, p0 + unknown_extra)
        if not filtered:
            filtered = [("UNKNOWN", 1.0)] if "UNKNOWN" in label_set else [(label_list[0], 1.0)]
        labels = [x[0] for x in filtered]
        probs  = np.array([x[1] for x in filtered], dtype=float)
        probs /= probs.sum()
        return labels, probs

    log.info(f"Generating {n_samples:,} synthetic training samples "
             f"(label set: {label_list})...")

    df = master_df.sample(n=n_samples, replace=True).reset_index(drop=True)

    root_causes = []
    for _, row in df.iterrows():
        tech         = str(row.get("TECHNOLOGY", ""))
        link_health  = float(row.get("LINK_HEALTH_SCORE", 0) or 0)
        age_days     = float(row.get("NE_AGE_DAYS", 0) or 0)
        warranty_exp = int(row.get("WARRANTY_EXPIRED", 0) or 0)
        betweenness  = float(row.get("BETWEENNESS", 0) or 0)

        if link_health > 5:
            signal = "link_degraded"
        elif warranty_exp and age_days > 1825:
            signal = "hw_aged"
        elif "5G" in tech or "T5G" in tech:
            signal = "radio_5g"
        elif "LTE" in tech or "T4G" in tech:
            signal = "radio_lte"
        elif betweenness > 0.1:
            signal = "core_node"
        else:
            signal = "default"

        lbls, probs = _weights(signal)
        root_causes.append(np.random.choice(lbls, p=probs))

    df[TARGET_COL] = root_causes
    log.info(f"  Root cause distribution:\n{pd.Series(root_causes).value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────
# TRAIN ROOT CAUSE CLASSIFIER
# ─────────────────────────────────────────────────────────────
def train_root_cause_classifier(training_df: pd.DataFrame) -> dict:
    """
    Train XGBoost classifier to predict root cause from NE features.
    """
    log.info("\n" + "=" * 50)
    log.info("TRAINING ROOT CAUSE CLASSIFIER")
    log.info("=" * 50)

    # ── Prepare features
    available_features = [c for c in FEATURE_COLS if c in training_df.columns]
    log.info(f"Using {len(available_features)} features")

    X = training_df[available_features].fillna(0).astype(float)
    y_raw = training_df[TARGET_COL]

    # ── Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)
    log.info(f"Classes ({n_classes}): {list(label_encoder.classes_)}")

    # ── Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ── XGBoost model
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 8,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        eval_metric       = 'mlogloss',
        early_stopping_rounds = 20,
        random_state      = 42,
        n_jobs            = -1,
        tree_method       = 'hist',
    )

    log.info("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # ── Evaluate
    y_pred = model.predict(X_test)

    # Only include labels that appear in y_test (rare classes may be absent in the split)
    labels_in_test = sorted(set(y_test))
    test_names     = [label_encoder.classes_[i] for i in labels_in_test]

    report = classification_report(
        y_test, y_pred,
        labels=labels_in_test,
        target_names=test_names,
        output_dict=True,
        zero_division=0,
    )
    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_pred,
                                    labels=labels_in_test,
                                    target_names=test_names,
                                    zero_division=0))

    accuracy = report['accuracy']
    log.info(f"\nOverall Accuracy: {accuracy:.3f}")

    # ── Feature importance
    importance = pd.DataFrame({
        'feature':    available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    log.info("\nTop 15 Most Important Features:")
    log.info(importance.head(15).to_string())

    # ── Save artifacts
    with open(os.path.join(MODELS_DIR, "root_cause_xgb.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODELS_DIR, "root_cause_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    with open(os.path.join(MODELS_DIR, "root_cause_features.json"), "w") as f:
        json.dump(available_features, f, indent=2)

    with open(os.path.join(MODELS_DIR, "root_cause_labels.json"), "w") as f:
        json.dump(list(label_encoder.classes_), f, indent=2)

    importance.to_csv(os.path.join(MODELS_DIR, "root_cause_feature_importance.csv"), index=False)

    log.info(f"\nModel saved to {MODELS_DIR}/")

    return {
        'model':          model,
        'label_encoder':  label_encoder,
        'features':       available_features,
        'accuracy':       accuracy,
        'report':         report,
        'importance':     importance,
        'n_classes':      n_classes,
    }


# ─────────────────────────────────────────────────────────────
# EXPORT TO ONNX (for Spring Boot)
# ─────────────────────────────────────────────────────────────
def export_to_onnx(model, feature_cols: list, n_classes: int):
    """
    Export trained XGBoost model to ONNX format.
    This .onnx file is what Spring Boot loads directly.
    """
    log.info("\nExporting model to ONNX...")

    n_features = len(feature_cols)

    # onnxmltools requires positional feature names (f0, f1, ...)
    # Clear named features so the booster uses positional indices internally.
    # This is safe because our inference always builds a positional float array.
    booster = model.get_booster()
    original_names = booster.feature_names
    booster.feature_names = [f"f{i}" for i in range(n_features)]

    initial_types = [('input', FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_types,
        target_opset=12
    )

    booster.feature_names = original_names  # restore

    onnx_path = f"{MODELS_DIR}/root_cause_classifier.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    log.info(f"ONNX model saved: {onnx_path}")
    log.info(f"  Input:  [{None}, {n_features}] float32")
    log.info(f"  Output: [{None}, {n_classes}] float32 (class probabilities)")
    log.info(f"  File size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    return onnx_path


# ─────────────────────────────────────────────────────────────
# INFERENCE TEST (verify ONNX works before Spring Boot)
# ─────────────────────────────────────────────────────────────
def test_onnx_inference(onnx_path: str, feature_cols: list,
                         label_encoder: LabelEncoder,
                         sample_features: np.ndarray):
    """
    Quick test: run ONNX inference in Python to confirm it works.
    Same logic will run in Spring Boot via ONNX Runtime Java.
    """
    import onnxruntime as ort

    log.info("\nTesting ONNX inference...")

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    input_data = sample_features.astype(np.float32).reshape(1, -1)
    outputs = sess.run(None, {input_name: input_data})

    # outputs[0] = predicted class index
    # outputs[1] = class probabilities dict
    predicted_class_idx = outputs[0][0]
    probabilities = outputs[1][0]  # dict: {class_label: prob}

    predicted_label = label_encoder.classes_[predicted_class_idx]
    confidence = max(probabilities.values()) if isinstance(probabilities, dict) \
                 else probabilities.max()

    log.info(f"  Predicted: {predicted_label} (confidence: {confidence:.3f})")
    log.info("  ONNX inference working correctly!")
    return predicted_label, confidence


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Loading feature data...")
    master_df = pd.read_pickle(f"{DATA_DIR}/master_features.pkl")

    # Load alarms if available
    alarm_path = f"{DATA_DIR}/alarms_raw.pkl"
    if os.path.exists(alarm_path):
        alarms_df = pd.read_pickle(alarm_path)
    else:
        alarms_df = pd.DataFrame()  # will trigger synthetic generation

    # Prepare training data
    training_df = prepare_training_data(master_df, alarms_df)

    # Train model
    results = train_root_cause_classifier(training_df)

    # Export to ONNX
    onnx_path = export_to_onnx(
        results['model'],
        results['features'],
        results['n_classes']
    )

    # Test inference
    sample = training_df[results['features']].fillna(0).iloc[0].values
    test_onnx_inference(
        onnx_path,
        results['features'],
        results['label_encoder'],
        sample
    )

    log.info("\n✅ Root Cause Classifier training complete!")
    log.info(f"   Accuracy  : {results['accuracy']:.3f}")
    log.info(f"   ONNX file : {onnx_path}")
    log.info(f"   Classes   : {list(results['label_encoder'].classes_)}")
