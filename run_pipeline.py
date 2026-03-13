#!/usr/bin/env python3
"""
run_pipeline.py — MASTER PIPELINE RUNNER
=========================================
Runs the entire NOC ML training pipeline end-to-end.

Modes:
  python run_pipeline.py --mode full          # full first-time training
  python run_pipeline.py --mode incremental   # delta retrain (run via cron)
  python run_pipeline.py --mode test          # synthetic data, no DB needed

Full mode steps:
  1. Load data from MySQL (railtel)
  2. Build ML features (NE + topology + ISIS health + hierarchy)
  3. Train Root Cause Classifier (XGBoost → ONNX)
  4. Train Alarm Propagation rules (sequential pattern mining)
  5. Train KPI Anomaly Detector (IsolationForest → ONNX)
  6. Deploy models to deploy/ folder (Spring Boot hot-swaps automatically)
"""

import sys
import os
import time
import json
import shutil
import argparse
import logging

# Ensure pipeline dir is on path so all imports work
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from config import (
    configure_logging, get_engine, OUTPUT_DIR, MODELS_DIR,
    DEPLOY_DIR, STAGING_DIR
)

log = configure_logging("run_pipeline")


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────
def run_full_pipeline(skip_inventory: bool = False):
    log.info("=" * 70)
    log.info("NOC ML PIPELINE — FULL TRAINING")
    log.info("=" * 70)
    total_start = time.time()
    results = {}

    # ── Step 1: Load data
    _section("Step 1/5: Load data from MySQL")
    t = time.time()
    from data_loader import run_all as load_all
    ne_df, links_df, kpi_f, kpi_c, alarms_df = load_all(skip_inventory=skip_inventory)
    results["data_load_sec"] = round(time.time() - t, 1)
    log.info(f"  Done in {results['data_load_sec']}s")

    # ── Step 2: Feature engineering
    _section("Step 2/5: Build ML features")
    t = time.time()
    import pandas as pd
    ne_df    = pd.read_pickle(f"{OUTPUT_DIR}/network_elements.pkl")
    links_df = pd.read_pickle(f"{OUTPUT_DIR}/topology_links.pkl")

    from feature_builder import build_master_feature_set
    master_df, G = build_master_feature_set(ne_df, links_df)
    results["feature_build_sec"] = round(time.time() - t, 1)
    log.info(f"  Done in {results['feature_build_sec']}s  "
             f"({len(master_df):,} NEs × {len(master_df.columns)} features)")

    # ── Step 3: Root Cause Classifier
    _section("Step 3/5: Train Root Cause Classifier (XGBoost → ONNX)")
    t = time.time()
    from train_root_cause import prepare_training_data, train_root_cause_classifier, export_to_onnx

    training_df = prepare_training_data(master_df, alarms_df)
    rc_results  = train_root_cause_classifier(training_df)
    onnx_path   = export_to_onnx(
        rc_results["model"],
        rc_results["features"],
        rc_results["n_classes"]
    )
    results["root_cause_accuracy"] = round(rc_results["accuracy"], 3)
    results["root_cause_sec"]      = round(time.time() - t, 1)
    log.info(f"  Done in {results['root_cause_sec']}s  accuracy={results['root_cause_accuracy']}")

    # ── Step 4: Alarm Propagation
    _section("Step 4/5: Train Alarm Propagation model")
    t = time.time()
    from train_propagation import build_alarm_sequences, learn_propagation_rules, save_propagation_rules

    sequences = build_alarm_sequences(alarms_df)
    rules     = learn_propagation_rules(sequences)
    prop_path = save_propagation_rules(rules)
    results["propagation_rules"] = sum(len(v) for v in rules.values())
    results["propagation_sec"]   = round(time.time() - t, 1)
    log.info(f"  Done in {results['propagation_sec']}s  "
             f"{results['propagation_rules']:,} rules")

    # ── Step 4b: Alarm correlation (incidents for sequence model)
    _section("Step 4b: Alarm correlation (incidents)")
    t = time.time()
    from alarm_correlator import run as run_correlator
    run_correlator()
    results["correlator_sec"] = round(time.time() - t, 1)
    log.info(f"  Done in {results['correlator_sec']}s")

    # ── Step 4c: Sequence model (LSTM on alarm sequences → ONNX)
    _section("Step 4c: Train sequence model (LSTM → ONNX)")
    t = time.time()
    from train_sequence_model import run as run_sequence_model
    run_sequence_model()
    results["sequence_model_sec"] = round(time.time() - t, 1)
    log.info(f"  Done in {results['sequence_model_sec']}s")

    # ── Step 4d: GNN (topology-aware root cause)
    _section("Step 4d: Train GNN (topology-aware root cause)")
    t = time.time()
    from train_gnn_model import run as run_gnn_model
    run_gnn_model()
    results["gnn_model_sec"] = round(time.time() - t, 1)
    log.info(f"  Done in {results['gnn_model_sec']}s")

    # ── Step 5: KPI Anomaly Detector
    _section("Step 5/5: Train KPI Anomaly Detector (IsolationForest → ONNX)")
    t = time.time()
    from train_anomaly import (
        generate_synthetic_kpi_data, compute_window_features,
        train_anomaly_detector, export_anomaly_to_onnx
    )

    kpi_path = f"{OUTPUT_DIR}/kpi_history.pkl"
    if os.path.exists(kpi_path):
        log.info("  Using existing KPI history from disk...")
        kpi_df = pd.read_pickle(kpi_path)
    else:
        log.info("  No KPI history found — generating synthetic data...")
        kpi_df = generate_synthetic_kpi_data()
        kpi_df.to_pickle(kpi_path)

    features_df   = compute_window_features(kpi_df)
    anomaly_result = train_anomaly_detector(features_df)
    anomaly_onnx  = export_anomaly_to_onnx(
        anomaly_result["pipeline"],
        anomaly_result["features"]
    )
    results["anomaly_sec"] = round(time.time() - t, 1)
    log.info(f"  Done in {results['anomaly_sec']}s")

    # ── Deploy
    _section("Deploying models to deploy/ folder")
    deploy_count = _deploy_models()

    # ── Summary
    elapsed = round(time.time() - total_start, 1)
    _print_summary(results, elapsed, deploy_count)

    return results


# ─────────────────────────────────────────────────────────────
# INCREMENTAL (delta) RETRAIN
# ─────────────────────────────────────────────────────────────
def run_incremental():
    log.info("Running incremental retrain...")
    from retrain_on_delta import run_incremental_retrain
    run_incremental_retrain()


# ─────────────────────────────────────────────────────────────
# TEST MODE — synthetic data, no DB needed
# ─────────────────────────────────────────────────────────────
def run_test():
    log.info("=" * 70)
    log.info("NOC ML PIPELINE — TEST MODE (synthetic data, no DB)")
    log.info("=" * 70)
    import pandas as pd
    import numpy as np

    # Synthetic NE data
    log.info("Generating synthetic NE data...")
    n = 5000
    ne_df = pd.DataFrame({
        "ID":                  range(n),
        "NE_NAME":             [f"NE_{i}" for i in range(n)],
        "NE_TYPE":             np.random.choice(["ROUTER", "SWITCH", "BTS", "DWDM", "eNB"], n),
        "TECHNOLOGY":          np.random.choice(["LTE", "5G", "GSM", "FIBER", "MPLS"], n),
        "VENDOR":              np.random.choice(["Ericsson", "Nokia", "Huawei", "Cisco"], n),
        "DOMAIN":              np.random.choice(["RAN", "CORE", "TRANSPORT", "POWER"], n),
        "NE_STATUS":           np.random.choice(["ACTIVE", "INACTIVE", "MAINTENANCE"], n),
        "OPERATIONAL_STATE":   np.random.choice(["ENABLED", "DISABLED"], n),
        "ADMIN_STATE":         np.random.choice(["UNLOCKED", "LOCKED"], n),
        "NE_STAGE":            np.random.choice(["LIVE", "TRIAL", "PRE_PRODUCTION"], n),
        "CATEGORY":            np.random.choice(["SWITCH", "SERVER", "VM", "RADIO"], n),
        "NE_CATEGORY":         np.random.choice(["CORE", "ACCESS", "BACKHAUL"], n),
        "IS_VIRTUAL":          np.random.choice([0, 1], n),
        "IS_SECURED":          np.random.choice([0, 1], n),
        "PARENT_NE_ID_FK":     [None if np.random.random() < 0.3 else np.random.randint(0, 100) for _ in range(n)],
        "GEOGRAPHY_L1_ID_FK":  np.random.randint(1, 30, n),
        "GEOGRAPHY_L2_ID_FK":  np.random.randint(1, 100, n),
        "GEOGRAPHY_L3_ID_FK":  np.random.randint(1, 300, n),
        "GEOGRAPHY_L4_ID_FK":  np.random.randint(1, 1000, n),
        "LATITUDE":            np.random.uniform(8.0, 37.0, n),
        "LONGITUDE":           np.random.uniform(68.0, 97.0, n),
        "IN_SERVICE_DATE":     pd.date_range("2015-01-01", periods=n, freq="1h"),
        "WARRANTY_END_DATE":   pd.date_range("2025-01-01", periods=n, freq="1h"),
        "BGP_STATUS":          np.random.choice(["UP", "DOWN", "YET_TO_START"], n),
        "OSPF_STATUS":         np.random.choice(["UP", "DOWN", "YET_TO_START"], n),
        "LLDP_STATUS":         np.random.choice(["UP", "DOWN", "YET_TO_START"], n),
        "AZIMUTH":             np.random.randint(0, 360, n),
        "ELECTRICAL_TILT":     np.random.randint(-15, 15, n),
        "MECHANICAL_TILT":     np.random.randint(-5, 5, n),
        "CREATION_TIME":       pd.date_range("2015-01-01", periods=n, freq="1h"),
        "MODIFICATION_TIME":   pd.date_range("2023-01-01", periods=n, freq="1h"),
        "SW_VERSION":          ["v1.0"] * n,
        "NE_ID":               [f"NE_{i:04d}" for i in range(n)],
    })

    # Apply same derived columns as data_loader
    now = pd.Timestamp.now()
    ne_df['NE_AGE_DAYS']             = (now - ne_df['IN_SERVICE_DATE']).dt.days.fillna(-1)
    ne_df['WARRANTY_REMAINING_DAYS'] = (ne_df['WARRANTY_END_DATE'] - now).dt.days.fillna(-999)
    ne_df['WARRANTY_EXPIRED']        = (ne_df['WARRANTY_REMAINING_DAYS'] < 0).astype(int)
    status_map = {'UP': 1, 'DOWN': 0, 'YET_TO_START': -1}
    for col in ['BGP_STATUS', 'OSPF_STATUS', 'LLDP_STATUS']:
        ne_df[f'{col}_NUM'] = ne_df[col].map(status_map).fillna(-1)
    ne_df['PROTOCOL_HEALTH_SCORE'] = (
        ne_df['BGP_STATUS_NUM'] + ne_df['OSPF_STATUS_NUM'] + ne_df['LLDP_STATUS_NUM']
    )
    ne_df.to_pickle(f"{OUTPUT_DIR}/network_elements.pkl")

    # Synthetic links
    links_df = pd.DataFrame({
        "src":       np.random.randint(0, n, n * 3),
        "dst":       np.random.randint(0, n, n * 3),
        "link_type": np.random.choice(["bgp", "isis", "ospf", "lldp"], n * 3),
        "link_status": np.random.choice(["UP", "DOWN", None], n * 3),
        "SOURCE_UTILIZATION": np.random.uniform(0, 100, n * 3),
        "SOURCE_ERROR_RATE":  np.random.uniform(0, 0.05, n * 3),
        "SOURCE_DROP_RATE":   np.random.uniform(0, 0.02, n * 3),
        "TARGET_UTILIZATION": np.random.uniform(0, 100, n * 3),
        "TARGET_ERROR_RATE":  np.random.uniform(0, 0.05, n * 3),
        "TARGET_DROP_RATE":   np.random.uniform(0, 0.02, n * 3),
        "SOURCE_UTILIZATION_SEVERITY": np.random.choice(["HEALTHY", "WARNING", "CRITICAL"], n * 3),
        "TARGET_UTILIZATION_SEVERITY": np.random.choice(["HEALTHY", "WARNING", "CRITICAL"], n * 3),
    })
    links_df.to_pickle(f"{OUTPUT_DIR}/topology_links.pkl")

    log.info(f"Synthetic data: {len(ne_df):,} NEs, {len(links_df):,} links")

    # Run the rest of the pipeline with synthetic data
    from feature_builder import build_master_feature_set
    master_df, _ = build_master_feature_set(ne_df, links_df)

    from train_root_cause import prepare_training_data, train_root_cause_classifier, export_to_onnx
    training_df = prepare_training_data(master_df, pd.DataFrame())
    rc_results  = train_root_cause_classifier(training_df)
    export_to_onnx(rc_results["model"], rc_results["features"], rc_results["n_classes"])

    from train_propagation import build_alarm_sequences, learn_propagation_rules, save_propagation_rules
    sequences = build_alarm_sequences(pd.DataFrame())
    rules = learn_propagation_rules(sequences)
    save_propagation_rules(rules)

    # Synthetic alarms for correlator + sequence model.
    # Root cause labels are loaded from root_cause_labels.json which was
    # written by train_root_cause_classifier above — no hardcoded list.
    log.info("Generating synthetic alarms for correlator and sequence model...")
    n_alarms = 8000
    from config import load_root_cause_labels
    root_causes = load_root_cause_labels()
    alarm_codes = [str(7400 + (i % 200)) for i in range(n_alarms)]
    alarms_synth = pd.DataFrame({
        "NE_ID_FK":    np.random.choice(ne_df["NE_ID"].values, n_alarms),
        "ALARM_TIME":  pd.date_range("2024-01-01", periods=n_alarms, freq="2min"),
        "ALARM_CODE":  np.random.choice(alarm_codes, n_alarms),
        "ROOT_CAUSE":  np.random.choice(root_causes, n_alarms),
    })
    alarms_synth.to_pickle(f"{OUTPUT_DIR}/alarms_raw.pkl")
    from alarm_correlator import run as run_correlator
    run_correlator()
    from train_sequence_model import run as run_sequence_model
    run_sequence_model()
    from train_gnn_model import run as run_gnn_model
    run_gnn_model()

    from train_anomaly import generate_synthetic_kpi_data, compute_window_features, train_anomaly_detector, export_anomaly_to_onnx
    kpi_df = generate_synthetic_kpi_data(n_nes=500, n_hours=168)
    feat_df = compute_window_features(kpi_df)
    anomaly = train_anomaly_detector(feat_df)
    export_anomaly_to_onnx(anomaly["pipeline"], anomaly["features"])

    deploy_count = _deploy_models()
    log.info(f"\n✅ Test pipeline complete — {deploy_count} models deployed to {DEPLOY_DIR}/")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _deploy_models() -> int:
    """Copy trained models from MODELS_DIR to DEPLOY_DIR."""
    model_files = [
        "root_cause_classifier.onnx",
        "root_cause_sequence.onnx",
        "sequence_model_vocab.json",
        "sequence_model_labels.json",
        "gnn_root_cause.pt",
        "gnn_config.json",
        "gnn_labels.json",
        "kpi_anomaly_detector.onnx",
        "alarm_propagation_rules.json",
        "root_cause_features.json",
        "root_cause_labels.json",
        "kpi_anomaly_features.json",
    ]

    deployed = 0
    for fname in model_files:
        src = os.path.join(MODELS_DIR, fname)
        dst = os.path.join(DEPLOY_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            log.info(f"  Deployed: {fname}")
            deployed += 1
        else:
            log.debug(f"  Not found (skip): {fname}")

    # Export entity_features.json for Spring Boot (lookup by ENTITY_ID)
    try:
        from export_entity_features import main as export_entity
        if export_entity() == 0:
            log.info("  entity_features.json exported for backend")
    except Exception as e:
        log.warning("  Could not export entity_features.json: %s", e)

    # Write manifest for Spring Boot version watcher
    if deployed > 0:
        import time as _time
        from datetime import datetime
        version = int(_time.time())
        manifest_models = [f for f in model_files
                          if os.path.exists(os.path.join(DEPLOY_DIR, f))]
        if os.path.exists(os.path.join(DEPLOY_DIR, "entity_features.json")):
            manifest_models.append("entity_features.json")
        manifest = {
            "version":     version,
            "deployed_at": datetime.now().isoformat(),
            "models":      manifest_models,
        }
        with open(os.path.join(DEPLOY_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"  manifest.json written (version={version})")

    return deployed


def _section(title: str):
    log.info(f"\n{'─' * 65}")
    log.info(f"  {title}")
    log.info(f"{'─' * 65}")


def _print_summary(results: dict, elapsed: float, deploy_count: int):
    log.info(f"\n{'=' * 70}")
    log.info("PIPELINE COMPLETE")
    log.info(f"{'=' * 70}")
    log.info(f"  Total time          : {elapsed:.1f}s")
    log.info(f"  Root cause accuracy : {results.get('root_cause_accuracy', 'n/a')}")
    log.info(f"  Propagation rules   : {results.get('propagation_rules', 0):,}")
    log.info(f"  Models deployed     : {deploy_count}")
    log.info(f"\n  Models in          : {DEPLOY_DIR}/")
    log.info("  Spring Boot reads from this folder automatically (manifest.json watcher)")
    log.info(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOC ML Pipeline")
    parser.add_argument("--mode",
                        choices=["full", "incremental", "test"],
                        default="full",
                        help="Pipeline execution mode")
    parser.add_argument("--skip-inventory", action="store_true",
                        help="Skip NETWORK_INVENTORY table (large, optional)")
    args = parser.parse_args()

    if   args.mode == "full":        run_full_pipeline(skip_inventory=args.skip_inventory)
    elif args.mode == "incremental": run_incremental()
    elif args.mode == "test":        run_test()
