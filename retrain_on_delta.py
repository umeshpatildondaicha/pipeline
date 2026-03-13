"""
retrain_on_delta.py — INCREMENTAL RETRAINING
=============================================
Detects database changes since the last run and retrains only affected models.
Schedule via cron:  0 6,18 * * *  cd /path/to/pipeline && python retrain_on_delta.py

Logic:
  1. Hash NE count + link count → if changed, retrain root cause + features
  2. Count new alarms since last run → if >100, retrain propagation rules
  3. Validate new model vs current (accuracy + size check)
  4. Hot-swap ONNX to deploy/ — Spring Boot NocInferenceService picks up within 10s
"""

import os
import sys
import json
import shutil
import hashlib
import logging
from datetime import datetime

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from config import (
    configure_logging, get_engine,
    OUTPUT_DIR, MODELS_DIR, STAGING_DIR, DEPLOY_DIR, BACKUP_DIR, STATE_FILE
)

log = configure_logging("retrain_on_delta")


# ─────────────────────────────────────────────────────────────
# STATE FILE — persists what we saw last time
# ─────────────────────────────────────────────────────────────
def load_retrain_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "last_run":        None,
        "ne_count_hash":   None,
        "link_count_hash": None,
        "alarm_count":     0,
        "model_version":   0,
    }


def save_retrain_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# DELTA DETECTION
# ─────────────────────────────────────────────────────────────
def detect_changes(engine, state: dict) -> tuple:
    log.info("Detecting changes since last run...")
    changes = {
        "ne_changed":     False,
        "links_changed":  False,
        "new_alarms":     0,
        "retrain_needed": False,
    }

    try:
        from sqlalchemy import text

        # NE count + ID sum fingerprint
        row = engine.execute(text("""
            SELECT COUNT(*) as cnt, SUM(ID) as id_sum
            FROM NETWORK_ELEMENT
            WHERE IS_DELETED = 0 AND DELETED = 0
        """)).fetchone()
        ne_hash = hashlib.md5(f"{row['cnt']}{row['id_sum']}".encode()).hexdigest()
        if ne_hash != state.get("ne_count_hash"):
            changes["ne_changed"] = True
            log.info(f"  NE changes detected: {row['cnt']:,} elements")
        state["ne_count_hash"] = ne_hash

        # Total link count fingerprint
        link_total = engine.execute(text("""
            SELECT
              (SELECT COUNT(*) FROM BGP_LINK     WHERE IS_DELETED=0) +
              (SELECT COUNT(*) FROM LLDP_LINK    WHERE IS_DELETED=0) +
              (SELECT COUNT(*) FROM OSPF_LINK    WHERE IS_DELETED=0) +
              (SELECT COUNT(*) FROM ISIS_LINK    WHERE IS_DELETED=0) +
              (SELECT COUNT(*) FROM PHYSICAL_LINK) as total
        """)).fetchone()["total"]
        link_hash = hashlib.md5(str(link_total).encode()).hexdigest()
        if link_hash != state.get("link_count_hash"):
            changes["links_changed"] = True
            log.info(f"  Link changes detected: {link_total:,} links")
        state["link_count_hash"] = link_hash

        # New alarms since last run
        from alarm_config import load_alarm_schema
        schema   = load_alarm_schema()
        ts_col   = schema.get("timestamp", "ALARM_TIME")
        tbl      = schema.get("table", "ALARM")
        last_run = state.get("last_run")
        if last_run:
            row2 = engine.execute(
                text(f"SELECT COUNT(*) as cnt FROM {tbl} WHERE {ts_col} > :lr"),
                {"lr": last_run}
            ).fetchone()
            changes["new_alarms"] = int(row2["cnt"]) if row2 else 0
            log.info(f"  New alarms since last run: {changes['new_alarms']:,}")

    except Exception as e:
        log.error(f"  Change detection failed: {e}")
        log.warning("  Forcing retrain due to detection error.")
        changes["ne_changed"] = True

    changes["retrain_needed"] = (
        changes["ne_changed"] or
        changes["links_changed"] or
        changes["new_alarms"] > 100
    )
    return changes, state


# ─────────────────────────────────────────────────────────────
# INCREMENTAL RETRAIN
# ─────────────────────────────────────────────────────────────
def retrain_models(changes: dict, engine) -> list:
    import pandas as pd

    log.info("\n" + "=" * 50)
    log.info("RETRAINING MODELS")
    log.info("=" * 50)

    retrained = []

    if changes["ne_changed"] or changes["links_changed"]:
        log.info("Re-running feature engineering (NE/link changes)...")

        from data_loader import load_network_elements, load_topology_links
        ne_df    = load_network_elements(engine)
        links_df = load_topology_links(engine)

        from feature_builder import build_master_feature_set
        master_df, _ = build_master_feature_set(ne_df, links_df)

        log.info("Retraining Root Cause Classifier...")
        from data_loader import load_alarms
        alarms_df = load_alarms(engine)

        from train_root_cause import prepare_training_data, train_root_cause_classifier, export_to_onnx
        training_df = prepare_training_data(master_df, alarms_df)
        rc_results  = train_root_cause_classifier(training_df)
        export_to_onnx(rc_results["model"], rc_results["features"], rc_results["n_classes"])

        # Save accuracy for validation comparison
        _save_score("new_accuracy", rc_results["accuracy"])

        retrained.append("root_cause_classifier.onnx")
        log.info(f"  Root Cause Classifier: accuracy={rc_results['accuracy']:.3f}")

    if changes["new_alarms"] > 100:
        log.info("Retraining Propagation model (new alarms)...")

        import pandas as pd
        from data_loader import load_alarms
        alarms_df = load_alarms(engine)

        from train_propagation import build_alarm_sequences, learn_propagation_rules, save_propagation_rules
        sequences = build_alarm_sequences(alarms_df)
        rules     = learn_propagation_rules(sequences)
        save_propagation_rules(rules)
        retrained.append("alarm_propagation_rules.json")

    return retrained


# ─────────────────────────────────────────────────────────────
# VALIDATION — new model must be as good as current
# ─────────────────────────────────────────────────────────────
def validate_new_model(model_file: str) -> dict:
    staging_path = os.path.join(STAGING_DIR, model_file)
    deploy_path  = os.path.join(DEPLOY_DIR,  model_file)

    if not os.path.exists(staging_path):
        return {"valid": False, "reason": "Staging file not found"}

    if not os.path.exists(deploy_path):
        return {"valid": True, "reason": "First deployment"}

    # Size sanity: new model must be at least 50% of current size
    new_size     = os.path.getsize(staging_path)
    current_size = os.path.getsize(deploy_path)
    if new_size < current_size * 0.5:
        return {
            "valid": False,
            "reason": f"New model too small ({new_size} vs {current_size} bytes)"
        }

    # Accuracy check (if scores available)
    score_path = os.path.join(MODELS_DIR, "model_scores.json")
    if os.path.exists(score_path):
        with open(score_path) as f:
            scores = json.load(f)
        new_acc  = scores.get("new_accuracy", 0)
        curr_acc = scores.get("current_accuracy", 0)
        if new_acc < curr_acc - 0.02:   # 2% tolerance
            return {
                "valid": False,
                "reason": f"Accuracy dropped: {curr_acc:.3f} → {new_acc:.3f}"
            }

    return {"valid": True, "reason": "Passed validation"}


def _save_score(key: str, value: float):
    score_path = os.path.join(MODELS_DIR, "model_scores.json")
    scores = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            scores = json.load(f)

    if key == "new_accuracy":
        scores["current_accuracy"] = scores.get("new_accuracy", value)
        scores["new_accuracy"]     = value

    with open(score_path, "w") as f:
        json.dump(scores, f, indent=2)


# ─────────────────────────────────────────────────────────────
# HOT-SWAP — atomic model file replacement
# ─────────────────────────────────────────────────────────────
def hot_swap_model(model_file: str, version: int):
    staging_path = os.path.join(STAGING_DIR, model_file)
    deploy_path  = os.path.join(DEPLOY_DIR,  model_file)
    backup_path  = os.path.join(BACKUP_DIR,  f"v{version}_{model_file}")

    if os.path.exists(deploy_path):
        shutil.copy2(deploy_path, backup_path)

    # Atomic replace (same filesystem)
    shutil.copy2(staging_path, deploy_path + ".tmp")
    os.replace(deploy_path + ".tmp", deploy_path)
    log.info(f"  Hot-swapped: {model_file} (v{version})")

    # Update manifest so Spring Boot NocInferenceService detects the new version
    manifest = {
        "version":     version,
        "deployed_at": datetime.now().isoformat(),
        "model_file":  model_file,
    }
    manifest_path = os.path.join(DEPLOY_DIR, "manifest.json")
    # Merge with existing manifest if present
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
        existing.update(manifest)
        manifest = existing
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def run_incremental_retrain():
    log.info("\n" + "=" * 60)
    log.info(f"INCREMENTAL RETRAIN — {datetime.now()}")
    log.info("=" * 60)

    state = load_retrain_state()
    log.info(f"Last run: {state.get('last_run', 'never')}")

    engine = get_engine()

    changes, state = detect_changes(engine, state)

    if not changes["retrain_needed"]:
        log.info("No significant changes — skipping retrain")
        state["last_run"] = datetime.now().isoformat()
        save_retrain_state(state)
        return

    log.info(f"Changes: {changes}")

    retrained_files = retrain_models(changes, engine)

    version  = state.get("model_version", 0) + 1
    deployed = []

    for model_file in retrained_files:
        # Stage
        src = os.path.join(MODELS_DIR, model_file)
        dst = os.path.join(STAGING_DIR, model_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

        validation = validate_new_model(model_file)
        if validation["valid"]:
            hot_swap_model(model_file, version)
            deployed.append(model_file)
        else:
            log.warning(f"  Skipped {model_file}: {validation['reason']}")

    state["last_run"]      = datetime.now().isoformat()
    state["model_version"] = version if deployed else state.get("model_version", 0)
    save_retrain_state(state)

    log.info("\n" + "=" * 60)
    log.info("RETRAIN SUMMARY")
    log.info("=" * 60)
    log.info(f"  Retrained : {len(retrained_files)} model(s)")
    log.info(f"  Deployed  : {len(deployed)} model(s)")
    log.info(f"  Version   : v{version}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_incremental_retrain()
