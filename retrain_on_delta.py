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
    OUTPUT_DIR, MODELS_DIR, STAGING_DIR, DEPLOY_DIR, BACKUP_DIR, STATE_FILE,
    RETRAIN_NEW_ALARM_THRESHOLD, RETRAIN_MODEL_SIZE_MIN_RATIO,
    RETRAIN_ACCURACY_TOLERANCE,
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

        with engine.connect() as conn:
            # NE count + ID sum fingerprint
            row = conn.execute(text("""
                SELECT COUNT(*) as cnt, SUM(ID) as id_sum
                FROM NETWORK_ELEMENT
                WHERE IS_DELETED = 0
            """)).fetchone()
            ne_hash = hashlib.md5(f"{row[0]}{row[1]}".encode()).hexdigest()
            if ne_hash != state.get("ne_count_hash"):
                changes["ne_changed"] = True
                log.info(f"  NE changes detected: {row[0]:,} elements")
            state["ne_count_hash"] = ne_hash

            # Total link count fingerprint
            link_total = conn.execute(text("""
                SELECT
                  (SELECT COUNT(*) FROM BGP_LINK     WHERE IS_DELETED=0) +
                  (SELECT COUNT(*) FROM LLDP_LINK    WHERE IS_DELETED=0) +
                  (SELECT COUNT(*) FROM OSPF_LINK    WHERE IS_DELETED=0) +
                  (SELECT COUNT(*) FROM ISIS_LINK    WHERE IS_DELETED=0) +
                  (SELECT COUNT(*) FROM PHYSICAL_LINK) as total
            """)).scalar()
            link_hash = hashlib.md5(str(link_total).encode()).hexdigest()
            if link_hash != state.get("link_count_hash"):
                changes["links_changed"] = True
                log.info(f"  Link changes detected: {link_total:,} links")
            state["link_count_hash"] = link_hash

            # New alarms since last run — use bind parameter (no injection risk)
            from alarm_config import load_alarm_schema
            schema   = load_alarm_schema()
            ts_col   = schema.get("timestamp", "ALARM_TIME")
            tbl      = schema.get("table", "ALARM")
            last_run = state.get("last_run")
            if last_run:
                count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {tbl} WHERE {ts_col} > :lr"),
                    {"lr": last_run}
                ).scalar()
                changes["new_alarms"] = int(count) if count else 0
                log.info(f"  New alarms since last run: {changes['new_alarms']:,}")

    except Exception as e:
        log.error(f"  Change detection failed: {e}")
        log.warning("  Forcing retrain due to detection error.")
        changes["ne_changed"] = True

    changes["retrain_needed"] = (
        changes["ne_changed"] or
        changes["links_changed"] or
        changes["new_alarms"] > RETRAIN_NEW_ALARM_THRESHOLD
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

        # Save overall accuracy + per-class F1 for validation comparison
        per_class_f1 = {
            cls: rc_results["report"].get(cls, {}).get("f1-score", 0.0)
            for cls in rc_results["label_encoder"].classes_
        }
        _save_score("new_accuracy", rc_results["accuracy"], per_class_f1=per_class_f1)

        retrained.append("root_cause_classifier.onnx")
        log.info(f"  Root Cause Classifier: accuracy={rc_results['accuracy']:.3f}")

    if changes["new_alarms"] > RETRAIN_NEW_ALARM_THRESHOLD:
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
    """Validate a newly trained model before promoting it to deploy/.

    Checks:
    1. Staging file exists.
    2. File size is at least RETRAIN_MODEL_SIZE_MIN_RATIO of the deployed size
       (guards against a corrupt / truncated export).
    3. Overall accuracy has not dropped more than RETRAIN_ACCURACY_TOLERANCE.
    4. Per-class F1 scores are logged so operators can spot class-level regressions.
    """
    staging_path = os.path.join(STAGING_DIR, model_file)
    deploy_path  = os.path.join(DEPLOY_DIR,  model_file)

    if not os.path.exists(staging_path):
        return {"valid": False, "reason": "Staging file not found"}

    if not os.path.exists(deploy_path):
        return {"valid": True, "reason": "First deployment"}

    # ── Size sanity check
    new_size     = os.path.getsize(staging_path)
    current_size = os.path.getsize(deploy_path)
    if new_size < current_size * RETRAIN_MODEL_SIZE_MIN_RATIO:
        return {
            "valid": False,
            "reason": (
                f"New model too small: {new_size:,} bytes vs "
                f"{current_size:,} bytes (threshold={RETRAIN_MODEL_SIZE_MIN_RATIO:.0%})"
            ),
        }

    # ── Accuracy + per-class metric check (if scores file available)
    score_path = os.path.join(MODELS_DIR, "model_scores.json")
    if os.path.exists(score_path):
        with open(score_path) as f:
            scores = json.load(f)

        new_acc  = scores.get("new_accuracy",     0.0)
        curr_acc = scores.get("current_accuracy", 0.0)

        if curr_acc > 0 and new_acc < curr_acc - RETRAIN_ACCURACY_TOLERANCE:
            return {
                "valid": False,
                "reason": (
                    f"Overall accuracy dropped beyond tolerance: "
                    f"{curr_acc:.3f} → {new_acc:.3f} "
                    f"(tolerance={RETRAIN_ACCURACY_TOLERANCE})"
                ),
            }

        # Log per-class F1 regression if available
        new_per_class  = scores.get("new_per_class_f1",     {})
        curr_per_class = scores.get("current_per_class_f1", {})
        if new_per_class and curr_per_class:
            for cls, new_f1 in new_per_class.items():
                old_f1 = curr_per_class.get(cls, new_f1)
                if new_f1 < old_f1 - RETRAIN_ACCURACY_TOLERANCE:
                    log.warning(
                        "  Per-class F1 regression — class '%s': %.3f → %.3f",
                        cls, old_f1, new_f1,
                    )

    return {"valid": True, "reason": "Passed validation"}


def _save_score(key: str, value, per_class_f1: dict = None):
    """Persist model scores (overall + per-class) to model_scores.json."""
    score_path = os.path.join(MODELS_DIR, "model_scores.json")
    scores = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            scores = json.load(f)

    if key == "new_accuracy":
        scores["current_accuracy"]     = scores.get("new_accuracy", value)
        scores["new_accuracy"]         = value
        if per_class_f1:
            scores["current_per_class_f1"] = scores.get("new_per_class_f1", per_class_f1)
            scores["new_per_class_f1"]     = per_class_f1

    with open(score_path, "w") as f:
        json.dump(scores, f, indent=2)


# ─────────────────────────────────────────────────────────────
# HOT-SWAP — atomic model file replacement
# ─────────────────────────────────────────────────────────────
def hot_swap_model(model_file: str, version: int):
    """Atomically replace the deployed model file and update manifest.json.

    The previous file is always backed up before replacement so that
    rollback_model() can restore it if inference fails after the swap.
    """
    staging_path = os.path.join(STAGING_DIR, model_file)
    deploy_path  = os.path.join(DEPLOY_DIR,  model_file)
    backup_path  = os.path.join(BACKUP_DIR,  f"v{version}_{model_file}")

    if os.path.exists(deploy_path):
        shutil.copy2(deploy_path, backup_path)
        log.info(f"  Backed up current model to {backup_path}")

    # os.replace is atomic on POSIX (same filesystem required)
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


def rollback_model(model_file: str, version: int) -> bool:
    """Restore the backup taken before the last hot-swap.

    Returns True if the rollback succeeded, False if no backup was found.
    """
    backup_path = os.path.join(BACKUP_DIR, f"v{version}_{model_file}")
    deploy_path = os.path.join(DEPLOY_DIR, model_file)

    if not os.path.exists(backup_path):
        log.error(f"  Rollback failed: no backup found at {backup_path}")
        return False

    shutil.copy2(backup_path, deploy_path + ".tmp")
    os.replace(deploy_path + ".tmp", deploy_path)
    log.warning(f"  Rolled back {model_file} to v{version} backup")

    # Rewrite manifest to signal Spring Boot to reload
    manifest_path = os.path.join(DEPLOY_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["version"]     = int(manifest.get("version", version)) - 1
        manifest["deployed_at"] = datetime.now().isoformat()
        manifest["rolled_back"] = True
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    return True


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
        if not validation["valid"]:
            log.warning(f"  Skipped {model_file}: {validation['reason']}")
            continue

        try:
            hot_swap_model(model_file, version)
            deployed.append(model_file)
        except Exception as exc:
            log.error(f"  Hot-swap failed for {model_file}: {exc}")
            log.warning("  Attempting rollback to previous version...")
            if rollback_model(model_file, version):
                log.info(f"  Rollback successful for {model_file}")
            else:
                log.error(f"  Rollback FAILED for {model_file} — manual intervention required")

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
