#!/usr/bin/env python3
"""
Export entity_features.json for Spring Boot backend.

Reads master_features.pkl and network_elements.pkl, joins by NE ID,
and writes deploy/entity_features.json: { "ENTITY_ID": [f1, f2, ...], ... }
so the backend can look up features by ENTITY_ID (alarm.ENTITY_ID).

Run after run_pipeline.py (so master_features.pkl and deploy/root_cause_features.json exist):

  .venv/bin/python export_entity_features.py
"""

import os
import sys
import json
import logging

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from config import OUTPUT_DIR, DEPLOY_DIR, configure_logging

log = configure_logging("export_entity_features")


def main():
    import pandas as pd

    master_path = os.path.join(OUTPUT_DIR, "master_features.pkl")
    feature_list_path = os.path.join(DEPLOY_DIR, "root_cause_features.json")

    for path in (master_path, feature_list_path):
        if not os.path.exists(path):
            log.error("Missing %s — run full pipeline first: python run_pipeline.py --mode full", path)
            return 1

    master_df = pd.read_pickle(master_path)
    with open(feature_list_path) as f:
        feature_cols = json.load(f)

    # master_features already has NE_ID (from network_elements), same as ALARM.ENTITY_ID
    ne_id_col = "NE_ID" if "NE_ID" in master_df.columns else "NE_NAME"
    if ne_id_col not in master_df.columns:
        log.error("master_features has no NE_ID or NE_NAME — cannot export entity_features")
        return 1

    entity_features = {}
    available = [c for c in feature_cols if c in master_df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(master_df.columns)
        log.warning("Some feature columns missing in master_features: %s", missing)

    for _, row in master_df.iterrows():
        entity_id = str(row[ne_id_col]).strip()
        if not entity_id:
            continue
        vec = []
        for col in feature_cols:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    vec.append(0.0)
                else:
                    try:
                        vec.append(float(val))
                    except (TypeError, ValueError):
                        vec.append(0.0)
            else:
                vec.append(0.0)
        entity_features[entity_id] = vec

    out_path = os.path.join(DEPLOY_DIR, "entity_features.json")
    with open(out_path, "w") as f:
        json.dump(entity_features, f, indent=0, separators=(",", ":"))

    log.info("Exported features for %d entities to %s", len(entity_features), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
