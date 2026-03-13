"""
alarm_correlator.py — Group alarms into incidents (NOC manager view).

Correlates by:
  - Same ENTITY_ID or topologically neighbouring NEs (from topology links)
  - Within a configurable time window (default 30 minutes)
  - Optionally same GEOGRAPHY_L3_ID (region)

Output: data/processed/incidents.pkl — each row one incident with
  incident_id, start_time, end_time, ne_ids[], alarm_sequence[], alarm_count, dominant_root_cause

Used as training data for the LSTM/Transformer sequence model.
"""

import os
import sys
import logging
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np

from config import configure_logging, OUTPUT_DIR

log = configure_logging("alarm_correlator")

DATA_DIR = OUTPUT_DIR
WINDOW_MINUTES = 30


def _build_entity_neighbours(ne_df: pd.DataFrame, links_df: pd.DataFrame):
    """
    Build map: entity_id (NE_ID string) -> set of entity_ids (self + neighbours).
    Uses topology links: src/dst are numeric IDs; ne_df has ID and NE_ID.
    """
    if links_df.empty or "src" not in links_df.columns or "dst" not in links_df.columns:
        id_to_entity = ne_df.set_index("ID")["NE_ID"].dropna().astype(str).to_dict()
        return {e: {e} for e in id_to_entity.values()}

    # ID -> single canonical NE_ID (take first if multiple rows per ID)
    id_to_entity = (
        ne_df[["ID", "NE_ID"]]
        .dropna(subset=["NE_ID"])
        .drop_duplicates("ID")
        .set_index("ID")["NE_ID"]
        .astype(str)
        .to_dict()
    )
    entity_to_ids = defaultdict(set)
    for iid, eid in id_to_entity.items():
        entity_to_ids[eid].add(iid)

    # Neighbour IDs from links
    neighbour_ids = defaultdict(set)
    for _, row in links_df.iterrows():
        s, d = row.get("src"), row.get("dst")
        if pd.isna(s) or pd.isna(d):
            continue
        try:
            s, d = int(s), int(d)
            neighbour_ids[s].add(d)
            neighbour_ids[d].add(s)
        except (TypeError, ValueError):
            continue

    # entity_id -> set of entity_ids (self + neighbours)
    entity_neighbours = {}
    for entity_id, ids in entity_to_ids.items():
        related_entities = {entity_id}
        for iid in ids:
            related_entities.add(id_to_entity.get(iid, entity_id))
            for nid in neighbour_ids.get(iid, []):
                related_entities.add(id_to_entity.get(nid, entity_id))
        entity_neighbours[entity_id] = related_entities

    return entity_neighbours


def _load_alarms_with_entity():
    """Load alarms; ensure we have entity_id and timestamp columns."""
    path = os.path.join(DATA_DIR, "alarms_raw.pkl")
    if not os.path.exists(path):
        log.error(f"Missing {path}. Run data_loader.py first.")
        return pd.DataFrame()

    df = pd.read_pickle(path)
    df = df.rename(columns={"NE_ID_FK": "entity_id", "ALARM_TIME": "timestamp"})
    if "entity_id" not in df.columns:
        df["entity_id"] = df.get("ENTITY_ID", df.get("NE_ID_FK", ""))
    if "timestamp" not in df.columns:
        df["timestamp"] = df.get("OPEN_TIME", pd.NaT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "entity_id"])
    df["entity_id"] = df["entity_id"].astype(str)
    return df


def correlate_alarms(
    window_minutes: int = WINDOW_MINUTES,
    use_geography: bool = False,
) -> pd.DataFrame:
    """
    Group alarms into incidents. Returns a DataFrame with one row per incident.
    """
    alarms_df = _load_alarms_with_entity()
    if alarms_df.empty:
        log.warning("No alarms to correlate.")
        return pd.DataFrame()

    ne_path = os.path.join(DATA_DIR, "network_elements.pkl")
    links_path = os.path.join(DATA_DIR, "topology_links.pkl")
    if not os.path.exists(ne_path):
        log.warning("Missing network_elements.pkl; correlating by entity_id + time only.")
        ne_df = pd.DataFrame()
        links_df = pd.DataFrame()
    else:
        ne_df = pd.read_pickle(ne_path)
        links_df = pd.read_pickle(links_path) if os.path.exists(links_path) else pd.DataFrame()

    entity_neighbours = _build_entity_neighbours(ne_df, links_df) if not ne_df.empty else {}
    window_sec = window_minutes * 60

    # Sort by time
    alarms_df = alarms_df.sort_values("timestamp").reset_index(drop=True)

    incidents = []
    current = {
        "start_time": None,
        "end_time": None,
        "entity_ids": set(),
        "alarm_codes": [],
        "timestamps": [],
        "root_causes": [],
    }

    def flush_incident(inc_id):
        if not current["alarm_codes"]:
            return
        incidents.append({
            "incident_id": inc_id,
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "ne_ids": list(current["entity_ids"]),
            "alarm_sequence": list(current["alarm_codes"]),
            "alarm_count": len(current["alarm_codes"]),
            "dominant_root_cause": pd.Series(current["root_causes"]).mode().iloc[0]
            if current["root_causes"] else "UNKNOWN",
        })

    inc_id = 0
    for _, row in alarms_df.iterrows():
        ts = row["timestamp"]
        eid = str(row["entity_id"])
        code = str(row.get("ALARM_CODE", row.get("alarm_code", "")))
        rc = str(row.get("ROOT_CAUSE", "UNKNOWN"))

        related = entity_neighbours.get(eid, {eid})
        if use_geography and "GEOGRAPHY_L3_ID" in row and pd.notna(row.get("GEOGRAPHY_L3_ID")):
            pass  # could filter by same geography here if we had it on alarms

        if current["start_time"] is None:
            current["start_time"] = ts
            current["end_time"] = ts
            current["entity_ids"] = {eid}
            current["alarm_codes"] = [code]
            current["timestamps"] = [ts]
            current["root_causes"] = [rc]
            continue

        gap = (ts - current["end_time"]).total_seconds()
        overlap = current["entity_ids"] & related and gap <= window_sec

        if overlap:
            current["end_time"] = ts
            current["entity_ids"] |= related
            current["alarm_codes"].append(code)
            current["timestamps"].append(ts)
            current["root_causes"].append(rc)
        else:
            inc_id += 1
            flush_incident(inc_id)
            current["start_time"] = ts
            current["end_time"] = ts
            current["entity_ids"] = {eid}
            current["alarm_codes"] = [code]
            current["timestamps"] = [ts]
            current["root_causes"] = [rc]

    inc_id += 1
    flush_incident(inc_id)

    out = pd.DataFrame(incidents)
    log.info(f"Correlated {len(alarms_df):,} alarms into {len(out):,} incidents (window={window_minutes} min)")
    return out


def run():
    """Correlate and save incidents to DATA_DIR/incidents.pkl."""
    df = correlate_alarms(window_minutes=WINDOW_MINUTES)
    if df.empty:
        return df
    path = os.path.join(DATA_DIR, "incidents.pkl")
    df.to_pickle(path)
    log.info(f"Saved {path}")
    return df


if __name__ == "__main__":
    run()
