#!/usr/bin/env python3
"""
NOC ML — Incident Aggregator (No LLM, No Hardcoded Playbooks)
==============================================================

This utility is ONLY a post-processing tool around the ML outputs.
It does NOT:
  - Call any LLM
  - Contain human-written playbooks or recommendations

It reads the ML model output CSV (`stream_inference_results.csv`) and produces
structured, algorithmic aggregations so that another layer (e.g. FaultAI Brain)
can generate natural-language RCA if desired.

What it does (deterministic, data-driven):
  - Correlates alarms into incidents using time + topology proximity
  - Computes per-incident and global statistics:
      * root_cause distributions
      * severity distributions
      * alarm counts, NE counts
      * basic topology-based scope (affected NEs, 1–2 hop neighbours)
  - Computes propagation predictions per incident using learned
    `alarm_propagation_rules.json`

Output:
  tests/output/noc_report_<timestamp>.json

This JSON contains ONLY structured fields derived from data/ML outputs.
No free-text narratives, no recommended actions.

Run from the pipeline directory:
  .venv/bin/python tests/noc_report_generator.py
  .venv/bin/python tests/noc_report_generator.py --limit 5000 --window 30
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)

import pandas as pd

log = logging.getLogger("noc_ml_incident_aggregator")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Helpers to load pipeline artifacts
# ---------------------------------------------------------------------------

def load_artifacts():
    """Load NE / topology / propagation artifacts from pipeline outputs."""
    from config import OUTPUT_DIR, MODELS_DIR

    processed = OUTPUT_DIR

    def _pkl(name):
        path = os.path.join(processed, name)
        if not os.path.exists(path):
            return None
        return pd.read_pickle(path)

    def _json(name):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    return {
        "master": _pkl("master_features.pkl"),
        "topology": _pkl("topology_links.pkl"),
        "propagation_rules": _json("alarm_propagation_rules.json"),
    }


def build_topology_graph(topo_df):
    """
    Build adjacency list: { node_id (str): set(neighbour_ids as str) }.

    Uses only src/dst integer IDs from topology_links.pkl. This is purely
    structural and does not perform any ML/LLM.
    """
    graph = defaultdict(set)
    if topo_df is None or topo_df.empty:
        return graph
    for _, row in topo_df.iterrows():
        s = str(row["src"])
        d = str(row["dst"])
        graph[s].add(d)
        graph[d].add(s)
    return graph


def get_neighbours(graph, node_id, max_hops=2):
    """
    Deterministic BFS over topology graph.

    Returns dict {ne_id: hop_distance}, excluding the start node.
    """
    start = str(node_id)
    visited = {start: 0}
    frontier = [start]

    for _ in range(max_hops):
        next_frontier = []
        for node in frontier:
            for nb in graph.get(node, []):
                if nb not in visited:
                    visited[nb] = visited[node] + 1
                    next_frontier.append(nb)
        frontier = next_frontier
        if not frontier:
            break

    visited.pop(start, None)
    return visited


def correlate_to_incidents(df, graph, window_minutes=30):
    """
    Correlate alarms into incidents using:
      - temporal closeness within window_minutes
      - same NE or 1-hop neighbours in topology graph

    This is a deterministic algorithm; no learned or LLM behaviour.
    """
    if df.empty:
        df["incident_id"] = []
        return df

    df = df.sort_values("open_time").reset_index(drop=True)
    incident_ids = [-1] * len(df)
    next_incident = 0

    for i, row in df.iterrows():
        if incident_ids[i] != -1:
            continue

        seed_time = row["open_time"]
        seed_ne = str(row["entity_id"])
        # 1-hop neighbours
        neighbourhood = {seed_ne} | graph.get(seed_ne, set())

        incident_ids[i] = next_incident

        for j in range(i + 1, len(df)):
            if incident_ids[j] != -1:
                continue

            row2 = df.iloc[j]
            ne2 = str(row2["entity_id"])
            if ne2 not in neighbourhood:
                continue

            dt = abs((row2["open_time"] - seed_time).total_seconds())
            if dt <= window_minutes * 60:
                incident_ids[j] = next_incident

        next_incident += 1

    df = df.copy()
    df["incident_id"] = incident_ids
    return df


def summarise_incident(inc_df, topology_graph, master_df, propagation_rules):
    """
    Build a purely structured summary for one incident.

    All fields here are computed from:
      - stream_inference_results.csv
      - master_features.pkl
      - topology_links.pkl
      - alarm_propagation_rules.json

    No hardcoded domain text or LLM usage.
    """
    inc_df = inc_df.sort_values("open_time")

    t_from = inc_df["open_time"].min()
    t_to = inc_df["open_time"].max()
    duration_min = max(
        0,
        int((t_to - t_from).total_seconds() / 60),
    )

    # Root-cause distribution (within incident)
    rc_counts = (
        inc_df["predicted_root_cause"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    # Choose primary as max-count; ties broken deterministically by label
    if rc_counts:
        primary_rc = sorted(
            rc_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[0][0]
    else:
        primary_rc = "UNKNOWN"

    # Confidence statistics
    conf_series = inc_df.get("confidence")
    avg_conf = float(conf_series.mean()) if conf_series is not None else None
    max_conf = float(conf_series.max()) if conf_series is not None else None

    # Severity distribution (just counts; mapping rules belong outside ML layer)
    sev_counts = (
        inc_df["severity"]
        .astype(str)
        .str.upper()
        .value_counts()
        .to_dict()
    )

    # Affected NEs
    ne_ids = inc_df["entity_id"].astype(str).unique().tolist()
    ne_summaries = []
    if master_df is not None and not master_df.empty:
        master_idx = (
            master_df.assign(NE_ID_STR=master_df["NE_ID"].astype(str))
            .set_index("NE_ID_STR")
        )
        for nid in ne_ids:
            if nid not in master_idx.index:
                ne_summaries.append({"ne_id": nid})
                continue
            row = master_idx.loc[nid]
            # Only a minimal subset of attributes; these are raw columns
            ne_summaries.append(
                {
                    "ne_id": nid,
                    "ne_name": row.get("NE_NAME"),
                    "ne_type": row.get("NE_TYPE"),
                    "technology": row.get("TECHNOLOGY"),
                    "vendor": row.get("VENDOR"),
                    "domain": row.get("DOMAIN"),
                    "topology_layer": int(row.get("TOPO_LAYER", 0))
                    if "TOPO_LAYER" in row
                    else None,
                    "graph_degree": int(row.get("GRAPH_DEGREE", 0))
                    if "GRAPH_DEGREE" in row
                    else None,
                }
            )
    else:
        ne_summaries = [{"ne_id": nid} for nid in ne_ids]

    # Topology neighbours (2 hops) that are not already in the incident
    neighbour_map = {}
    for nid in ne_ids:
        neighbour_map.update(get_neighbours(topology_graph, nid, max_hops=2))
    at_risk_ne_ids = sorted(
        nid for nid in neighbour_map.keys() if nid not in ne_ids
    )

    # Propagation predictions: purely from learned rules
    propagation = []
    if propagation_rules:
        unique_alarm_codes = inc_df["alarm_code"].astype(str).unique().tolist()
        for code in unique_alarm_codes:
            for rule in propagation_rules.get(code, []):
                propagation.append(
                    {
                        "source_alarm_code": code,
                        "target_alarm_code": str(rule.get("consequent")),
                        "probability": float(rule.get("confidence", 0.0)),
                        "support_count": rule.get("support_count"),
                    }
                )

    incident = {
        "time_window": {
            "from": t_from.isoformat(),
            "to": t_to.isoformat(),
            "duration_minutes": duration_min,
        },
        "alarm_count": int(len(inc_df)),
        "alarms": [
            {
                "alarm_id": int(row["alarm_id"]),
                "entity_id": str(row["entity_id"]),
                "alarm_code": str(row["alarm_code"]),
                "alarm_name": str(row.get("alarm_name", "")),
                "severity": str(row["severity"]),
                "predicted_root_cause": str(row["predicted_root_cause"]),
                "confidence": float(row.get("confidence", 0.0)),
            }
            for _, row in inc_df.iterrows()
        ],
        "root_cause_distribution": rc_counts,
        "primary_root_cause": primary_rc,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "severity_distribution": sev_counts,
        "affected_ne_ids": ne_ids,
        "affected_ne_summaries": ne_summaries,
        "topology_at_risk_ne_ids": at_risk_ne_ids,
        "propagation_predictions": propagation,
    }
    return incident


def summarise_global(df, incident_summaries):
    """
    Global statistics across all analysed alarms.
    Pure counting / aggregation, no domain text.
    """
    if df.empty:
        return {
            "total_alarms": 0,
            "total_incidents": 0,
            "root_cause_distribution": {},
            "severity_distribution": {},
            "avg_confidence": None,
            "top_entities_by_alarm_count": {},
        }

    rc_dist = (
        df["predicted_root_cause"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    sev_dist = (
        df["severity"]
        .astype(str)
        .str.upper()
        .value_counts()
        .to_dict()
    )
    avg_conf = float(df["confidence"].mean()) if "confidence" in df else None
    top_entities = (
        df["entity_id"]
        .astype(str)
        .value_counts()
        .head(20)
        .to_dict()
    )

    return {
        "total_alarms": int(len(df)),
        "total_incidents": int(len(incident_summaries)),
        "root_cause_distribution": rc_dist,
        "severity_distribution": sev_dist,
        "avg_confidence": avg_conf,
        "top_entities_by_alarm_count": top_entities,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pure ML incident aggregator (no LLM / no playbooks)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to stream_inference_results.csv "
        "(default: tests/output/stream_inference_results.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max number of alarms to read from CSV (default: 10000)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Incident correlation window in minutes (default: 30)",
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=100,
        help="Max number of incidents to include in output (by alarm count)",
    )
    args = parser.parse_args()

    # Resolve CSV path
    if args.input:
        csv_path = args.input
    else:
        csv_path = os.path.join(
            PIPELINE_DIR, "tests", "output", "stream_inference_results.csv"
        )

    if not os.path.exists(csv_path):
        log.error("Input CSV not found: %s", csv_path)
        log.error(
            "Generate it first using tests/stream_inference_test.py "
            "(pure ML pipeline).",
        )
        sys.exit(1)

    log.info("Loading inference results from %s", csv_path)
    df = pd.read_csv(csv_path, nrows=args.limit)
    if "open_time" not in df.columns:
        log.error("CSV missing 'open_time' column.")
        sys.exit(1)
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])

    log.info("  Loaded %d alarm rows after timestamp cleaning", len(df))

    # Load supporting artifacts
    arts = load_artifacts()
    master_df = arts["master"]
    topo_df = arts["topology"]
    propagation_rules = arts["propagation_rules"]

    log.info(
        "Artifacts: master_features rows=%d, topology_links rows=%d, "
        "propagation_rules keys=%d",
        0 if master_df is None else len(master_df),
        0 if topo_df is None else len(topo_df),
        0 if not propagation_rules else len(propagation_rules),
    )

    topo_graph = build_topology_graph(topo_df)

    # Correlate into incidents
    log.info("Correlating alarms into incidents (window=%d min)...", args.window)
    df_inc = correlate_to_incidents(df, topo_graph, window_minutes=args.window)

    if "incident_id" not in df_inc.columns:
        log.error("Incident correlation failed: no 'incident_id' column.")
        sys.exit(1)

    # Sort incidents by alarm count and keep top N
    incident_sizes = (
        df_inc.groupby("incident_id")
        .size()
        .sort_values(ascending=False)
    )
    top_incident_ids = incident_sizes.head(args.max_incidents).index.tolist()

    incident_summaries = []
    for inc_idx, inc_id in enumerate(top_incident_ids, start=1):
        inc_df = df_inc[df_inc["incident_id"] == inc_id]
        inc_summary = summarise_incident(
            inc_df, topo_graph, master_df, propagation_rules
        )
        inc_summary["incident_id"] = f"INC-{inc_idx:04d}"
        incident_summaries.append(inc_summary)
        log.info(
            "  Incident %s  alarms=%d  primary_root_cause=%s",
            inc_summary["incident_id"],
            inc_summary["alarm_count"],
            inc_summary["primary_root_cause"],
        )

    # Global summary
    global_summary = summarise_global(df, incident_summaries)

    analysis_from = (
        df["open_time"].min().isoformat() if not df.empty else None
    )
    analysis_to = (
        df["open_time"].max().isoformat() if not df.empty else None
    )

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "analysis_period": {"from": analysis_from, "to": analysis_to},
        "global_summary": global_summary,
        "incidents": incident_summaries,
    }

    # Write JSON only (no text report / narrative)
    out_dir = os.path.join(PIPELINE_DIR, "tests", "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"noc_report_{ts}.json")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Structured ML incident report written to %s", json_path)


if __name__ == "__main__":
    main()

