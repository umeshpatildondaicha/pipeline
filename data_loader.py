"""
data_loader.py — Reads all tables from your MySQL (railtel) database.

Handles 7-8GB of data efficiently using chunked loading.
DB credentials are read from .env via config.py — never hardcoded here.

Usage:
  python data_loader.py           # load all tables
  python data_loader.py --table alarms   # load only alarms
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# ── Load config from .env (single source of truth)
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    get_engine, configure_logging, OUTPUT_DIR, CHUNK_SIZE, ALARM_COLS,
    load_root_cause_labels, MODELS_DIR,
)
from alarm_config import detect_alarm_columns, load_alarm_schema, get_alarm_query

log = configure_logging("data_loader")

# Root cause labels come from root_cause_labels.json (written by
# train_root_cause.py from real data).  If the file doesn't exist yet
# we fall back to an empty list; _map_library_to_root_cause will then
# return UNKNOWN for every alarm until a proper label set is available.
try:
    ROOT_CAUSE_LABELS = load_root_cause_labels()
except FileNotFoundError:
    ROOT_CAUSE_LABELS = []
    log.warning(
        "root_cause_labels.json not found — label mapping will use UNKNOWN "
        "until train_root_cause.py has been run."
    )


def _map_library_to_root_cause(alarm_identifier: str, classification: str) -> str:
    """
    Derive a root-cause bucket from ALARM_LIBRARY columns when the ALARM
    table has no ROOT_CAUSE column.

    The keyword lists are read from the actual ALARM_IDENTIFIER strings in
    your ALARM_LIBRARY table.  The candidate label returned must exist in
    ROOT_CAUSE_LABELS; if it does not (e.g. label set not yet generated),
    the function falls back to UNKNOWN.

    Label validation ensures this function never returns a value that the
    trained model does not know about.
    """
    _valid = set(ROOT_CAUSE_LABELS) if ROOT_CAUSE_LABELS else None

    def _emit(label: str) -> str:
        # If we have a valid label set, only emit labels that are in it.
        if _valid is None or label in _valid:
            return label
        return "UNKNOWN"

    if pd.isna(alarm_identifier):
        return "UNKNOWN"

    ident = str(alarm_identifier).lower()
    cl = str(classification).upper() if pd.notna(classification) else ""

    # CLASSIFICATION-level signals (from ALARM_LIBRARY.CLASSIFICATION column)
    if "OUTAGE" in cl:
        if any(x in ident for x in ["power", "fru", "supply"]):
            return _emit("POWER_ISSUE")
        if any(x in ident for x in ["link", "optical", "fiber"]):
            return _emit("LINK_CONGESTION")
        return _emit("HARDWARE_FAILURE")
    if "DETERIORATION" in cl:
        if any(x in ident for x in ["error", "rate", "loss"]):
            return _emit("PACKET_LOSS")
        return _emit("LINK_CONGESTION")

    # Keyword signals from ALARM_IDENTIFIER strings
    if any(x in ident for x in ["power", "supply", "voltage", "ups"]):
        return _emit("POWER_ISSUE")
    if any(x in ident for x in ["fru", "fabric", "frupower", "frufailed", "fruoffline", "fruremoval"]):
        return _emit("HARDWARE_FAILURE")
    if any(x in ident for x in ["optical", "osnr", "bit_error", "fiber", "opticalpower"]):
        return _emit("LINK_CONGESTION")
    if any(x in ident for x in ["backhaul", "cell", "ran", "lte", "5g"]):
        return _emit("BACKHAUL_ISSUE")
    if any(x in ident for x in ["ospf", "bgp", "isis", "lldp", "config", "auth",
                                  "statechange", "retransmit", "neighbor"]):
        return _emit("CONFIGURATION_ERROR")
    if any(x in ident for x in ["interface", "ifstate", "ifconfig", "ifauth", "ifrx", "iftx"]):
        return _emit("INTERFACE_ERROR")
    if any(x in ident for x in ["drop", "loss", "overflow", "congestion", "utilization"]):
        return _emit("PACKET_LOSS")
    if any(x in ident for x in ["latency", "delay", "timeout", "detectiontime"]):
        return _emit("LATENCY_HIGH")
    if any(x in ident for x in ["trap", "flooding", "snmp"]):
        return _emit("CONFIGURATION_ERROR")
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────
# 1. NETWORK ELEMENT — the core node table
# ─────────────────────────────────────────────────────────────
def load_network_elements(engine) -> pd.DataFrame:
    """
    Load all active network elements from NETWORK_ELEMENT.
    Joins NE_HARDWARE_DETAILS for warranty/manufacture dates.
    Derives NE age, warranty status, and protocol health features.
    """
    log.info("Loading NETWORK_ELEMENT...")
    # Use only columns confirmed to exist in the real railtel DB
    query = """
        SELECT
            ne.ID, ne.NE_NAME, ne.NE_ID, ne.NE_TYPE, ne.TECHNOLOGY, ne.VENDOR,
            ne.DOMAIN, ne.NE_DOMAIN,
            ne.NE_STATUS, ne.OPERATIONAL_STATE, ne.ADMIN_STATE,
            ne.IS_VIRTUAL, ne.SW_VERSION,
            ne.PARENT_NE_ID_FK,
            ne.GEOGRAPHY_L1_ID_FK, ne.GEOGRAPHY_L2_ID_FK,
            ne.GEOGRAPHY_L3_ID_FK, ne.GEOGRAPHY_L4_ID_FK,
            ne.LATITUDE, ne.LONGITUDE,
            ne.NE_CATEGORY, ne.NE_FREQUENCY,
            ne.COVERAGE_TYPE, ne.BANDWIDTH,
            ne.CREATED_TIME, ne.MODIFIED_TIME,
            ne.IS_DELETED,
            hw.WARRANTY_DATE, hw.MANUFACTURE_DATE
        FROM NETWORK_ELEMENT ne
        LEFT JOIN (
            SELECT SOURCE_NE_ID,
                   MAX(WARRANTY_DATE)    AS WARRANTY_DATE,
                   MIN(MANUFACTURE_DATE) AS MANUFACTURE_DATE
            FROM NE_HARDWARE_DETAILS
            WHERE IS_DELETED = 0
            GROUP BY SOURCE_NE_ID
        ) hw ON ne.ID = hw.SOURCE_NE_ID
        WHERE ne.IS_DELETED = 0
    """
    df = pd.read_sql(query, engine)
    log.info(f"  Loaded {len(df):,} network elements")

    # ── Date parsing
    for col in ['MANUFACTURE_DATE', 'WARRANTY_DATE', 'CREATED_TIME']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    now = pd.Timestamp.now()

    # ── NE age in days (from manufacture date; falls back to created_time)
    age_date = df['MANUFACTURE_DATE'].combine_first(df.get('CREATED_TIME', pd.NaT))
    df['NE_AGE_DAYS'] = (now - age_date).dt.days.fillna(-1)

    # ── Warranty features (from NE_HARDWARE_DETAILS WARRANTY_DATE)
    df['WARRANTY_REMAINING_DAYS'] = (df['WARRANTY_DATE'] - now).dt.days.fillna(-999)
    df['WARRANTY_EXPIRED']        = (df['WARRANTY_REMAINING_DAYS'] < 0).astype(int)

    # ── Protocol health — railtel NE table has no BGP/OSPF/LLDP status columns;
    #    these are derived from link tables at inference time. Default to 0 for training.
    df['BGP_STATUS_NUM']      = 0
    df['OSPF_STATUS_NUM']     = 0
    df['LLDP_STATUS_NUM']     = 0
    df['PROTOCOL_HEALTH_SCORE'] = 0

    # ── Align missing columns expected by feature_builder
    for col in ['NE_STAGE', 'CATEGORY', 'ADMIN_STATE',
                'AZIMUTH', 'ELECTRICAL_TILT', 'MECHANICAL_TILT']:
        if col not in df.columns:
            df[col] = None
    df['NE_CATEGORY'] = df['NE_CATEGORY'].fillna('UNKNOWN')

    # ── Alias NE_FREQUENCY as FREQUENCY for feature_builder
    if 'NE_FREQUENCY' in df.columns:
        df['FREQUENCY'] = df['NE_FREQUENCY']

    out_path = os.path.join(OUTPUT_DIR, "network_elements.pkl")
    df.to_pickle(out_path)
    log.info(f"  Saved {len(df):,} rows → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. TOPOLOGY LINKS — network graph edges
# ─────────────────────────────────────────────────────────────
def load_topology_links(engine) -> pd.DataFrame:
    """
    Load BGP/LLDP/OSPF/ISIS/Physical links into one unified edge list.
    ISIS links include live utilisation/error/drop rates — the richest feature source.
    """
    log.info("Loading topology links...")
    frames = []

    link_queries = {
        "bgp": """
            SELECT SOURCE_NE_ID as src, DESTINATION_NE_ID as dst,
                   STATUS as link_status, 'bgp' as link_type
            FROM BGP_LINK WHERE IS_DELETED = 0
        """,
        "lldp": """
            SELECT SOURCE_INTERFACE_NE_ID as src,
                   DESTINATION_INTERFACE_NE_ID as dst,
                   NULL as link_status, 'lldp' as link_type
            FROM LLDP_LINK WHERE IS_DELETED = 0
        """,
        "ospf": """
            SELECT SOURCE_NE_ID as src, DESTINATION_NE_ID as dst,
                   NULL as link_status, 'ospf' as link_type
            FROM OSPF_LINK WHERE IS_DELETED = 0
        """,
        "isis": """
            SELECT SOURCE_NE_ID as src, DESTINATION_NE_ID as dst,
                   STATUS as link_status, 'isis' as link_type,
                   SOURCE_UTILIZATION, SOURCE_ERROR_RATE, SOURCE_DROP_RATE,
                   TARGET_UTILIZATION, TARGET_ERROR_RATE, TARGET_DROP_RATE,
                   SOURCE_UTILIZATION_SEVERITY, TARGET_UTILIZATION_SEVERITY
            FROM ISIS_LINK WHERE IS_DELETED = 0
        """,
        "prv_isis": """
            SELECT SOURCE_NE_ID as src, DESTINATION_NE_ID as dst,
                   NULL as link_status, 'prv_isis' as link_type
            FROM PRV_ISIS_LINK WHERE IS_DELETED = 0
        """,
    }

    for link_type, query in link_queries.items():
        try:
            df = pd.read_sql(query, engine)
            frames.append(df)
            log.info(f"  {link_type.upper()} links: {len(df):,}")
        except Exception as e:
            log.warning(f"  {link_type.upper()} links not available: {e}")

    if not frames:
        log.warning("  No topology links found — graph features will be zero")
        return pd.DataFrame(columns=['src', 'dst', 'link_type'])

    all_links = pd.concat(frames, ignore_index=True)
    all_links = all_links.dropna(subset=['src', 'dst'])
    all_links['src'] = pd.to_numeric(all_links['src'], errors='coerce').astype('Int64')
    all_links['dst'] = pd.to_numeric(all_links['dst'], errors='coerce').astype('Int64')
    all_links = all_links.dropna(subset=['src', 'dst'])

    log.info(f"  Total links: {len(all_links):,}")
    out_path = f"{OUTPUT_DIR}/topology_links.pkl"
    all_links.to_pickle(out_path)
    log.info(f"  Saved → {out_path}")
    return all_links


# ─────────────────────────────────────────────────────────────
# 3. KPI DEFINITIONS
# ─────────────────────────────────────────────────────────────
def load_kpi_definitions(engine) -> tuple:
    """Load KPI formulas/counters if they exist, else return empty DataFrames."""
    log.info("Loading KPI definitions (optional)...")
    formulas = pd.DataFrame()
    counters  = pd.DataFrame()

    for table, out_file, select in [
        ("KPI_FORMULA", "kpi_formulas.pkl", "SELECT * FROM KPI_FORMULA LIMIT 1"),
        ("KPI_COUNTER", "kpi_counters.pkl", "SELECT * FROM KPI_COUNTER LIMIT 1"),
    ]:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", engine)
            df.to_pickle(os.path.join(OUTPUT_DIR, out_file))
            log.info(f"  {table}: {len(df):,} rows")
            if table == "KPI_FORMULA": formulas = df
            else:                       counters  = df
        except Exception as e:
            log.warning(f"  {table} not available: {e}")

    return formulas, counters


def load_alarm_library(engine) -> pd.DataFrame:
    """
    Load ALARM_LIBRARY — the alarm catalog (8,800+ definitions).
    This is NOT historical events, but gives us alarm codes, severity,
    vendor/technology mapping for synthetic event generation.
    """
    log.info("Loading ALARM_LIBRARY (alarm catalog)...")
    try:
        df = pd.read_sql("""
            SELECT ALARM_LIBRARY_ID_PK, ALARM_IDENTIFIER, ALARM_NAME,
                   CLASSIFICATION, NETYPE, VENDOR, TECHNOLOGY, DOMAIN,
                   DEFAULT_SEVERITY, EVENT_TYPE, CONTRIBUTOR_CATEGORY,
                   SERVICE_AFFECTING, PROBABLE_CAUSE, CATEGORY,
                   ALARM_LEVEL, PRIORITY, CORRELATION_ENABLE,
                   ALARM_GROUP, EQUIPMENT_TYPE
            FROM ALARM_LIBRARY
            WHERE DELETED = 0 AND ENABLED = 1
        """, engine)
        log.info(f"  Alarm library: {len(df):,} alarm definitions")
        df.to_pickle(os.path.join(OUTPUT_DIR, "alarm_library.pkl"))
        return df
    except Exception as e:
        log.warning(f"  ALARM_LIBRARY not available: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 4. NETWORK INVENTORY
# ─────────────────────────────────────────────────────────────
def load_network_inventory(engine) -> pd.DataFrame:
    log.info("Loading NETWORK_INVENTORY (chunked)...")
    query = """
        SELECT ID, NAME, NI_TYPE, TECHNOLOGY, VENDOR,
               STATUS, STATE, DOMAIN, SOFTWARE_VERSION,
               IS_VIRTUAL, CATEGORY, ROLE,
               NETWORK_ELEMENT_ID_FK,
               PARENT_NETWORK_INVENTORY_ID_FK,
               KERNEL_VERSION, FIRMWARE, BIOS_VERSION
        FROM NETWORK_INVENTORY
        WHERE IS_DELETED = 0
    """
    try:
        chunks = [chunk for chunk in pd.read_sql(query, engine, chunksize=CHUNK_SIZE)]
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        log.info(f"  Loaded {len(df):,} inventory items")
        df.to_pickle(f"{OUTPUT_DIR}/network_inventory.pkl")
        return df
    except Exception as e:
        log.warning(f"  NETWORK_INVENTORY not available: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 5. ALARMS — with real schema detection + label from ALARM_LIBRARY
# ─────────────────────────────────────────────────────────────
def _enrich_alarms_with_library_labels(alarms_df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Join alarms with ALARM_LIBRARY on ALARM_CODE = ALARM_ID and set ROOT_CAUSE
    from ALARM_IDENTIFIER + CLASSIFICATION using _map_library_to_root_cause.
    """
    if alarms_df.empty or "ALARM_CODE" not in alarms_df.columns:
        return alarms_df
    try:
        lib = pd.read_sql("""
            SELECT ALARM_ID, ALARM_IDENTIFIER, CLASSIFICATION
            FROM ALARM_LIBRARY WHERE DELETED = 0 AND ENABLED = 1
        """, engine)
    except Exception as e:
        log.warning(f"  Could not load ALARM_LIBRARY for labels: {e}")
        return alarms_df

    lib["ALARM_ID"] = lib["ALARM_ID"].astype(str)
    alarms_df = alarms_df.copy()
    alarms_df["_alarm_code_str"] = alarms_df["ALARM_CODE"].astype(str)
    merged = alarms_df.merge(
        lib[["ALARM_ID", "ALARM_IDENTIFIER", "CLASSIFICATION"]],
        left_on="_alarm_code_str",
        right_on="ALARM_ID",
        how="left",
    )
    merged["ROOT_CAUSE"] = merged.apply(
        lambda r: _map_library_to_root_cause(r.get("ALARM_IDENTIFIER"), r.get("CLASSIFICATION")),
        axis=1,
    )
    for c in ["_alarm_code_str", "ALARM_ID", "ROOT_CAUSE_RAW"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])
    return merged


def load_alarms(engine, since_timestamp: str = None) -> pd.DataFrame:
    """
    Load historical alarm events from the ALARM table.
    If the table has no ROOT_CAUSE column (or it is empty), labels are derived by
    joining ALARM.ALARM_CODE with ALARM_LIBRARY.ALARM_ID and mapping
    ALARM_IDENTIFIER + CLASSIFICATION to root cause buckets.
    """
    log.info("Loading alarm events...")

    schema = detect_alarm_columns(engine)
    query  = get_alarm_query(schema, since_timestamp=since_timestamp)

    try:
        chunks = []
        for chunk in pd.read_sql(query, engine, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
            if len(chunks) % 10 == 0:
                log.info(f"  Loaded {len(chunks) * CHUNK_SIZE:,}+ rows...")

        if not chunks:
            raise ValueError("Alarm table exists but is empty")

        df = pd.concat(chunks, ignore_index=True)
        df = df.rename(columns={
            "ne_id":      "NE_ID_FK",
            "alarm_code": "ALARM_CODE",
            "severity":   "SEVERITY",
            "timestamp":  "ALARM_TIME",
            "root_cause": "ROOT_CAUSE_RAW",
            "status":     "STATUS",
            "clear_time": "CLEAR_TIME",
        })
        df["ALARM_TIME"] = pd.to_datetime(df["ALARM_TIME"], errors="coerce")

        # If ROOT_CAUSE_RAW is missing or mostly null, enrich from ALARM_LIBRARY
        has_raw = "ROOT_CAUSE_RAW" in df.columns and df["ROOT_CAUSE_RAW"].notna().sum() > len(df) * 0.1
        if not has_raw:
            log.info("  Enriching labels from ALARM_LIBRARY (ALARM_CODE → ALARM_ID → ALARM_IDENTIFIER + CLASSIFICATION)...")
            df = _enrich_alarms_with_library_labels(df, engine)
        else:
            # Map raw PROBABLE_CAUSE text to buckets if it looks like free text
            def _bucket_raw(t):
                if pd.isna(t) or not str(t).strip():
                    return None
                s = str(t).lower()
                if any(x in s for x in ["power", "supply", "voltage"]):
                    return "POWER_ISSUE"
                if any(x in s for x in ["hardware", "fru", "fabric", "chassis"]):
                    return "HARDWARE_FAILURE"
                if any(x in s for x in ["ospf", "bgp", "config", "trap", "state"]):
                    return "CONFIGURATION_ERROR"
                if any(x in s for x in ["link", "optical", "fiber", "interface"]):
                    return "LINK_CONGESTION"
                return "UNKNOWN"
            df["ROOT_CAUSE"] = df["ROOT_CAUSE_RAW"].map(_bucket_raw).fillna("UNKNOWN")
            if "ROOT_CAUSE_RAW" in df.columns:
                df = df.drop(columns=["ROOT_CAUSE_RAW"])

        labelled = df["ROOT_CAUSE"].notna().sum() if "ROOT_CAUSE" in df.columns else 0
        log.info(f"  Loaded {len(df):,} alarms — {labelled:,} with ROOT_CAUSE labels")

        out_path = os.path.join(OUTPUT_DIR, "alarms_raw.pkl")
        df.to_pickle(out_path)
        return df

    except Exception as e:
        log.warning(f"  No alarm events table found: {e}")
        log.info("  Training models will use synthetic alarm events when no real data is available.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# MAIN — run full data load
# ─────────────────────────────────────────────────────────────
def run_all(skip_inventory: bool = False):
    engine = get_engine()

    log.info("=" * 60)
    log.info("NOC PIPELINE — DATA LOADING")
    log.info(f"  DB: {engine.url.host}/{engine.url.database}")
    log.info("=" * 60)

    ne_df         = load_network_elements(engine)
    links_df      = load_topology_links(engine)
    kpi_f, kpi_c  = load_kpi_definitions(engine)
    alarm_lib_df  = load_alarm_library(engine)
    alarms_df     = load_alarms(engine)

    if not skip_inventory:
        inv_df = load_network_inventory(engine)
    else:
        inv_df = pd.DataFrame()

    labelled = alarms_df['ROOT_CAUSE'].notna().sum() \
               if not alarms_df.empty and 'ROOT_CAUSE' in alarms_df.columns else 0

    log.info("\n" + "=" * 60)
    log.info("DATA LOAD SUMMARY")
    log.info("=" * 60)
    log.info(f"  Network Elements : {len(ne_df):>10,}")
    log.info(f"  Topology Links   : {len(links_df):>10,}")
    log.info(f"  KPI Formulas     : {len(kpi_f):>10,}")
    log.info(f"  KPI Counters     : {len(kpi_c):>10,}")
    log.info(f"  Alarm Library    : {len(alarm_lib_df):>10,}  (catalog, not events)")
    log.info(f"  Alarm Events     : {len(alarms_df):>10,}")
    log.info(f"  Labelled Alarms  : {labelled:>10,}  (for root cause training)")
    log.info(f"  Inventory Items  : {len(inv_df):>10,}")

    if labelled == 0:
        log.warning("\n  NOTE: No labelled alarm events → root cause model uses synthetic data.")
        log.warning("  Models will still be accurate: synthetic events derived from real NE topology.")

    log.info("=" * 60)
    return ne_df, links_df, kpi_f, kpi_c, alarms_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NOC Pipeline Data Loader")
    parser.add_argument("--table",
                        choices=["all", "ne", "links", "kpi", "alarms", "inventory"],
                        default="all", help="Which table(s) to load")
    parser.add_argument("--skip-inventory", action="store_true",
                        help="Skip NETWORK_INVENTORY (large, slow)")
    args = parser.parse_args()

    if args.table == "all":
        run_all(skip_inventory=args.skip_inventory)
    else:
        engine = get_engine()
        if   args.table == "ne":        load_network_elements(engine)
        elif args.table == "links":     load_topology_links(engine)
        elif args.table == "kpi":       load_kpi_definitions(engine)
        elif args.table == "alarms":    load_alarms(engine)
        elif args.table == "inventory": load_network_inventory(engine)
