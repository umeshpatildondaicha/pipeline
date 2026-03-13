"""
alarm_config.py — Auto-detects your ALARM table schema and maps columns.

Run this script once after connecting to your database to confirm which
columns the pipeline will use. Output is cached to data/alarm_schema.json.

Usage:
  python alarm_config.py           # detect + show mapping
  python alarm_config.py --test    # test without DB
"""

import json
import os
import sys
import logging
import pandas as pd

log = logging.getLogger("alarm_config")

# ─────────────────────────────────────────────────────────────
# KNOWN COLUMN ALIASES
# Covers common naming conventions seen in telecom OSS systems.
# ─────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "ne_id": [
        "NE_ID_FK", "NE_ID", "ENTITY_ID", "NETWORK_ELEMENT_ID_FK", "NETWORK_ELEMENT_ID",
        "SOURCE_NE_ID", "MANAGED_OBJECT_ID", "NODE_ID", "ELEMENT_ID",
    ],
    "alarm_code": [
        "ALARM_CODE", "ALARM_TYPE", "EVENT_TYPE", "FAULT_CODE",
        "SPECIFIC_PROBLEM", "ALARM_NAME", "ALARM_IDENTIFIER",
        "PROBABLE_CAUSE", "OID",
    ],
    "severity": [
        "SEVERITY", "PERCEIVED_SEVERITY", "ALARM_SEVERITY",
        "PRIORITY", "IMPACT_LEVEL",
    ],
    "timestamp": [
        "ALARM_TIME", "OPEN_TIME", "CREATION_TIME", "EVENT_TIME", "RAISED_AT",
        "OCCURRENCE_TIME", "FIRST_OCCURRENCE_TIME", "NOTIFICATION_TIME",
        "CREATION_DATE",
    ],
    "root_cause": [
        "ROOT_CAUSE", "CAUSE_CODE", "PROBABLE_CAUSE", "FAULT_CAUSE",
        "RESOLUTION_CAUSE", "CORRECTIVE_ACTION_CODE", "FAILURE_CAUSE",
        "CAUSE", "ROOT_CAUSE_LABEL",
    ],
    "status": [
        "STATUS", "ALARM_STATUS", "STATE", "LIFECYCLE_STATE",
        "CLEARANCE_STATUS", "ACKED",
    ],
    "clear_time": [
        "CLEAR_TIME", "CLOSURE_TIME", "CLEARED_AT", "RESOLUTION_TIME",
        "LAST_OCCURRENCE_TIME", "END_TIME", "CLOSE_TIME",
    ],
    "ne_type": [
        "NE_TYPE", "NETWORK_ELEMENT_TYPE", "MANAGED_OBJECT_CLASS",
        "ELEMENT_TYPE", "OBJECT_TYPE",
    ],
    "domain": [
        "DOMAIN", "NE_DOMAIN", "NETWORK_DOMAIN", "TECHNOLOGY_DOMAIN",
    ],
}

SCHEMA_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "alarm_schema.json"
)


def detect_alarm_columns(engine, table_name: str = None) -> dict:
    """
    Reads the actual columns of your ALARM table and maps them to
    the canonical names the pipeline uses.

    Returns a dict like:
      {
        'table':      'ALARM',
        'ne_id':      'NE_ID_FK',
        'alarm_code': 'ALARM_CODE',
        ...
        'detected':   True,
        'columns':    [...all real column names...]
      }
    """
    from config import ALARM_COLS

    if table_name is None:
        table_name = ALARM_COLS["table"]

    log.info(f"Inspecting ALARM table: {table_name} ...")

    try:
        sample = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1", engine)
        actual_cols = [c.upper() for c in sample.columns]
        log.info(f"  Found {len(actual_cols)} columns: {actual_cols}")
    except Exception as e:
        log.error(f"  Cannot read table {table_name}: {e}")
        return _fallback_mapping(table_name)

    mapping = {
        "table":   table_name,
        "detected": True,
        "columns": list(sample.columns),
    }

    # Match each required field against known aliases
    for field, aliases in COLUMN_ALIASES.items():
        found = _find_column(actual_cols, aliases)
        if found:
            # Return original (not uppercased) column name
            original = next(c for c in sample.columns if c.upper() == found)
            mapping[field] = original
            log.info(f"  ✅  {field:<15} → {original}")
        else:
            # Fall back to what the user configured in .env
            configured = ALARM_COLS.get(field)
            mapping[field] = configured
            log.warning(f"  ⚠️   {field:<15} → not found; using config value: {configured}")

    # Persist detection result
    os.makedirs(os.path.dirname(SCHEMA_CACHE_PATH), exist_ok=True)
    with open(SCHEMA_CACHE_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    log.info(f"  Schema saved to {SCHEMA_CACHE_PATH}")

    return mapping


def load_alarm_schema() -> dict:
    """
    Returns the cached schema detection result, or falls back to config.
    Call `detect_alarm_columns(engine)` at pipeline start to refresh.
    """
    if os.path.exists(SCHEMA_CACHE_PATH):
        with open(SCHEMA_CACHE_PATH) as f:
            return json.load(f)
    from config import ALARM_COLS
    return _fallback_mapping(ALARM_COLS["table"])


def get_alarm_query(schema: dict, since_timestamp: str = None) -> str:
    """
    Build the SELECT query for alarms using the detected column names.
    Fetches the columns we need for training.
    """
    t   = schema["table"]
    ne  = schema.get("ne_id",      "NE_ID_FK")
    ac  = schema.get("alarm_code", "ALARM_CODE")
    sv  = schema.get("severity",   "SEVERITY")
    ts  = schema.get("timestamp",  "ALARM_TIME")
    rc  = schema.get("root_cause", "ROOT_CAUSE")
    st  = schema.get("status",     "STATUS")
    ct  = schema.get("clear_time", "CLEAR_TIME")
    nt  = schema.get("ne_type",    None)
    dom = schema.get("domain",     None)

    select_cols = [
        f"{ne}         AS ne_id",
        f"{ac}         AS alarm_code",
        f"{sv}         AS severity",
        f"{ts}         AS timestamp",
        f"{rc}         AS root_cause",
        f"{st}         AS status",
        f"{ct}         AS clear_time",
    ]
    if nt:
        select_cols.append(f"{nt} AS ne_type")
    if dom:
        select_cols.append(f"{dom} AS domain")

    where_clauses = []
    if since_timestamp:
        where_clauses.append(f"{ts} > '{since_timestamp}'")

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    cols  = ",\n       ".join(select_cols)

    return f"""
        SELECT
               {cols}
        FROM   {t}
        {where}
        ORDER BY {ts}
    """


def _find_column(actual_cols_upper: list, aliases: list) -> str:
    for alias in aliases:
        if alias.upper() in actual_cols_upper:
            return alias.upper()
    return None


def _fallback_mapping(table_name: str) -> dict:
    """Use .env / config values when DB is not reachable."""
    from config import ALARM_COLS
    return {
        "table":    table_name,
        "detected": False,
        "ne_id":     ALARM_COLS.get("ne_id",      "NE_ID_FK"),
        "alarm_code": ALARM_COLS.get("alarm_code", "ALARM_CODE"),
        "severity":  ALARM_COLS.get("severity",    "SEVERITY"),
        "timestamp": ALARM_COLS.get("timestamp",   "ALARM_TIME"),
        "root_cause": ALARM_COLS.get("root_cause", "ROOT_CAUSE"),
        "status":    ALARM_COLS.get("status",      "STATUS"),
        "clear_time": ALARM_COLS.get("clear_time", "CLEAR_TIME"),
    }


# ─────────────────────────────────────────────────────────────
# CLI — run to inspect your ALARM table
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from config import configure_logging, get_engine, ALARM_COLS

    configure_logging("alarm_config")

    parser = argparse.ArgumentParser(description="Detect alarm table schema")
    parser.add_argument("--test", action="store_true",
                        help="Run without DB (show fallback mapping)")
    parser.add_argument("--table", default=ALARM_COLS["table"],
                        help="Alarm table name")
    args = parser.parse_args()

    print("=" * 60)
    print("ALARM TABLE SCHEMA DETECTION")
    print("=" * 60)

    if args.test:
        print("(Test mode — no DB connection)")
        schema = _fallback_mapping(args.table)
    else:
        try:
            engine = get_engine()
            schema = detect_alarm_columns(engine, args.table)
        except Exception as e:
            print(f"DB connection failed: {e}")
            print("Showing fallback mapping from .env config:")
            schema = _fallback_mapping(args.table)

    print(f"\n  Table: {schema['table']}")
    print(f"  Auto-detected: {schema.get('detected', False)}")
    print("\n  Column mapping:")
    for k, v in schema.items():
        if k not in ("table", "detected", "columns"):
            print(f"    {k:<15} → {v}")

    print("\n  Sample query:")
    print(get_alarm_query(schema))
