"""
config.py — Single source of truth for all pipeline settings.

All other scripts import from here. DB credentials, paths, and
alarm column names are all configured via .env (or environment variables).
"""

import os
import logging
from dotenv import load_dotenv

# Load .env from the same directory as this file
_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_env_path)

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME",     "railtel"),
}

# ─────────────────────────────────────────────────────────────
# ALARM TABLE — column names in your real database
# Override via .env if your ALARM table uses different names.
# ─────────────────────────────────────────────────────────────
ALARM_COLS = {
    "table":       os.getenv("ALARM_TABLE",          "ALARM"),
    "ne_id":       os.getenv("ALARM_NE_ID_COL",      "NE_ID_FK"),
    "alarm_code":  os.getenv("ALARM_CODE_COL",        "ALARM_CODE"),
    "severity":    os.getenv("ALARM_SEVERITY_COL",    "SEVERITY"),
    "timestamp":   os.getenv("ALARM_TIMESTAMP_COL",   "ALARM_TIME"),
    "root_cause":  os.getenv("ALARM_ROOT_CAUSE_COL",  "ROOT_CAUSE"),
    "status":      os.getenv("ALARM_STATUS_COL",      "STATUS"),
    "clear_time":  os.getenv("ALARM_CLEAR_TIME_COL",  "CLEAR_TIME"),
}

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PIPELINE_DIR, os.getenv("OUTPUT_DIR", "data/processed"))
MODELS_DIR   = os.path.join(PIPELINE_DIR, os.getenv("MODELS_DIR", "data/models"))
STAGING_DIR  = os.path.join(MODELS_DIR, "staging")
DEPLOY_DIR   = os.path.join(PIPELINE_DIR, os.getenv("DEPLOY_DIR", "data/models/deploy"))
BACKUP_DIR   = os.path.join(MODELS_DIR, "backup")
STATE_FILE   = os.path.join(PIPELINE_DIR, "data", "retrain_state.json")

# ─────────────────────────────────────────────────────────────
# CHUNKING / PERFORMANCE
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50000"))

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─────────────────────────────────────────────────────────────
# ENSURE DIRECTORIES EXIST
# ─────────────────────────────────────────────────────────────
for _d in [OUTPUT_DIR, MODELS_DIR, STAGING_DIR, DEPLOY_DIR, BACKUP_DIR,
           os.path.join(PIPELINE_DIR, "data")]:
    os.makedirs(_d, exist_ok=True)


def load_root_cause_labels() -> list:
    """
    Load the canonical root-cause label set from root_cause_labels.json.

    This file is written by train_root_cause.py after the XGBoost classifier
    is trained on real alarm data; the labels come from the actual data, not
    from code.  Every other script that needs the label set must call this
    function instead of defining the list inline.

    Raises FileNotFoundError if the file has not been generated yet so that
    callers fail clearly rather than silently using wrong labels.
    """
    import json
    path = os.path.join(MODELS_DIR, "root_cause_labels.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"root_cause_labels.json not found at {path}. "
            "Run train_root_cause.py (or run_pipeline.py --mode full) first "
            "so the label set is derived from real data."
        )
    with open(path) as f:
        return json.load(f)


def get_engine():
    """Create SQLAlchemy engine from DB_CONFIG."""
    from sqlalchemy import create_engine
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?charset=utf8mb3"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def configure_logging(name: str = "noc_pipeline") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(PIPELINE_DIR, "pipeline.log")),
        ],
    )
    return logging.getLogger(name)


if __name__ == "__main__":
    print("=" * 55)
    print("NOC Pipeline — Active Configuration")
    print("=" * 55)
    print(f"  DB host     : {DB_CONFIG['host']}")
    print(f"  DB port     : {DB_CONFIG['port']}")
    print(f"  DB name     : {DB_CONFIG['database']}")
    print(f"  DB user     : {DB_CONFIG['user']}")
    print(f"  Alarm table : {ALARM_COLS['table']}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Models dir  : {MODELS_DIR}")
    print(f"  Deploy dir  : {DEPLOY_DIR}")
    print(f"  Chunk size  : {CHUNK_SIZE:,}")

    print("\nTesting DB connection...")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        print("  ✅ DB connection OK")
    except Exception as e:
        print(f"  ❌ DB connection FAILED: {e}")
