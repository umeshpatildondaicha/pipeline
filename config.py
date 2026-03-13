"""
config.py — Single source of truth for all pipeline settings.

All other scripts import from here. DB credentials, paths, and
alarm column names are all configured via .env (or environment variables).

Environment variable reference (all optional unless marked REQUIRED):
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD (REQUIRED), DB_NAME
  ALARM_TABLE, ALARM_NE_ID_COL, ALARM_CODE_COL, ALARM_SEVERITY_COL,
  ALARM_TIMESTAMP_COL, ALARM_ROOT_CAUSE_COL, ALARM_STATUS_COL,
  ALARM_CLEAR_TIME_COL
  OUTPUT_DIR, MODELS_DIR, DEPLOY_DIR, CHUNK_SIZE, LOG_LEVEL
  CORRELATION_WINDOW_MINUTES, BETWEENNESS_SAMPLE_K, MAX_HIERARCHY_DEPTH
  TOPO_LAYER_THRESHOLDS (e.g. "1,5,15")
  LINK_HEALTH_UTILIZATION_WEIGHT, LINK_HEALTH_ERROR_WEIGHT,
  LINK_HEALTH_DROP_WEIGHT, LINK_HEALTH_CRITICAL_WEIGHT
  XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE,
  XGB_COLSAMPLE_BYTREE, XGB_MIN_CHILD_WEIGHT, XGB_EARLY_STOPPING_ROUNDS,
  XGB_RANDOM_STATE
  SEQ_MAX_LEN, SEQ_EMBED_DIM, SEQ_HIDDEN_SIZE, SEQ_BATCH_SIZE,
  SEQ_VALID_FRAC, SEQUENCE_EPOCHS, SEQ_MAX_TRAIN_SAMPLES
  GNN_HIDDEN_DIM, GNN_BATCH_SIZE, GNN_VAL_FRAC, GNN_RANDOM_STATE,
  GNN_EPOCHS, GNN_MAX_NODES
  ANOMALY_N_ESTIMATORS, ANOMALY_SYNTHETIC_NES, ANOMALY_SYNTHETIC_HOURS,
  ANOMALY_INJECTION_RATE
  RETRAIN_NEW_ALARM_THRESHOLD, RETRAIN_MODEL_SIZE_MIN_RATIO,
  RETRAIN_ACCURACY_TOLERANCE
  SYNTHETIC_N_SAMPLES, SYNTHETIC_MIN_SAMPLES
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
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME",     "railtel"),
}

# Warn loudly if credentials look like they were never changed from defaults.
if DB_CONFIG["user"] == "root" and DB_CONFIG["password"] in ("", "root"):
    import warnings
    warnings.warn(
        "DB credentials appear to be default/empty. "
        "Set DB_USER and DB_PASSWORD in .env before connecting to a real database.",
        UserWarning,
        stacklevel=2,
    )

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
# ALARM CORRELATION
# ─────────────────────────────────────────────────────────────
CORRELATION_WINDOW_MINUTES = int(os.getenv("CORRELATION_WINDOW_MINUTES", "30"))

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
# Number of nodes to sample when computing betweenness centrality (approx.)
BETWEENNESS_SAMPLE_K = int(os.getenv("BETWEENNESS_SAMPLE_K", "500"))

# Maximum parent-chain depth to traverse before aborting (cycle guard)
MAX_HIERARCHY_DEPTH = int(os.getenv("MAX_HIERARCHY_DEPTH", "20"))

# Degree thresholds for TOPO_LAYER: [isolated→access, access→aggregation, aggregation→core]
TOPO_LAYER_THRESHOLDS = [
    int(x) for x in os.getenv("TOPO_LAYER_THRESHOLDS", "1,5,15").split(",")
]

# Weights for the ISIS combined link-health score.
# All four components are normalised to a 0–1 scale before weighting so
# that the score itself is also in the 0–1 range.
#   utilization  : 0–100 % → divide by 100
#   error_rate   : 0–1   (fraction) → already 0–1
#   drop_rate    : 0–1   (fraction) → already 0–1
#   critical_links: count → divide by SRC_LINK_COUNT
LINK_HEALTH_UTILIZATION_WEIGHT = float(os.getenv("LINK_HEALTH_UTILIZATION_WEIGHT", "0.4"))
LINK_HEALTH_ERROR_WEIGHT       = float(os.getenv("LINK_HEALTH_ERROR_WEIGHT",       "0.3"))
LINK_HEALTH_DROP_WEIGHT        = float(os.getenv("LINK_HEALTH_DROP_WEIGHT",        "0.2"))
LINK_HEALTH_CRITICAL_WEIGHT    = float(os.getenv("LINK_HEALTH_CRITICAL_WEIGHT",    "0.1"))

# ─────────────────────────────────────────────────────────────
# XGBOOST HYPERPARAMETERS (root cause classifier)
# ─────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS         = int(os.getenv("XGB_N_ESTIMATORS",          "300"))
XGB_MAX_DEPTH            = int(os.getenv("XGB_MAX_DEPTH",              "8"))
XGB_LEARNING_RATE        = float(os.getenv("XGB_LEARNING_RATE",        "0.05"))
XGB_SUBSAMPLE            = float(os.getenv("XGB_SUBSAMPLE",            "0.8"))
XGB_COLSAMPLE_BYTREE     = float(os.getenv("XGB_COLSAMPLE_BYTREE",     "0.8"))
XGB_MIN_CHILD_WEIGHT     = int(os.getenv("XGB_MIN_CHILD_WEIGHT",       "5"))
XGB_EARLY_STOPPING_ROUNDS= int(os.getenv("XGB_EARLY_STOPPING_ROUNDS",  "20"))
XGB_RANDOM_STATE         = int(os.getenv("XGB_RANDOM_STATE",           "42"))

# ─────────────────────────────────────────────────────────────
# SEQUENCE MODEL HYPERPARAMETERS (LSTM alarm-sequence classifier)
# ─────────────────────────────────────────────────────────────
SEQ_MAX_LEN         = int(os.getenv("SEQ_MAX_LEN",          "20"))
SEQ_EMBED_DIM       = int(os.getenv("SEQ_EMBED_DIM",        "32"))
SEQ_HIDDEN_SIZE     = int(os.getenv("SEQ_HIDDEN_SIZE",      "64"))
SEQ_BATCH_SIZE      = int(os.getenv("SEQ_BATCH_SIZE",       "64"))
SEQ_VALID_FRAC      = float(os.getenv("SEQ_VALID_FRAC",     "0.15"))
SEQ_N_EPOCHS        = int(os.getenv("SEQUENCE_EPOCHS",      "3"))
# Cap training samples to limit RAM usage on constrained machines (0 = no cap)
SEQ_MAX_TRAIN_SAMPLES = int(os.getenv("SEQ_MAX_TRAIN_SAMPLES", "3000"))

# ─────────────────────────────────────────────────────────────
# GNN HYPERPARAMETERS (graph convolutional network)
# ─────────────────────────────────────────────────────────────
GNN_HIDDEN_DIM  = int(os.getenv("GNN_HIDDEN_DIM",  "64"))
GNN_BATCH_SIZE  = int(os.getenv("GNN_BATCH_SIZE",  "256"))
GNN_VAL_FRAC    = float(os.getenv("GNN_VAL_FRAC",  "0.2"))
GNN_N_EPOCHS    = int(os.getenv("GNN_EPOCHS",       "10"))
GNN_RANDOM_STATE= int(os.getenv("GNN_RANDOM_STATE", "42"))
# Cap node count for GNN adjacency matrix to limit RAM (0 = no cap)
GNN_MAX_NODES   = int(os.getenv("GNN_MAX_NODES",    "5000"))

# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTOR HYPERPARAMETERS (Isolation Forest)
# ─────────────────────────────────────────────────────────────
ANOMALY_N_ESTIMATORS    = int(os.getenv("ANOMALY_N_ESTIMATORS",    "200"))
ANOMALY_SYNTHETIC_NES   = int(os.getenv("ANOMALY_SYNTHETIC_NES",   "1000"))
ANOMALY_SYNTHETIC_HOURS = int(os.getenv("ANOMALY_SYNTHETIC_HOURS", "720"))
ANOMALY_INJECTION_RATE  = float(os.getenv("ANOMALY_INJECTION_RATE","0.05"))

# ─────────────────────────────────────────────────────────────
# INCREMENTAL RETRAINING THRESHOLDS
# ─────────────────────────────────────────────────────────────
# Retrain when at least this many new alarms have arrived since the last run
RETRAIN_NEW_ALARM_THRESHOLD  = int(os.getenv("RETRAIN_NEW_ALARM_THRESHOLD",   "100"))
# New ONNX file must be at least this fraction of the current deployed size
RETRAIN_MODEL_SIZE_MIN_RATIO = float(os.getenv("RETRAIN_MODEL_SIZE_MIN_RATIO","0.5"))
# Maximum allowed accuracy drop before refusing to deploy the new model
RETRAIN_ACCURACY_TOLERANCE   = float(os.getenv("RETRAIN_ACCURACY_TOLERANCE",  "0.02"))

# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA PARAMETERS
# ─────────────────────────────────────────────────────────────
SYNTHETIC_N_SAMPLES  = int(os.getenv("SYNTHETIC_N_SAMPLES",  "50000"))
SYNTHETIC_MIN_SAMPLES= int(os.getenv("SYNTHETIC_MIN_SAMPLES","100"))

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
