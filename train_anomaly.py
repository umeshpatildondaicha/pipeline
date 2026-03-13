"""
MODEL 3: KPI Anomaly Detector
==============================
Detect KPI degradation BEFORE alarms fire.
Uses Isolation Forest — unsupervised, no labels needed.

Input : rolling window of KPI values for a NE
Output: anomaly_score (0=normal, 1=anomalous) + which KPI is anomalous

Export: ONNX for Spring Boot (runs every 30 seconds per NE)
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import configure_logging, OUTPUT_DIR, MODELS_DIR

log = configure_logging("train_anomaly")
DATA_DIR = OUTPUT_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

# KPI window features — what we compute over a rolling window
# These map to your KPI_COUNTER and KPI_FORMULA tables
KPI_WINDOW_FEATURES = [
    # Statistical features over 1-hour rolling window
    'kpi_mean',           # average value
    'kpi_std',            # variability
    'kpi_min',            # minimum
    'kpi_max',            # maximum
    'kpi_trend',          # slope (is it going up/down?)
    'kpi_pct_change_1h',  # % change vs 1 hour ago
    'kpi_pct_change_24h', # % change vs same time yesterday
    'kpi_z_score',        # how many std devs from mean?
    'kpi_above_threshold',# is value above KPI_FORMULA.THRESHOLD?

    # Context features
    'hour_of_day',        # 0-23 (traffic patterns differ)
    'day_of_week',        # 0-6
    'is_peak_hour',       # 1 if busy hour

    # NE context (from NETWORK_ELEMENT join)
    'NE_TYPE_ENC',
    'TECHNOLOGY_ENC',
    'TOPO_LAYER',         # core vs access
    'GRAPH_DEGREE',
]


def generate_synthetic_kpi_data(n_nes: int = 1000,
                                  n_hours: int = 720) -> pd.DataFrame:
    """
    Generate synthetic KPI time series (720 hours = 30 days per NE).
    Includes normal traffic patterns + injected anomalies.
    """
    log.info(f"Generating KPI data: {n_nes} NEs × {n_hours} hours...")

    rows = []
    for ne_id in range(n_nes):
        ne_type_enc = np.random.randint(0, 10)
        tech_enc    = np.random.randint(0, 15)
        topo_layer  = np.random.randint(0, 4)
        degree      = np.random.randint(1, 20)

        # Generate baseline KPI (each NE has different baseline)
        baseline = np.random.uniform(20, 80)
        noise_level = np.random.uniform(2, 10)

        for h in range(n_hours):
            hour_of_day = h % 24
            day_of_week = (h // 24) % 7

            # Traffic pattern (busy hours)
            traffic_factor = 1.0
            if 8 <= hour_of_day <= 22:
                traffic_factor = 1.3
            if hour_of_day in [9, 10, 11, 19, 20, 21]:
                traffic_factor = 1.6

            # Weekend dip
            if day_of_week in [5, 6]:
                traffic_factor *= 0.8

            kpi_value = baseline * traffic_factor + np.random.normal(0, noise_level)
            is_anomaly = 0

            # Inject anomalies (~5% of time)
            if np.random.random() < 0.05:
                anomaly_type = np.random.choice(['spike', 'drop', 'drift'])
                if anomaly_type == 'spike':
                    kpi_value *= np.random.uniform(2.0, 5.0)
                elif anomaly_type == 'drop':
                    kpi_value *= np.random.uniform(0.1, 0.4)
                else:  # drift
                    kpi_value *= np.random.uniform(1.3, 1.8)
                is_anomaly = 1

            rows.append({
                'ne_id':        ne_id,
                'timestamp':    pd.Timestamp.now() - pd.Timedelta(hours=n_hours - h),
                'kpi_value':    max(0, kpi_value),
                'is_anomaly':   is_anomaly,
                'NE_TYPE_ENC':  ne_type_enc,
                'TECHNOLOGY_ENC': tech_enc,
                'TOPO_LAYER':   topo_layer,
                'GRAPH_DEGREE': degree,
            })

    df = pd.DataFrame(rows)
    log.info(f"  Generated {len(df):,} KPI measurements "
             f"({df['is_anomaly'].sum():,} anomalies = "
             f"{df['is_anomaly'].mean()*100:.1f}%)")
    return df


def compute_window_features(kpi_df: pd.DataFrame,
                             threshold: float = 80.0) -> pd.DataFrame:
    """
    For each KPI measurement, compute statistical features
    over a rolling window.
    """
    log.info("Computing window features...")

    df = kpi_df.sort_values(['ne_id', 'timestamp']).copy()

    feature_rows = []

    for ne_id, group in df.groupby('ne_id'):
        group = group.set_index('timestamp').sort_index()
        vals  = group['kpi_value']

        # Rolling stats (1-hour window = 12 × 5min samples)
        # pandas ≥2.2 uses lowercase aliases: '1h', '24h'
        roll_1h  = vals.rolling('1h', min_periods=3)
        roll_24h = vals.rolling('24h', min_periods=3)

        # Trend: linear regression slope over last 1h
        def rolling_slope(series, min_p=3):
            slopes = []
            for i in range(len(series)):
                window = series.iloc[max(0, i-12):i+1]
                if len(window) >= min_p:
                    x = np.arange(len(window))
                    slope = np.polyfit(x, window.values, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(0.0)
            return pd.Series(slopes, index=series.index)

        trend = rolling_slope(vals)

        for idx in range(len(group)):
            row = group.iloc[idx]
            ts  = group.index[idx]

            v   = vals.iloc[idx]
            m   = roll_1h.mean().iloc[idx]
            s   = roll_1h.std().iloc[idx] if not pd.isna(roll_1h.std().iloc[idx]) else 1.0
            s   = max(s, 0.001)  # avoid division by zero

            v_24h_ago = roll_24h.mean().iloc[idx]

            feature_rows.append({
                'ne_id':              ne_id,
                'timestamp':          ts,
                'kpi_mean':           m if not pd.isna(m) else v,
                'kpi_std':            s,
                'kpi_min':            roll_1h.min().iloc[idx] or v,
                'kpi_max':            roll_1h.max().iloc[idx] or v,
                'kpi_trend':          trend.iloc[idx],
                'kpi_pct_change_1h':  (v - m) / m if m != 0 else 0,
                'kpi_pct_change_24h': (v - v_24h_ago) / v_24h_ago
                                      if v_24h_ago and v_24h_ago != 0 else 0,
                'kpi_z_score':        (v - m) / s,
                'kpi_above_threshold': int(v > threshold),
                'hour_of_day':        ts.hour,
                'day_of_week':        ts.dayofweek,
                'is_peak_hour':       int(ts.hour in [9, 10, 11, 19, 20, 21]),
                'NE_TYPE_ENC':        row.get('NE_TYPE_ENC', 0),
                'TECHNOLOGY_ENC':     row.get('TECHNOLOGY_ENC', 0),
                'TOPO_LAYER':         row.get('TOPO_LAYER', 0),
                'GRAPH_DEGREE':       row.get('GRAPH_DEGREE', 0),
                'is_anomaly':         row.get('is_anomaly', 0),
            })

    features_df = pd.DataFrame(feature_rows)
    log.info(f"  Computed {len(features_df):,} feature rows")
    return features_df


def train_anomaly_detector(features_df: pd.DataFrame) -> dict:
    """
    Train Isolation Forest anomaly detector.
    Unsupervised — learns what "normal" looks like, flags deviations.
    """
    log.info("\n" + "=" * 50)
    log.info("TRAINING KPI ANOMALY DETECTOR")
    log.info("=" * 50)

    available_features = [c for c in KPI_WINDOW_FEATURES
                          if c in features_df.columns]
    log.info(f"Using features: {available_features}")

    X = features_df[available_features].fillna(0).astype(float)
    y_true = features_df['is_anomaly'].values  # for evaluation only

    # Train on normal data only (contamination = expected anomaly rate)
    contamination = min(0.1, y_true.mean() + 0.02)
    log.info(f"Contamination rate: {contamination:.3f}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', IsolationForest(
            n_estimators  = 200,
            contamination = contamination,
            max_samples   = 'auto',
            random_state  = 42,
            n_jobs        = -1,
        ))
    ])

    log.info("Training Isolation Forest...")
    pipeline.fit(X)

    # Evaluate
    preds  = pipeline.predict(X)          # -1 = anomaly, 1 = normal
    scores = pipeline.decision_function(X) # lower = more anomalous

    predicted_anomalies = (preds == -1).astype(int)

    # Simple evaluation
    if y_true.sum() > 0:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, predicted_anomalies, zero_division=0)
        recall    = recall_score(y_true, predicted_anomalies, zero_division=0)
        f1        = f1_score(y_true, predicted_anomalies, zero_division=0)
        log.info(f"\nAnomaly Detection Performance:")
        log.info(f"  Precision : {precision:.3f}")
        log.info(f"  Recall    : {recall:.3f}")
        log.info(f"  F1 Score  : {f1:.3f}")

    # Save pipeline
    with open(os.path.join(MODELS_DIR, "kpi_anomaly_detector.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    with open(os.path.join(MODELS_DIR, "kpi_anomaly_features.json"), "w") as f:
        json.dump(available_features, f)

    log.info(f"Model saved to {MODELS_DIR}/kpi_anomaly_detector.pkl")
    return {'pipeline': pipeline, 'features': available_features}


def export_anomaly_to_onnx(pipeline, feature_cols: list):
    """Export the StandardScaler + IsolationForest pipeline to ONNX."""
    log.info("Exporting anomaly detector to ONNX...")

    n_features = len(feature_cols)
    initial_types = [('input', FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types,
                                      target_opset={"": 12, "ai.onnx.ml": 3})
        path = os.path.join(MODELS_DIR, "kpi_anomaly_detector.onnx")
        with open(path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        log.info(f"ONNX saved: {path}")
        return path
    except Exception as e:
        log.warning(f"ONNX export failed ({e}), using pickle fallback")
        return os.path.join(MODELS_DIR, "kpi_anomaly_detector.pkl")


if __name__ == "__main__":
    kpi_path = os.path.join(DATA_DIR, "kpi_history.pkl")
    if os.path.exists(kpi_path):
        kpi_df = pd.read_pickle(kpi_path)
    else:
        kpi_df = generate_synthetic_kpi_data()
        kpi_df.to_pickle(kpi_path)

    features_df = compute_window_features(kpi_df)
    result = train_anomaly_detector(features_df)
    export_anomaly_to_onnx(result['pipeline'], result['features'])

    log.info("\n✅ KPI Anomaly Detector training complete!")
