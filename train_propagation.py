"""
MODEL 2: Alarm Propagation Predictor
=====================================
When alarm A fires, predict:
  - Which other alarms will fire next
  - How long before they fire (timing)
  - Confidence for each prediction

Algorithm: Sequential Pattern Mining on historical alarm sequences
           + Markov chain transition probabilities
           + Timing statistics from historical data

Output: JSON rules file loaded at runtime (no ONNX needed — it's a lookup)
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
import logging
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import configure_logging, OUTPUT_DIR, MODELS_DIR

log = configure_logging("train_propagation")
DATA_DIR = OUTPUT_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# ALARM SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────
def build_alarm_sequences(alarms_df: pd.DataFrame,
                           session_gap_minutes: int = 30) -> list:
    """
    Group alarms into "sessions" — bursts of alarms that belong together.
    A new session starts when gap between alarms > session_gap_minutes.

    Each session = one fault event with its cascade.
    First alarm = trigger, rest = consequences.

    REQUIRES columns (update to match your ALARM table):
      - alarm_code    : the alarm type/code
      - ne_id         : which network element
      - timestamp     : when it fired
      - severity      : CRITICAL/MAJOR/MINOR/WARNING
      - ne_type       : from NETWORK_ELEMENT join
    """
    log.info("Building alarm sequences...")

    if alarms_df is None or len(alarms_df) == 0:
        log.warning("No alarm data — generating synthetic sequences")
        return _generate_synthetic_sequences()

    # Map to canonical column names if needed
    col_renames = {}
    for canon, candidates in [
        ("alarm_code", ["ALARM_CODE", "ALARM_TYPE", "EVENT_TYPE", "FAULT_CODE"]),
        ("ne_id",      ["NE_ID_FK", "NE_ID", "NETWORK_ELEMENT_ID_FK"]),
        ("ne_type",    ["NE_TYPE", "NETWORK_ELEMENT_TYPE"]),
        ("timestamp",  ["ALARM_TIME", "CREATION_TIME", "EVENT_TIME"]),
        ("severity",   ["SEVERITY", "PERCEIVED_SEVERITY"]),
    ]:
        if canon not in alarms_df.columns:
            for c in candidates:
                if c in alarms_df.columns:
                    col_renames[c] = canon
                    break

    if col_renames:
        alarms_df = alarms_df.rename(columns=col_renames)

    required = ["alarm_code", "timestamp"]
    missing = [c for c in required if c not in alarms_df.columns]
    if missing:
        log.warning(f"Missing columns {missing} — using synthetic sequences")
        return _generate_synthetic_sequences()

    # Sort by time
    df = alarms_df.sort_values('timestamp').copy()

    sessions = []
    current_session = []
    last_time = None

    for _, row in df.iterrows():
        curr_time = pd.to_datetime(row['timestamp'])

        if last_time is None or \
           (curr_time - last_time).total_seconds() > session_gap_minutes * 60:
            if len(current_session) > 1:
                sessions.append(current_session)
            current_session = []

        current_session.append({
            'alarm_code': str(row.get('alarm_code', 'UNKNOWN')),
            'ne_id':      str(row.get('ne_id', '')),
            'ne_type':    str(row.get('ne_type', 'UNKNOWN')),
            'timestamp':  curr_time,
            'severity':   str(row.get('severity', 'MAJOR')),
        })
        last_time = curr_time

    if len(current_session) > 1:
        sessions.append(current_session)

    log.info(f"  Built {len(sessions):,} alarm sessions from {len(alarms_df):,} alarms")
    return sessions


def _generate_synthetic_sequences() -> list:
    """Synthetic sequences representing real telecom patterns."""
    patterns = [
        # Fiber cut cascade
        ['OPTICAL_POWER_LOW', 'BGP_SESSION_DOWN', 'MPLS_LSP_FAIL',
         'BTS_BACKHAUL_LOSS', 'CELL_OUTAGE', 'VoLTE_QUALITY_DROP'],

        # Power failure cascade
        ['DC_POWER_LOW', 'UPS_SWITCH', 'NE_REBOOT',
         'BGP_SESSION_DOWN', 'TRAFFIC_LOSS'],

        # RAN interference
        ['HIGH_INTERFERENCE_DETECTED', 'CQI_DEGRADATION',
         'THROUGHPUT_DROP', 'HANDOVER_FAILURE_RATE_HIGH'],

        # Hardware failure
        ['HARDWARE_TEMP_HIGH', 'FAN_FAILURE', 'NE_OVERTEMP',
         'CARD_FAILURE', 'NE_PARTIAL_OUTAGE'],

        # BGP flap
        ['BGP_HOLD_TIMER_EXPIRE', 'BGP_SESSION_DOWN',
         'ROUTE_WITHDRAWAL', 'TRAFFIC_REROUTE', 'CONGESTION_ON_ALT_PATH'],

        # ISIS adjacency loss
        ['ISIS_HELLO_TIMEOUT', 'ISIS_ADJACENCY_DOWN',
         'ROUTE_RECALCULATION', 'TRAFFIC_IMPACT'],

        # DWDM degradation
        ['OPTICAL_OSNR_DEGRADED', 'BIT_ERROR_RATE_HIGH',
         'CHANNEL_PERFORMANCE_DEGRADED', 'PROTECTION_SWITCH'],

        # Capacity breach
        ['INTERFACE_UTILIZATION_HIGH', 'BUFFER_OVERFLOW',
         'PACKET_DROP_HIGH', 'LATENCY_INCREASED', 'SLA_BREACH_WARNING'],
    ]

    sessions = []
    base_time = pd.Timestamp.now()

    for _ in range(10000):
        pattern = patterns[np.random.randint(0, len(patterns))]
        # Add some noise — not all alarms fire every time
        n_alarms = np.random.randint(2, len(pattern) + 1)
        selected = pattern[:n_alarms]

        session = []
        t = base_time + pd.Timedelta(minutes=np.random.randint(0, 10000))
        for i, alarm_code in enumerate(selected):
            delay = pd.Timedelta(seconds=np.random.randint(10, 300) * i)
            session.append({
                'alarm_code': alarm_code,
                'ne_id':      f'NE_{np.random.randint(1, 1000)}',
                'ne_type':    np.random.choice(['DWDM', 'ROUTER', 'BTS', 'SWITCH']),
                'timestamp':  t + delay,
                'severity':   np.random.choice(['CRITICAL', 'MAJOR', 'MINOR']),
            })
        sessions.append(session)

    log.info(f"  Generated {len(sessions):,} synthetic alarm sequences")
    return sessions


# ─────────────────────────────────────────────────────────────
# LEARN PROPAGATION RULES
# ─────────────────────────────────────────────────────────────
def learn_propagation_rules(sessions: list,
                             min_support: float = 0.01,
                             min_confidence: float = 0.30) -> dict:
    """
    Learn: "If alarm A fires, alarm B follows X% of the time, after Y seconds"

    Uses:
    - Co-occurrence counting
    - Conditional probability (confidence)
    - Timing statistics (mean delay + std dev)
    """
    log.info("Learning alarm propagation rules...")

    # Count: how many sessions contain each alarm
    alarm_count       = defaultdict(int)
    # Count: how many sessions contain both A then B
    pair_count        = defaultdict(int)
    # Timing: delay from A to B in seconds
    pair_delays       = defaultdict(list)
    # Severity context
    alarm_severity    = defaultdict(list)

    total_sessions = len(sessions)

    for session in sessions:
        alarms_in_session = [a['alarm_code'] for a in session]
        times_in_session  = {a['alarm_code']: a['timestamp'] for a in session}
        unique_alarms     = list(dict.fromkeys(alarms_in_session))  # preserve order, dedupe

        for alarm in unique_alarms:
            alarm_count[alarm] += 1

        # All ordered pairs A→B within session
        for i in range(len(unique_alarms)):
            for j in range(i + 1, len(unique_alarms)):
                a = unique_alarms[i]
                b = unique_alarms[j]
                pair_count[(a, b)] += 1

                # Compute delay
                try:
                    delay = (times_in_session[b] - times_in_session[a]).total_seconds()
                    if 0 <= delay <= 3600:  # only within 1 hour
                        pair_delays[(a, b)].append(delay)
                except Exception:
                    pass

    # ── Build rules: A → B with confidence + timing
    rules = {}

    for (a, b), count in pair_count.items():
        support    = count / total_sessions
        confidence = count / alarm_count[a] if alarm_count[a] > 0 else 0

        if support < min_support or confidence < min_confidence:
            continue

        delays = pair_delays.get((a, b), [])
        avg_delay = np.mean(delays) if delays else None
        std_delay = np.std(delays)  if delays else None

        if a not in rules:
            rules[a] = []

        rules[a].append({
            'consequent':   b,
            'confidence':   round(confidence, 4),
            'support':      round(support, 4),
            'count':        count,
            'avg_delay_sec': round(avg_delay, 1) if avg_delay else None,
            'std_delay_sec': round(std_delay, 1) if std_delay else None,
            'min_delay_sec': round(min(delays), 1) if delays else None,
            'max_delay_sec': round(max(delays), 1) if delays else None,
        })

    # Sort each alarm's rules by confidence descending
    for alarm in rules:
        rules[alarm] = sorted(rules[alarm],
                               key=lambda x: x['confidence'],
                               reverse=True)

    # Stats
    total_rules = sum(len(v) for v in rules.values())
    log.info(f"  Learned {total_rules:,} propagation rules")
    log.info(f"  Covering {len(rules):,} root alarms")

    high_conf = sum(1 for v in rules.values()
                    for r in v if r['confidence'] >= 0.7)
    log.info(f"  High confidence rules (≥70%): {high_conf:,}")

    return rules


# ─────────────────────────────────────────────────────────────
# SAVE + SAMPLE OUTPUT
# ─────────────────────────────────────────────────────────────
def save_propagation_rules(rules: dict):
    path = f"{MODELS_DIR}/alarm_propagation_rules.json"
    with open(path, 'w') as f:
        json.dump(rules, f, indent=2, default=str)
    log.info(f"Rules saved to {path}")
    log.info(f"File size: {os.path.getsize(path) / 1024:.1f} KB")
    return path


def print_sample_rules(rules: dict, n: int = 3):
    print("\n" + "=" * 60)
    print("SAMPLE PROPAGATION RULES")
    print("=" * 60)

    for alarm, consequents in list(rules.items())[:n]:
        print(f"\nIF alarm fires: {alarm}")
        print(f"THEN these will likely follow:")
        for r in consequents[:5]:
            delay_str = ""
            if r['avg_delay_sec']:
                mins = int(r['avg_delay_sec'] // 60)
                secs = int(r['avg_delay_sec'] % 60)
                delay_str = f"  after ~{mins}m {secs}s"
            print(f"  → {r['consequent']:<40} "
                  f"confidence: {r['confidence']*100:.0f}%"
                  f"{delay_str}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    alarm_path = f"{DATA_DIR}/alarms_raw.pkl"
    if os.path.exists(alarm_path):
        alarms_df = pd.read_pickle(alarm_path)
    else:
        alarms_df = pd.DataFrame()

    sessions = build_alarm_sequences(alarms_df)
    rules    = learn_propagation_rules(sessions)
    path     = save_propagation_rules(rules)
    print_sample_rules(rules)

    log.info("\n✅ Alarm Propagation model training complete!")
    log.info(f"   Rules file: {path}")
    log.info("   Load this JSON in Spring Boot for instant propagation prediction")
