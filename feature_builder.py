"""
Feature Engineering — converts your raw schema data into ML features.

KEY INSIGHT: Your schema gives us 5 types of features:
  1. NE static features    (NE_TYPE, VENDOR, TECHNOLOGY, age, warranty)
  2. Topology features     (degree, centrality, neighbor health)
  3. Link health features  (from ISIS_LINK utilization/error/drop rates)
  4. Hierarchy features    (geo level, parent-child depth, layer position)
  5. Protocol features     (BGP/OSPF/LLDP status combination)
"""

import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import configure_logging, OUTPUT_DIR, MODELS_DIR

log = configure_logging("feature_builder")
DATA_DIR = OUTPUT_DIR
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. NE STATIC FEATURES
# ─────────────────────────────────────────────────────────────
def build_ne_static_features(ne_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical NE attributes into numeric features.
    """
    log.info("Building NE static features...")
    df = ne_df.copy()

    # ── Categorical columns to encode (only those present in the dataframe)
    cat_cols = ['NE_TYPE', 'TECHNOLOGY', 'VENDOR', 'DOMAIN',
                'NE_STATUS', 'OPERATIONAL_STATE', 'CATEGORY',
                'NE_STAGE', 'ADMIN_STATE', 'NE_CATEGORY']

    encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            log.debug(f"  Column {col} not found — skipping encoding")
            df[f'{col}_ENC'] = 0
            continue
        le = LabelEncoder()
        df[f'{col}_ENC'] = le.fit_transform(
            df[col].fillna('UNKNOWN').astype(str)
        )
        encoders[col] = le

    # ── Binary flags (guard against missing columns)
    df['IS_VIRTUAL_NUM']  = df['IS_VIRTUAL'].fillna(0).astype(int)   if 'IS_VIRTUAL'      in df.columns else 0
    df['IS_SECURED_NUM']  = df['IS_SECURED'].fillna(0).astype(int)   if 'IS_SECURED'      in df.columns else 0
    df['HAS_PARENT']      = (df['PARENT_NE_ID_FK'].notna()).astype(int) if 'PARENT_NE_ID_FK' in df.columns else 0

    # ── Technology generation (maps to numeric generation)
    gen_map = {
        'GSM': 2, 'T2G': 2, 'UMTS': 3, 'T3G': 3,
        'LTE': 4, 'T4G': 4, 'FDD': 4, 'TDD': 4,
        'T5G': 5, 'T4G_T5G': 5, 'TDD10': 5, 'TDD20': 5,
        'FIBER': 0, 'Wireline': 0, 'WIFI': 0,
        'COMMON': 0, 'HARDWARE': 0
    }
    df['TECH_GENERATION'] = df['TECHNOLOGY'].map(gen_map).fillna(0)

    # ── Save encoders for use at inference time
    with open(f"{MODELS_DIR}/ne_feature_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    log.info(f"  NE static features: {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 2. TOPOLOGY / GRAPH FEATURES
# ─────────────────────────────────────────────────────────────
def build_graph_features(ne_df: pd.DataFrame,
                         links_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a NetworkX graph from your BGP/LLDP/OSPF/ISIS/Physical links.
    Extract per-NE graph features:
      - degree (how many connections)
      - betweenness centrality (how critical in the path)
      - clustering coefficient (how meshed is neighborhood)
      - is_leaf (end node with only 1 connection)
      - topology layer (core / aggregation / access)
    """
    log.info("Building graph topology features...")

    G = nx.Graph()

    # Add all NEs as nodes
    for _, row in ne_df[['ID', 'NE_TYPE', 'TECHNOLOGY']].iterrows():
        G.add_node(row['ID'],
                   ne_type=row['NE_TYPE'],
                   technology=row['TECHNOLOGY'])

    # Add all links as edges
    for _, row in links_df[['src', 'dst', 'link_type']].iterrows():
        if row['src'] != row['dst']:  # skip self-loops
            G.add_edge(int(row['src']), int(row['dst']),
                       link_type=row['link_type'])

    log.info(f"  Graph: {G.number_of_nodes():,} nodes, "
             f"{G.number_of_edges():,} edges")

    # ── Compute graph metrics
    degree_dict = dict(G.degree())

    # Betweenness centrality — expensive for huge graphs, use approximation
    n_nodes = G.number_of_nodes()
    k_sample = min(500, n_nodes)  # sample 500 nodes for approximation
    log.info(f"  Computing betweenness centrality (k={k_sample})...")
    betweenness = nx.betweenness_centrality(G, k=k_sample, normalized=True)

    # Clustering coefficient
    log.info("  Computing clustering coefficients...")
    clustering = nx.clustering(G)

    # Connected component size (isolates vs well-connected)
    components = {}
    for comp in nx.connected_components(G):
        for node in comp:
            components[node] = len(comp)

    # ── Build feature dataframe
    graph_features = pd.DataFrame({
        'ID':                   list(degree_dict.keys()),
        'GRAPH_DEGREE':         list(degree_dict.values()),
        'BETWEENNESS':          [betweenness.get(n, 0) for n in degree_dict],
        'CLUSTERING_COEF':      [clustering.get(n, 0) for n in degree_dict],
        'COMPONENT_SIZE':       [components.get(n, 1) for n in degree_dict],
    })

    graph_features['IS_LEAF']      = (graph_features['GRAPH_DEGREE'] == 1).astype(int)
    graph_features['IS_ISOLATED']  = (graph_features['GRAPH_DEGREE'] == 0).astype(int)

    # ── Topology layer inference from degree
    # Core nodes have high degree + high betweenness
    graph_features['TOPO_LAYER'] = pd.cut(
        graph_features['GRAPH_DEGREE'],
        bins=[-1, 1, 5, 15, 9999],
        labels=[0, 1, 2, 3]  # isolated, access, aggregation, core
    ).astype(int)

    log.info(f"  Graph features built for {len(graph_features):,} nodes")
    return graph_features, G


# ─────────────────────────────────────────────────────────────
# 3. ISIS LINK HEALTH FEATURES (per NE)
# ─────────────────────────────────────────────────────────────
def build_isis_health_features(links_df: pd.DataFrame) -> pd.DataFrame:
    """
    ISIS links have live utilization/error/drop rates.
    Aggregate these per NE — gives real-time health signal.
    """
    log.info("Building ISIS link health features per NE...")

    isis = links_df[links_df['link_type'] == 'isis'].copy()

    if len(isis) == 0:
        log.warning("  No ISIS links found — skipping ISIS features")
        return pd.DataFrame()

    # Severity encoding
    sev_map = {'HEALTHY': 0, 'WARNING': 1, 'CRITICAL': 2}
    for col in ['SOURCE_UTILIZATION_SEVERITY', 'TARGET_UTILIZATION_SEVERITY']:
        if col in isis.columns:
            isis[f'{col}_NUM'] = isis[col].map(sev_map).fillna(0)

    # Per source NE — aggregate outgoing link health
    src_agg = isis.groupby('src').agg(
        SRC_MAX_UTILIZATION    = ('SOURCE_UTILIZATION', 'max'),
        SRC_AVG_UTILIZATION    = ('SOURCE_UTILIZATION', 'mean'),
        SRC_MAX_ERROR_RATE     = ('SOURCE_ERROR_RATE', 'max'),
        SRC_MAX_DROP_RATE      = ('SOURCE_DROP_RATE', 'max'),
        SRC_CRITICAL_LINKS     = ('SOURCE_UTILIZATION_SEVERITY_NUM',
                                  lambda x: (x == 2).sum()),
        SRC_WARNING_LINKS      = ('SOURCE_UTILIZATION_SEVERITY_NUM',
                                  lambda x: (x == 1).sum()),
        SRC_LINK_COUNT         = ('src', 'count'),
    ).reset_index().rename(columns={'src': 'ID'})

    # Per destination NE — aggregate incoming link health
    dst_agg = isis.groupby('dst').agg(
        DST_MAX_UTILIZATION    = ('TARGET_UTILIZATION', 'max'),
        DST_AVG_UTILIZATION    = ('TARGET_UTILIZATION', 'mean'),
        DST_MAX_ERROR_RATE     = ('TARGET_ERROR_RATE', 'max'),
        DST_MAX_DROP_RATE      = ('TARGET_DROP_RATE', 'max'),
    ).reset_index().rename(columns={'dst': 'ID'})

    isis_features = src_agg.merge(dst_agg, on='ID', how='outer')
    isis_features = isis_features.fillna(0)

    # ── Combined health score (0=healthy, higher=worse)
    isis_features['LINK_HEALTH_SCORE'] = (
        isis_features['SRC_MAX_UTILIZATION'] * 10 +
        isis_features['SRC_MAX_ERROR_RATE']  * 100 +
        isis_features['SRC_MAX_DROP_RATE']   * 100 +
        isis_features['SRC_CRITICAL_LINKS']  * 5
    )

    log.info(f"  ISIS health features for {len(isis_features):,} NEs")
    return isis_features


# ─────────────────────────────────────────────────────────────
# 4. HIERARCHY DEPTH FEATURES
# ─────────────────────────────────────────────────────────────
def build_hierarchy_features(ne_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build parent-child depth from PARENT_NE_ID_FK chain.
    Root nodes (no parent) = core infrastructure.
    Deep nodes = leaf/access equipment.
    """
    log.info("Building hierarchy depth features...")

    df = ne_df[['ID', 'PARENT_NE_ID_FK',
                'GEOGRAPHY_L1_ID_FK', 'GEOGRAPHY_L2_ID_FK',
                'GEOGRAPHY_L3_ID_FK', 'GEOGRAPHY_L4_ID_FK']].copy()

    # Build parent map
    parent_map = dict(zip(df['ID'], df['PARENT_NE_ID_FK']))

    def get_depth(ne_id, depth=0, visited=None):
        if visited is None:
            visited = set()
        if ne_id in visited or depth > 20:  # prevent cycles
            return depth
        visited.add(ne_id)
        parent = parent_map.get(ne_id)
        if pd.isna(parent) or parent not in parent_map:
            return depth
        return get_depth(parent, depth + 1, visited)

    log.info("  Computing hierarchy depths (this may take a moment)...")
    df['HIERARCHY_DEPTH'] = df['ID'].apply(get_depth)

    # Is root node (no parent) — these are most critical
    df['IS_ROOT_NODE'] = df['PARENT_NE_ID_FK'].isna().astype(int)

    # Geo completeness (how many geo levels are filled)
    geo_cols = ['GEOGRAPHY_L1_ID_FK', 'GEOGRAPHY_L2_ID_FK',
                'GEOGRAPHY_L3_ID_FK', 'GEOGRAPHY_L4_ID_FK']
    df['GEO_COMPLETENESS'] = df[geo_cols].notna().sum(axis=1)

    log.info(f"  Max hierarchy depth: {df['HIERARCHY_DEPTH'].max()}")
    return df[['ID', 'HIERARCHY_DEPTH', 'IS_ROOT_NODE',
               'GEO_COMPLETENESS',
               'GEOGRAPHY_L1_ID_FK', 'GEOGRAPHY_L2_ID_FK',
               'GEOGRAPHY_L3_ID_FK', 'GEOGRAPHY_L4_ID_FK']]


# ─────────────────────────────────────────────────────────────
# 5. COMBINE ALL FEATURES
# ─────────────────────────────────────────────────────────────
def build_master_feature_set(ne_df, links_df):
    """
    Combines all feature groups into one master feature dataframe.
    This is what the ML models train on.
    """
    log.info("\n" + "=" * 50)
    log.info("Building master feature set...")
    log.info("=" * 50)

    # Build each feature group
    ne_static       = build_ne_static_features(ne_df)
    graph_feats, G  = build_graph_features(ne_df, links_df)
    isis_feats      = build_isis_health_features(links_df)
    hier_feats      = build_hierarchy_features(ne_df)

    # Merge everything on NE ID
    master = ne_static.merge(graph_feats,  on='ID', how='left')
    master = master.merge(hier_feats,      on='ID', how='left')

    if len(isis_feats) > 0:
        master = master.merge(isis_feats,  on='ID', how='left')

    # Fill missing graph features (NEs with no links)
    graph_fill_cols = ['GRAPH_DEGREE', 'BETWEENNESS', 'CLUSTERING_COEF',
                       'COMPONENT_SIZE', 'IS_LEAF', 'IS_ISOLATED', 'TOPO_LAYER']
    master[graph_fill_cols] = master[graph_fill_cols].fillna(0)

    log.info(f"\nMaster feature set: {len(master):,} NEs × {len(master.columns)} features")

    # Save
    master.to_pickle(f"{DATA_DIR}/master_features.pkl")
    log.info(f"Saved to {DATA_DIR}/master_features.pkl")

    return master, G


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PREVIEW
# ─────────────────────────────────────────────────────────────
def print_feature_summary(master_df):
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    feature_groups = {
        "NE Static":    [c for c in master_df.columns if '_ENC' in c or
                         c in ['NE_AGE_DAYS', 'WARRANTY_EXPIRED',
                                'WARRANTY_REMAINING_DAYS', 'TECH_GENERATION',
                                'IS_VIRTUAL_NUM', 'PROTOCOL_HEALTH_SCORE']],
        "Graph":        [c for c in master_df.columns if c in
                         ['GRAPH_DEGREE', 'BETWEENNESS', 'CLUSTERING_COEF',
                          'COMPONENT_SIZE', 'IS_LEAF', 'TOPO_LAYER']],
        "ISIS Health":  [c for c in master_df.columns if 'UTILIZATION' in c
                         or 'ERROR_RATE' in c or 'DROP_RATE' in c
                         or 'LINK_HEALTH_SCORE' in c],
        "Hierarchy":    [c for c in master_df.columns if 'HIERARCHY' in c
                         or 'GEO_' in c or 'IS_ROOT' in c],
    }

    for group, cols in feature_groups.items():
        print(f"\n  {group} ({len(cols)} features):")
        for c in cols[:8]:
            print(f"    - {c}")
        if len(cols) > 8:
            print(f"    ... and {len(cols)-8} more")


if __name__ == "__main__":
    ne_path    = os.path.join(DATA_DIR, "network_elements.pkl")
    links_path = os.path.join(DATA_DIR, "topology_links.pkl")

    if not os.path.exists(ne_path):
        log.error(f"network_elements.pkl not found at {ne_path}")
        log.error("Run `python data_loader.py` first.")
        sys.exit(1)

    ne_df    = pd.read_pickle(ne_path)
    links_df = pd.read_pickle(links_path) if os.path.exists(links_path) \
               else pd.DataFrame(columns=["src", "dst", "link_type"])

    master, G = build_master_feature_set(ne_df, links_df)
    print_feature_summary(master)
