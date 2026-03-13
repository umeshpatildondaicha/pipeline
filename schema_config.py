"""
Schema configuration derived from your actual database tables.
Maps your real columns to the ML pipeline's expected features.
"""

# ─────────────────────────────────────────────
# NETWORK ELEMENT — core node features
# ─────────────────────────────────────────────
NETWORK_ELEMENT_COLS = {
    "id":               "ID",
    "ne_name":          "NE_NAME",
    "ne_type":          "NE_TYPE",
    "technology":       "TECHNOLOGY",        # FDD, GSM, LTE, 5G, FIBER, etc.
    "vendor":           "VENDOR",
    "domain":           "DOMAIN",
    "ne_status":        "NE_STATUS",
    "operational_state":"OPERATIONAL_STATE",
    "is_virtual":       "IS_VIRTUAL",
    "sw_version":       "SW_VERSION",
    "parent_ne_id":     "PARENT_NE_ID_FK",   # hierarchy!
    "geo_l1":           "GEOGRAPHY_L1_ID_FK",
    "geo_l2":           "GEOGRAPHY_L2_ID_FK",
    "geo_l3":           "GEOGRAPHY_L3_ID_FK",
    "geo_l4":           "GEOGRAPHY_L4_ID_FK",
    "latitude":         "LATITUDE",
    "longitude":        "LONGITUDE",
    "in_service_date":  "IN_SERVICE_DATE",
    "category":         "CATEGORY",          # SWITCH, SPLITTER, VM, SERVER
    "ne_stage":         "NE_STAGE",
    "bgp_status":       "BGP_STATUS",
    "ospf_status":      "OSPF_STATUS",
    "lldp_status":      "LLDP_STATUS",
    "admin_state":      "ADMIN_STATE",
    "band":             "BAND",
    "frequency":        "FREQUENCY",
    "enb_id":           "ENB_ID",
    "pci":              "PCI",
    "azimuth":          "AZIMUTH",
    "warranty_end":     "WARRANTY_END_DATE", # aging indicator!
}

# ─────────────────────────────────────────────
# TOPOLOGY LINKS — network graph edges
# ─────────────────────────────────────────────
LINK_TABLES = {
    "bgp": {
        "table":  "BGP_LINK",
        "src":    "SOURCE_NE_ID",
        "dst":    "DESTINATION_NE_ID",
        "status": "STATUS",
        "type":   "bgp"
    },
    "lldp": {
        "table":  "LLDP_LINK",
        "src":    "SOURCE_INTERFACE_NE_ID",
        "dst":    "DESTINATION_INTERFACE_NE_ID",
        "status": None,
        "type":   "lldp"
    },
    "ospf": {
        "table":  "OSPF_LINK",
        "src":    "SOURCE_NE_ID",
        "dst":    "DESTINATION_NE_ID",
        "status": None,
        "type":   "ospf"
    },
    "isis": {
        "table":  "ISIS_LINK",
        "src":    "SOURCE_NE_FK",
        "dst":    "DESTINATION_NE_FK",
        "status": "STATUS",
        "type":   "isis",
        # ISIS has live utilization/error/drop rates — very valuable!
        "src_utilization": "SOURCE_UTILIZATION",
        "src_error_rate":  "SOURCE_ERROR_RATE",
        "src_drop_rate":   "SOURCE_DROP_RATE",
        "dst_utilization": "TARGET_UTILIZATION",
        "dst_error_rate":  "TARGET_ERROR_RATE",
        "dst_drop_rate":   "TARGET_DROP_RATE",
        "src_util_sev":    "SOURCE_UTILIZATION_SEVERITY",  # HEALTHY/WARNING/CRITICAL
        "dst_util_sev":    "TARGET_UTILIZATION_SEVERITY",
    },
    "physical": {
        "table":  "PHYSICAL_LINK",
        "src":    "SOURCE_INTERFACE_NE_ID",
        "dst":    "DESTINATION_INTERFACE_NE_ID",
        "status": None,
        "type":   "physical"
    },
}

# ─────────────────────────────────────────────
# KPI COUNTER — performance counter definitions
# ─────────────────────────────────────────────
KPI_COUNTER_COLS = {
    "id":           "KPI_COUNTER_ID_PK",
    "name":         "COUNTER_HEADER_NAME",
    "technology":   "TECHNOLOGY",
    "unit":         "UNIT",
    "description":  "DESCRIPTION",
    "param_type":   "PARAM_TYPE",
    "kpi_range":    "KPI_RANGE",
    "time_agg":     "TIME_AGGREGATION",
    "node_agg":     "NODE_AGGREGATION",
    "sampling":     "SAMPLING_INTERVAL",
    "delta":        "DELTA",
    "granularity":  "GRANULARITY",
}

# ─────────────────────────────────────────────
# KPI FORMULA — calculated KPI definitions
# ─────────────────────────────────────────────
KPI_FORMULA_COLS = {
    "id":           "KPI_FORMULA_ID_PK",
    "name":         "KPI_NAME",
    "formula":      "KPI_FORMULA",
    "description":  "KPI_FORMULA_DESC",
    "domain":       "DOMAIN",
    "technology":   "TECHNOLOGY",
    "vendor":       "VENDOR",
    "threshold":    "THRESHOLD",
    "unit":         "KPI_UNIT",
    "range_low":    "RANGE_GTEQ",
    "range_high":   "RANGE_LTEQ",
    "kpi_group":    "KPI_GROUP",
    "node":         "NODE",
    "time_agg":     "KPI_TIME_AGGREGATION",
    "node_agg":     "KPI_NODE_AGGREGATION",
}

# ─────────────────────────────────────────────
# TECHNOLOGY ENUM from your schema
# ─────────────────────────────────────────────
TECHNOLOGY_VALUES = [
    'FDD', 'FDD10', 'FDD5', 'GSM', 'LTE',
    'T2G', 'T3G', 'T4G', 'T5G', 'TDD',
    'TDD10', 'TDD20', 'UMTS', 'WIFI',
    'FIBER', 'Wireline', 'T4G_T5G', 'COMMON', 'HARDWARE'
]

# ─────────────────────────────────────────────
# HIERARCHY LAYERS (from your geography FKs)
# ─────────────────────────────────────────────
GEO_HIERARCHY = {
    "L1": "GEOGRAPHY_L1_ID_FK",  # Country / Circle
    "L2": "GEOGRAPHY_L2_ID_FK",  # Region / Zone
    "L3": "GEOGRAPHY_L3_ID_FK",  # District / Cluster
    "L4": "GEOGRAPHY_L4_ID_FK",  # Site / Cell level
}

# NE hierarchy via PARENT_NE_ID_FK — parent-child tree
NE_PARENT_COL = "PARENT_NE_ID_FK"
