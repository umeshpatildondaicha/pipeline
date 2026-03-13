# How the stream inference results are calculated

This document explains exactly how each column in `stream_inference_results.csv` is produced.

---

## 1. Data flow (high level)

```
ALARM table (DB)  →  join with NE  →  join with master_features  →  feature vector
                                                                         ↓
stream_inference_results.csv  ←  propagation lookup  ←  ONNX model  ←  feature vector
```

---

## 2. Feature vector (input to the model)

**What it is:** A single row of 41 float numbers in a fixed order. The model was trained on these same 41 features.

**How it’s built:**

1. **Alarm → NE:** Each alarm has `ENTITY_ID` (e.g. `172.31.31.209`). We match it to `network_elements.pkl` on `NE_ID` to get the numeric `ID` of that network element.

2. **NE → features:** We take the row for that `ID` from `master_features.pkl` (built by `feature_builder.py` from NETWORK_ELEMENT + topology + ISIS link health + hierarchy).

3. **Vector in fixed order:** We build a vector of length 41 by reading the 41 feature names from `root_cause_features.json` in order and, for each name, taking the value from that master row (or 0 if missing/NaN):

   ```text
   feature_vector[i] = master_row[feature_cols[i]]  if present else 0.0
   ```

**Code:** `build_feature_row()` in `stream_inference_test.py` (lines 106–119).

**Feature list (order matters):**  
`data/models/root_cause_features.json` — e.g. `NE_TYPE_ENC`, `TECHNOLOGY_ENC`, `VENDOR_ENC`, …, `LINK_HEALTH_SCORE`, `AZIMUTH`, `ELECTRICAL_TILT`, `MECHANICAL_TILT`.

---

## 3. Predicted root cause

**What it is:** One of the 9 root-cause labels (e.g. `CONFIGURATION_ERROR`, `HARDWARE_FAILURE`).

**How it’s calculated:**

1. The 41-dim feature vector is passed to the **ONNX model** `root_cause_classifier.onnx` (XGBoost classifier exported from `train_root_cause.py`).

2. The model returns:
   - **Class index:** integer 0–8 (which of the 9 classes).
   - **Probabilities:** one probability per class (they sum to 1).

3. We map the **winning class index** to a string using `root_cause_labels.json`:

   ```text
   predicted_root_cause = label_list[predicted_class_index]
   ```

   Example: index `1` → `"CONFIGURATION_ERROR"`.

**Code:** `run_inference()` (lines 122–135) and the loop that builds each result row (lines 224–238).

**Labels (index → label):**  
`data/models/root_cause_labels.json`:  
`["BACKHAUL_ISSUE", "CONFIGURATION_ERROR", "HARDWARE_FAILURE", "INTERFACE_ERROR", "LATENCY_HIGH", "LINK_CONGESTION", "PACKET_LOSS", "POWER_ISSUE", "UNKNOWN"]`.

---

## 4. Confidence

**What it is:** The model’s probability for the predicted class (a number between 0 and 1).

**How it’s calculated:**

- After `session.run()`, we have probabilities for all 9 classes.
- **Confidence = probability of the predicted class** (the max of the 9):

  ```text
  confidence = max(probabilities for the 9 classes)
  ```

  So if the model outputs 40% for CONFIGURATION_ERROR and 60% for HARDWARE_FAILURE, we predict HARDWARE_FAILURE and confidence = 0.6.

**Code:** `run_inference()` — `confidences = np.max(probs, axis=1)` (or equivalent for dict output).

---

## 5. Propagation count and propagation alarms

**What they are:** How many “consequent” alarms the rules predict, and their names/codes.

**How they’re calculated:**

1. We use **`alarm_propagation_rules.json`** produced by `train_propagation.py`. Structure:

   ```json
   {
     "ALARM_CODE_OR_NAME": [
       { "consequent": "NEXT_ALARM_1", "confidence": 0.8, "avg_delay_sec": 120, ... },
       { "consequent": "NEXT_ALARM_2", ... }
     ],
     ...
   }
   ```

2. For each alarm we take its **alarm code** (e.g. `8588` or `"8588"`) and look it up as the **key** in this JSON:

   ```text
   rules = propagation_rules[str(alarm_code)]
   ```

3. **propagation_count** = number of consequent entries we keep (capped at 5):  
   `len(rules[:5])`.

4. **propagation_alarms** = list of the `"consequent"` strings for those entries, e.g.  
   `["BGP_SESSION_DOWN", "NE_REBOOT"]`, stored in the CSV as a JSON array.

**Code:** `get_propagation()` (lines 138–150).

**Why you often see 0 and []:**  
The current rules file was learned from **synthetic** alarm **names** (e.g. `"DC_POWER_LOW"`, `"OPTICAL_OSNR_DEGRADED"`). Your real table uses **numeric** `ALARM_CODE` (e.g. `8588`, `7459`). So `propagation_rules["8588"]` and `propagation_rules["7459"]` don’t exist → we get 0 consequents and `[]`.  
Once you retrain the propagation model on real alarm data (with your numeric or string codes), the keys in the JSON will match and these columns will start to fill.

---

## 6. Summary table

| Column                  | Source / formula |
|-------------------------|------------------|
| alarm_id, open_time, entity_id, alarm_code, alarm_name, severity | Directly from `ALARM` table. |
| predicted_root_cause    | `root_cause_labels.json[ ONNX predicted class index ]`. |
| confidence              | Max of the 9 class probabilities from the ONNX model. |
| propagation_count       | Number of entries for `alarm_code` in `alarm_propagation_rules.json` (up to 5). |
| propagation_alarms      | List of `"consequent"` from those entries (JSON array in CSV). |

---

## 7. Where each artifact comes from (pipeline)

| Artifact                    | Produced by                    | Used for |
|----------------------------|---------------------------------|----------|
| master_features.pkl        | feature_builder.py              | 41-dim feature vector per NE. |
| network_elements.pkl        | data_loader.py                  | ENTITY_ID → NE ID. |
| root_cause_classifier.onnx | train_root_cause.py (XGBoost→ONNX) | Root cause class + probabilities. |
| root_cause_features.json   | train_root_cause.py             | Order of the 41 features. |
| root_cause_labels.json     | train_root_cause.py             | Index → root cause label. |
| alarm_propagation_rules.json | train_propagation.py          | Alarm code → consequent alarms. |
