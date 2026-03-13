# Pipeline & Architecture Analysis

This document summarizes what’s wrong in the current NOC ML pipeline and how it fits (or doesn’t) with the FaultAI backend architecture. Use it for follow-up questions and refactors.

---

## 1. Goal (from your description)

- **FaultAI** should behave like a NOC manager: recommendations, root-cause detection, actions.
- The **Python pipeline** trains models (XGBoost, LSTM, GNN, propagation, anomaly) and exports ONNX/artifacts.
- The **Spring Boot backend** (FaultAI) consumes those artifacts and runs the orchestration + Brain (ML, KG, memory, LLM).
- Target: replace the human NOC manager with this model when you stream real alarms/data.

---

## 2. Pipeline ↔ Backend split

| Pipeline (Python) | Backend (Spring) |
|-------------------|------------------|
| Trains root-cause (XGBoost), sequence (LSTM), GNN, propagation rules, anomaly | Uses ONNX + rules via `NocInferenceService` / `NocMlSignalProvider` |
| Writes to `data/models/deploy/` | Reads from `noc.models.deploy-dir` (same path or copied) |
| No direct link to Neo4j/MySQL of FaultAI | Brain uses MySQL (alarms, experience, memory) + Neo4j (KG) + LLM |

**Gap:** The pipeline trains on **your** MySQL (railtel); the backend expects to run on **its** MySQL/Neo4j. Schema and ID semantics (e.g. alarm ↔ NE) must align between pipeline output and backend config.

---

## 3. What’s wrong in the pipeline code

### 3.1 Training data join → 0 samples

- **Symptom:** “Training samples after join: 0” then fallback to 50k synthetic.
- **Cause:** Alarms are joined to `master_df` (NE features) on NE identifier. Your DB uses `ENTITY_ID` for the alarm’s NE; the code looks for `NE_ID_FK`, `NE_ID`, etc. Even when the loader renames to `NE_ID_FK`, the **values** in that column may not match `master_df["ID"]` (integer) or `master_df["NE_ID"]` (string) depending on how your ALARM table references NEs.
- **Fixes applied / to do:**
  - Include `ENTITY_ID` in the candidate list for the NE key in `train_root_cause.prepare_training_data`.
  - Try join on `master_df["ID"]` first (with numeric coercion); if result is empty, try join on `master_df["NE_ID"]` with both sides as string so that string identifiers (e.g. hostname) match.

### 3.2 Hardcoded / config drift

- Most magic numbers are now in `config.py` / `.env` (correlation window, XGB/LSTM/GNN params, retrain thresholds, deploy dir, etc.).
- **Remaining:**
  - `NocInferenceService`: `MAX_SEQ_LEN = 20` is hardcoded; should match `SEQ_MAX_LEN` from pipeline config (or a shared property).
  - Root-cause **label set** in synthetic generator and elsewhere should only come from `root_cause_labels.json` (or config); the “default 9 buckets” are a fallback only when the file is missing.

### 3.3 Data loader ↔ schema

- Alarm query uses `get_alarm_query(schema)` and selects with AS aliases (`ENTITY_ID AS ne_id`, …), so the DataFrame has canonical names; then we rename `ne_id` → `NE_ID_FK`. So downstream sees `NE_ID_FK`. If in some code paths the schema is not used and raw columns (e.g. `ENTITY_ID`) appear, `train_root_cause` and `alarm_correlator` should recognize both `NE_ID_FK` and `ENTITY_ID` for the NE key.

### 3.4 LSTM / GNN resource usage

- **LSTM:** With limited RAM (e.g. 169M free + heavy swap), training was effectively stuck. Mitigations already in place: `SEQ_MAX_TRAIN_SAMPLES`, smaller batch, MPS (Apple Silicon), per-batch logging.
- **GNN:** Dense 5k×5k adjacency was replaced with sparse COO; `GNN_MAX_NODES` caps node count. Ensure `train_gnn_model` uses the same feature set as the root-cause model (via `root_cause_features.json`) so Spring-side inference is consistent.

### 3.5 ONNX opset

- PyTorch ONNX export can emit opset 15+; requesting opset 14 caused version-converter failures. Bumping to opset 17 (or at least 15) avoids that.

### 3.6 Retrain / deploy

- `retrain_on_delta`: SQLAlchemy 2.x compatibility (no `engine.execute`, use `with engine.connect()`), config-driven thresholds, per-class F1, and rollback on failed deploy are addressed.
- Spring: `noc.models.deploy-dir` is required; no silent fallback to `/models/deploy`.

---

## 4. Architecture issues (pipeline + FaultAI)

### 4.1 Two sources of “root cause”

- **Pipeline:** XGBoost + LSTM + GNN produce root-cause **codes/labels** (e.g. from `root_cause_labels.json`).
- **Backend:** Brain uses `TelecomMlEngine` (NaiveBayes + learned rules), KG recall, and LLM agents, then **AgreementChecker** (ML vs KG vs LLM).
- **Risk:** Label sets or code semantics (e.g. “BACKHAUL_ISSUE”) must match between pipeline output and backend (MySQL/Neo4j and LLM prompts). Any mismatch causes agreement logic to fail or metrics to be wrong.

### 4.2 Where pipeline models plug in

- ARCHITECTURE says: “Optional: NocMlSignalProvider (sequence + GNN from NOC ONNX → MlSignal)”. So the pipeline’s ONNX models are one **input** to the Brain (as MlSignal), not the only one. The rest is KG + memory + LLM.
- **Gap:** Clear contract for MlSignal (feature vector, root-cause code, confidence) and who fills it (NocInferenceService vs existing TelecomMlEngine) is missing. Defining that contract (e.g. in PROJECT_DOCUMENTATION or a small ADR) will avoid duplication and inconsistency.

### 4.3 Training data provenance

- Pipeline trains from **railtel** MySQL (alarms, NEs, topology, alarm library). FaultAI persists experience and learning in **its** MySQL and Neo4j.
- For “replace NOC manager” you need either:
  - Periodic sync of experience/cleared alarms from FaultAI back into the pipeline’s DB (or a shared DB), or
  - Pipeline to read from the same DB as FaultAI so that learning is on the same data. Right now the two can be disconnected.

### 4.4 Operational concerns

- **Scheduler:** Pipeline has no scheduler; you run `run_pipeline.py --mode full` or `--mode incremental` (e.g. via cron). FaultAI has `AlarmProcessingScheduler` reading from MySQL. They are independent; ensure batch size and frequency don’t conflict (e.g. pipeline overwriting deploy dir while Spring is loading).
- **Versioning:** `manifest.json` with timestamp is good; ensure the backend only swaps models after a successful load and optional validation (e.g. one inference run) to avoid partial deploy.

---

## 5. Recommended next steps

1. **Fix join in `train_root_cause.prepare_training_data`:** Add ENTITY_ID to NE-key candidates; try join on `ID` then fallback to `NE_ID` (string) so real alarms produce non-zero training samples when your ALARM.ENTITY_ID matches NE_ID or ID.
2. **Align MAX_SEQ_LEN:** Make Spring’s sequence length configurable (e.g. `noc.models.sequence-max-len`) and set it from the same value as pipeline `SEQ_MAX_LEN` (or document the single source of truth).
3. **Define MlSignal contract:** Document how NocInferenceService (and optionally export_entity_features / entity_features.json) feeds the Brain: feature list, root-cause code set, confidence. Then implement NocMlSignalProvider to that contract.
4. **Single label set:** Ensure `root_cause_labels.json` is the only source of class names for training, evaluation, and backend agreement logic; remove or narrow hardcoded fallbacks.
5. **Data flow doc:** One-page diagram: railtel MySQL → pipeline → deploy/ → Spring → Brain → MySQL/Neo4j (and optional feedback path). Helps onboarding and avoids the “two worlds” confusion.

---

## 6. Session / branch context

- Refactor work was done on branch `claude/refactor-telecom-pipeline-Jvi3F` (security, config, GNN sparse, ONNX opset, retrain rollback, etc.).
- If that branch is not on your machine: `git fetch origin && git checkout claude/refactor-telecom-pipeline-Jvi3F`.
- Merge to main via PR or local merge when ready.

This file is the context for “what’s wrong” and “what to do next” for the telecom NOC model and FaultAI backend.
