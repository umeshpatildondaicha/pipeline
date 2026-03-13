# NOC ML Pipeline — Tests

This folder contains tests that stream alarms from the database and run the trained models to verify predictions.

## Stream inference test

Simulates streaming alarms from the `ALARM` table, runs root-cause and propagation models, and writes results.

### Prerequisites

- Pipeline has been run at least once (`run_pipeline.py --mode full`) so that:
  - `data/processed/master_features.pkl` and `network_elements.pkl` exist
  - `data/models/deploy/` contains `root_cause_classifier.onnx`, `alarm_propagation_rules.json`, feature/label JSONs
- Database has the `ALARM` table with columns: `ENTITY_ID`, `ALARM_CODE`, `OPEN_TIME`, `SEVERITY`, etc.

### Run

From the **pipeline** directory (parent of `tests/`):

```bash
# Use pipeline venv
../.venv/bin/python stream_inference_test.py

# Or with options
../.venv/bin/python stream_inference_test.py --batch-size 500 --limit 2000 --output-dir ./output
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size` | 1000 | Alarms per stream batch |
| `--limit` | 5000 | Max alarms to process (0 = no limit) |
| `--output-dir` | `tests/output` | Where to write `stream_inference_results.csv` and summary |
| `--order` | OPEN_TIME | Column to order by when streaming |

### Output

- **stream_inference_results.csv**: one row per alarm with columns:
  - alarm_id, open_time, entity_id, alarm_code, severity, predicted_root_cause, confidence, propagation_count, propagation_alarms (JSON array)
- Console: summary counts, top predicted root causes, sample rows.
