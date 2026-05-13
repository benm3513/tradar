# Phase 5.7 Logging, Monitoring, and Ops Visibility — rollout and smoke tests

## Files added
- `tradarbot/monitoring/__init__.py`
- `tradarbot/monitoring/metrics.py`
- `tradarbot/monitoring/heartbeat.py`
- `tradarbot/monitoring/reporting.py`
- `tradarbot/portfolio/__init__.py`
- `tradarbot/portfolio/positions.py` compatibility dataclasses for the uploaded Phase 5.6 store/state snapshot
- `scripts/print_live_status.py`
- `scripts/export_live_metrics.py`

## Files patched
- `tradarbot/storage/sqlite_store.py`
- `tradarbot/app/main.py`
- `run.py`
- `deploy/healthcheck.py`
- `config/tradar.yaml`

## What changed
- Adds SQLite tables:
  - `runtime_heartbeat`
  - `runtime_metrics`
  - `runtime_status_events`
- Adds a defensive async monitoring loop to `app/main.py`.
- Persists heartbeat rows, flattened metrics snapshots, and runtime status events.
- Emits structured operational logs:
  - `HEARTBEAT_OK`
  - `METRICS_SNAPSHOT`
  - `RUNTIME_SUMMARY`
  - `EXECUTION_SUMMARY`
  - `MARKET_DATA_SUMMARY`
  - `ML_RUNTIME_SUMMARY`
- Adds SSH-friendly status and metrics export scripts.
- Keeps monitoring failure-isolated via `MONITORING_LOOP_FAILED` logging; failures do not crash runtime.

## Config added
```yaml
monitoring:
  enabled: true
  heartbeat:
    enabled: true
    interval_seconds: 15
  metrics:
    enabled: true
    snapshot_interval_seconds: 30
    persist_metrics: true
  reporting:
    enabled: true
    summary_interval_seconds: 120
```

## Smoke commands

### 1. Syntax/import validation
```bash
PYTHONPATH=. ./.venv/bin/python -m py_compile \
  tradarbot/monitoring/*.py \
  tradarbot/storage/sqlite_store.py \
  tradarbot/app/main.py \
  scripts/print_live_status.py \
  scripts/export_live_metrics.py \
  deploy/healthcheck.py \
  run.py
```

### 2. Deployment healthcheck
```bash
PYTHONPATH=. ./.venv/bin/python deploy/healthcheck.py --config config/tradar.yaml
```
Expected:
```text
HEALTHCHECK_OK config=config/tradar.yaml
```

### 3. Verify monitoring tables
```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from tradarbot.storage.sqlite_store import SQLiteStore
s = SQLiteStore('tradarbot.db')
s.init_schema()
for t in ['runtime_heartbeat','runtime_metrics','runtime_status_events']:
    print(t, s.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone())
PY
```

### 4. Start/observe runtime
```bash
TRADAR_PROFILE=paper PYTHONPATH=. ./.venv/bin/python run.py --config config/tradar.yaml --profile paper --log-level INFO
```

In another terminal:
```bash
journalctl -u tradar -f | grep -Ei "HEARTBEAT_OK|METRICS_SNAPSHOT|RUNTIME_SUMMARY|EXECUTION_SUMMARY|MARKET_DATA_SUMMARY|ML_RUNTIME_SUMMARY|MONITORING_LOOP_FAILED"
```

### 5. Print live status
```bash
PYTHONPATH=. ./.venv/bin/python scripts/print_live_status.py --db-path tradarbot.db
```

or through `run.py`:
```bash
PYTHONPATH=. ./.venv/bin/python run.py --print-status --db-path tradarbot.db
```

### 6. Export metrics
```bash
PYTHONPATH=. ./.venv/bin/python scripts/export_live_metrics.py \
  --db-path tradarbot.db \
  --output runtime_metrics.csv
```

Filtered recent export:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/export_live_metrics.py \
  --db-path tradarbot.db \
  --metric-group execution \
  --since 6h \
  --output execution_metrics.csv
```

Aggregated export:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/export_live_metrics.py \
  --db-path tradarbot.db \
  --metric-group execution \
  --since 6h \
  --aggregate-window-seconds 300 \
  --output execution_metrics_5m.csv
```

## VPS/systemd validation
```bash
sudo systemctl restart tradar
systemctl status tradar
journalctl -u tradar -f
```

After at least 60 seconds:
```bash
cd /opt/tradar
PYTHONPATH=. ./.venv/bin/python scripts/print_live_status.py --db-path tradarbot.db
PYTHONPATH=. ./.venv/bin/python scripts/export_live_metrics.py --db-path tradarbot.db --since 30m --output /tmp/tradar_metrics.csv
```

## Expected successful behavior
- Heartbeat rows update approximately every 15 seconds.
- Metrics rows update approximately every 30 seconds.
- Summary logs emit approximately every 120 seconds.
- No `MONITORING_LOOP_FAILED` spam.
- No SQLite lock escalation.
- Existing safety loop remains unchanged.
- Existing paper/dry_run/live profile guards remain intact.
