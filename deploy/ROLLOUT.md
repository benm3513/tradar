# Tradar Phase 5.9 — Go-Live Cutover Runbook

This runbook is intentionally fail-closed. If any step is ambiguous, stay in `paper`, `shadow`, or `dry_run_live`.

## Deployment checklist

- Repo is deployed at `/opt/tradar`.
- Runtime user exists: `tradar`.
- Service exists: `tradar`.
- Python venv exists: `/opt/tradar/.venv`.
- SQLite DB path is known and backed up.
- `config/tradar.yaml` remains paper-first.
- `config/tradar.live.yaml` exists for live-only operation.
- `.env` or `/etc/tradar/tradar.env` contains API credentials.
- `deploy/healthcheck.py` passes before live mode.
- `scripts/print_live_status.py` works against the active DB.
- ML artifacts exist in the configured artifact directory.
- Rollback script is executable: `chmod +x deploy/rollback.sh`.

## VPS setup checklist

```bash
cd /opt/tradar
git status --short
./.venv/bin/python --version
systemctl status tradar --no-pager
ls -lah config/tradar.yaml config/tradar.live.yaml deploy/rollback.sh deploy/healthcheck.py
```

## Environment variable checklist

For Alpaca paper/live execution:

```bash
sudo grep -E "TRADAR_|ALPACA_|EXECUTION_" /etc/tradar/tradar.env
```

Required for live:

```bash
TRADAR_PROFILE=live
TRADAR_CONFIG=/opt/tradar/config/tradar.live.yaml
TRADAR_CONFIRM_LIVE=true
TRADAR_LIVE_ACK=I_UNDERSTAND_LIVE_RISK
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
EXECUTION_PROVIDER=alpaca
EXECUTION_MODE=live
```

Recommended for first live session:

```bash
TRADAR_LOG_LEVEL=INFO
TRADAR_DB_PATH=/opt/tradar/data/tradarbot.db
TRADAR_ARTIFACT_DIR=/opt/tradar/artifacts/models/v50_sig_importance
```

## Healthcheck procedure

```bash
cd /opt/tradar
set -a
source /etc/tradar/tradar.env
set +a

PYTHONPATH=. ./.venv/bin/python run.py \
  --config config/tradar.live.yaml \
  --profile live \
  --live-confirm \
  --healthcheck
```

Expected:

```text
HEALTHCHECK_OK
```

Do not continue if healthcheck fails.

## Startup validation steps

```bash
cd /opt/tradar

PYTHONPATH=. ./.venv/bin/python run.py \
  --config config/tradar.live.yaml \
  --profile live \
  --live-confirm \
  --print-status
```

Inspect current service state:

```bash
systemctl status tradar --no-pager
journalctl -u tradar -n 200 --no-pager
```

Start service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart tradar
sudo journalctl -u tradar -f
```

Look for:

```bash
journalctl -u tradar -n 300 --no-pager | grep -Ei \
"STARTUP_MODE|LIVE_MODE_CONFIRMED|LIVE_PROVIDER_CHECK_OK|LIVE_ACCOUNT_VALIDATED|LIVE_STARTUP_CHECK_OK|KILL_SWITCH|SAFE_MODE|HEALTH"
```

## Staged rollout plan

### Stage 0 — shadow only, no execution

Config intent:

```yaml
runtime:
  profile: paper
ml_live:
  enabled: true
  mode: shadow
execution_live:
  enabled: true
  broker: paper
```

Run:

```bash
sudo systemctl restart tradar
journalctl -u tradar -f | grep -Ei "ML_SHADOW|HEARTBEAT|RUNTIME|SAFETY"
```

Validate:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_ml_shadow.py \
  --db-path /opt/tradar/data/tradarbot.db \
  --config config/tradar.yaml \
  --since 6h \
  --report \
  --parity-check
```

### Stage 1 — dry_run_live only, tiny caps

Config intent:

```yaml
runtime:
  profile: dry_run
ml_live:
  mode: paper
execution_live:
  enabled: true
  broker: dry_run_live
  mode: dry_run
```

Run:

```bash
PYTHONPATH=. ./.venv/bin/python run.py \
  --config config/tradar.live.yaml \
  --profile dry_run \
  --healthcheck

sudo systemctl restart tradar
```

Validate that orders are routed but not real exchange-capital orders:

```bash
journalctl -u tradar -n 500 --no-pager | grep -Ei \
"LIVE_ROUTED_ORDER|DRY|LIVE_ORDER|PRETRADE|REJECT|FILL"
```

### Stage 2 — live enabled, strict caps, 1–2 symbols only

Config intent:

```yaml
runtime:
  profile: live
rollout:
  stage: 2
execution_live:
  enabled: true
  broker: live
  mode: live
  tradable_symbols: [BTCUSDT, ETHUSDT]
risk:
  max_positions: 1
  max_notional_per_trade_usd: 25
  max_total_exposure_usd: 50
```

Run manually first:

```bash
PYTHONPATH=. ./.venv/bin/python run.py \
  --config config/tradar.live.yaml \
  --profile live \
  --live-confirm \
  --healthcheck
```

Then systemd:

```bash
sudo systemctl restart tradar
sudo journalctl -u tradar -f
```

### Stage 3 — controlled expansion

Only expand one dimension at a time:

- symbol count
- max notional
- max positions
- daily loss cap
- max exposure

Keep rollback ready in another SSH session.

## Emergency shutdown commands

Immediate stop:

```bash
sudo systemctl stop tradar
```

Disable auto-start:

```bash
sudo systemctl disable tradar
```

Rollback to paper:

```bash
cd /opt/tradar
sudo TRADAR_CONFIG=/opt/tradar/config/tradar.live.yaml ./deploy/rollback.sh
```

Dry-run rollback preview:

```bash
cd /opt/tradar
sudo TRADAR_CONFIG=/opt/tradar/config/tradar.live.yaml ./deploy/rollback.sh --dry-run
```

## Revert to paper

```bash
cd /opt/tradar
sudo ./deploy/rollback.sh
sudo systemctl restart tradar
```

Verify:

```bash
PYTHONPATH=. ./.venv/bin/python run.py --config config/tradar.yaml --profile paper --print-status
journalctl -u tradar -n 100 --no-pager | grep -Ei "paper|ROLLBACK|LIVE_MODE_BLOCKED"
```

## Disable live execution immediately

Edit env:

```bash
sudo nano /etc/tradar/tradar.env
```

Set:

```bash
TRADAR_PROFILE=paper
TRADAR_CONFIG=/opt/tradar/config/tradar.yaml
TRADAR_CONFIRM_LIVE=false
EXECUTION_MODE=paper
```

Restart:

```bash
sudo systemctl restart tradar
```

## Recommended initial capital limits

Stage 2 first session:

```yaml
risk:
  max_positions: 1
  max_notional_per_trade_usd: 25
  max_total_exposure_usd: 50
  max_daily_loss_usd: 25
execution_live:
  tradable_symbols:
    - BTCUSDT
    - ETHUSDT
```

Do not increase caps until at least one full supervised session completes with no safety escalation, no rejected order storm, and expected accounting.

## Monitoring commands

```bash
systemctl status tradar --no-pager
journalctl -u tradar -f
journalctl -u tradar -n 500 --no-pager | grep -Ei "LIVE_|KILL_SWITCH|SAFE_MODE|HEALTH|ROLLBACK|ORDER|FILL|REJECT"
PYTHONPATH=. ./.venv/bin/python scripts/print_live_status.py --db-path /opt/tradar/data/tradarbot.db
PYTHONPATH=. ./.venv/bin/python scripts/export_live_metrics.py --db-path /opt/tradar/data/tradarbot.db --since 6h --output /tmp/tradar_metrics.csv
```

## DB inspection

```bash
sqlite3 /opt/tradar/data/tradarbot.db "SELECT * FROM runtime_heartbeat ORDER BY ts_ms DESC LIMIT 3;"
sqlite3 /opt/tradar/data/tradarbot.db "SELECT event_type,severity,message,details_json FROM runtime_status_events ORDER BY ts_ms DESC LIMIT 20;"
sqlite3 /opt/tradar/data/tradarbot.db "SELECT status,COUNT(*) FROM orders GROUP BY status;"
sqlite3 /opt/tradar/data/tradarbot.db "SELECT * FROM fills ORDER BY ts_ms DESC LIMIT 10;"
```

## Kill switch validation

Do this only in paper or dry_run first:

```bash
journalctl -u tradar -n 200 --no-pager | grep -Ei "KILL_SWITCH|SAFE_MODE"
sqlite3 /opt/tradar/data/tradarbot.db "SELECT * FROM safety_events ORDER BY ts_ms DESC LIMIT 20;"
```

If live service starts with previous kill switch state, it must block live startup and log:

```text
LIVE_STARTUP_BLOCKED
```

## Rollback validation

```bash
cd /opt/tradar
sudo ./deploy/rollback.sh --dry-run
sudo ./deploy/rollback.sh
echo $?
sudo systemctl status tradar --no-pager
grep -E "TRADAR_PROFILE|TRADAR_CONFIRM_LIVE|EXECUTION_MODE" /etc/tradar/tradar.env
```

Expected:

```text
ROLLBACK_COMPLETE
```
