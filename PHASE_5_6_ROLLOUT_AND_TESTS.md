# Phase 5.6 Safety Systems — rollout and smoke tests

## Files added
- `tradarbot/safety/__init__.py`
- `tradarbot/safety/kill_switch.py`
- `tradarbot/safety/safe_mode.py`
- `tradarbot/safety/stale_data_guard.py`
- `tradarbot/safety/health_rules.py`

## Files patched
- `tradarbot/risk/risk_manager.py`
- `tradarbot/app/main.py`
- `tradarbot/core/engine.py`
- `tradarbot/core/state.py`
- `tradarbot/storage/sqlite_store.py`
- `config/tradar.yaml`

## Rollout order
1. Apply files to a clean branch.
2. Run syntax checks.
3. Start with `TRADAR_PROFILE=paper`; do not use live mode first.
4. Verify safety tables exist.
5. Run forced test orders only in paper/dry-run.
6. Let the bot run long enough for the safety loop to write snapshots.
7. Review `safety_events`, `runtime_health`, and `safety_state_snapshots`.
8. Only then move to VPS/systemd deployment.

## Smoke commands

### 1. Syntax/import smoke
```bash
PYTHONPATH=. ./.venv/bin/python -m py_compile \
  tradarbot/safety/*.py \
  tradarbot/risk/risk_manager.py \
  tradarbot/app/main.py \
  tradarbot/core/engine.py \
  tradarbot/core/state.py \
  tradarbot/storage/sqlite_store.py
```

### 2. SQLite safety table verification
```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from tradarbot.storage.sqlite_store import SQLiteStore
s = SQLiteStore('tradarbot.db')
s.init_schema()
for t in ['safety_events','runtime_health','safety_state_snapshots']:
    print(t, s.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone())
PY
```

### 3. Health loop smoke test
```bash
TRADAR_PROFILE=paper PYTHONPATH=. ./.venv/bin/python run.py --config config/tradar.yaml --profile paper --log-level INFO 2>&1 | tee phase5_6_safety_smoke.log
```

Then in another terminal:
```bash
tail -f phase5_6_safety_smoke.log | grep -Ei "SAFE_MODE|KILL_SWITCH|STALE_DATA|HEALTH_RULE|ENTRY_BLOCKED_SAFETY|EXIT_ALLOWED_SAFETY|runtime_health"
```

### 4. Safe mode trigger simulation
```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from types import SimpleNamespace
from tradarbot.core.state import State
from tradarbot.core.events import OrderIntent
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.safety.safe_mode import SafeModeManager
from tradarbot.safety.kill_switch import KillSwitchManager
from tradarbot.safety.stale_data_guard import StaleDataGuard
from tradarbot.safety.health_rules import HealthMonitor
cfg={'safety': {'enabled': True, 'safe_mode': {'enabled': True, 'size_multiplier': 0.5}, 'kill_switch': {'enabled': True}, 'stale_data': {'enabled': False}, 'health': {'enabled': False}}, 'risk': {'enabled': True}}
store=SQLiteStore('tradarbot.db'); store.init_schema()
state=State(); broker=SimpleNamespace(positions={}, cash=10000)
ctx=SimpleNamespace(state=state, store=store, broker=broker)
risk=RiskManager(cfg); sm=SafeModeManager(cfg, store); ks=KillSwitchManager(cfg, store); sg=StaleDataGuard(cfg, store); hm=HealthMonitor(cfg, store, sg)
risk.attach_safety(ks, sm, hm, sg)
sm.activate('manual_test')
d=risk.check(OrderIntent('BUY','ETHUSDT',1.0,100.0,'IOC'), ctx, 'ml_strategy')
print(d)
PY
```

### 5. Kill switch trigger simulation
```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from types import SimpleNamespace
from tradarbot.core.state import State
from tradarbot.core.events import OrderIntent
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.safety.kill_switch import KillSwitchManager
from tradarbot.safety.safe_mode import SafeModeManager
from tradarbot.safety.stale_data_guard import StaleDataGuard
from tradarbot.safety.health_rules import HealthMonitor
cfg={'safety': {'enabled': True, 'kill_switch': {'enabled': True}, 'safe_mode': {'enabled': True}, 'stale_data': {'enabled': False}, 'health': {'enabled': False}}, 'risk': {'enabled': True}}
store=SQLiteStore('tradarbot.db'); store.init_schema()
state=State(); broker=SimpleNamespace(positions={}, cash=10000)
ctx=SimpleNamespace(state=state, store=store, broker=broker)
risk=RiskManager(cfg); ks=KillSwitchManager(cfg, store); sm=SafeModeManager(cfg, store); sg=StaleDataGuard(cfg, store); hm=HealthMonitor(cfg, store, sg)
risk.attach_safety(ks, sm, hm, sg)
ks.activate('manual_test')
print('BUY:', risk.check(OrderIntent('BUY','ETHUSDT',1,100,'IOC'), ctx, 'test'))
print('SELL:', risk.check(OrderIntent('SELL','ETHUSDT',1,100,'IOC'), ctx, 'test'))
PY
```

### 6. Stale data simulation
```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
import time
from tradarbot.safety.stale_data_guard import StaleDataGuard
cfg={'safety': {'enabled': True, 'stale_data': {'enabled': True, 'max_book_age_seconds': 1, 'stale_data_kill_seconds': 3}}}
g=StaleDataGuard(cfg)
g.update_book('ETHUSDT', int((time.time()-5)*1000))
print([v.to_dict() for v in g.entry_violations('ETHUSDT')])
PY
```

### 7. API error escalation test
```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from types import SimpleNamespace
from tradarbot.core.state import State
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.safety.health_rules import HealthMonitor
store=SQLiteStore('tradarbot.db'); store.init_schema()
state=State(); state.api_error_counts=99
ctx=SimpleNamespace(state=state)
h=HealthMonitor({'safety': {'enabled': True, 'health': {'enabled': True, 'api_error_warn_threshold': 2, 'api_error_kill_threshold': 5}}}, store)
results=h.evaluate(ctx)
print(h.worst_status(results), [r.to_dict() for r in results])
PY
```

### 8. Runtime snapshot verification
```bash
sqlite3 tradarbot.db "SELECT ts_ms,status,safe_mode,kill_switch,stale_symbols FROM runtime_health ORDER BY id DESC LIMIT 10;"
sqlite3 tradarbot.db "SELECT ts_ms,safe_mode,kill_switch,reasons_json FROM safety_state_snapshots ORDER BY id DESC LIMIT 10;"
```

### 9. Replay parity verification
Use your existing Phase 4/5 replay command unchanged. This patch does not alter replay ranking, ensemble math, or Phase 4 tables.

### 10. Forced flatten verification
Set `safety.kill_switch.flatten_positions_on_trigger: true` only in paper/dry-run, create a small forced test order, then trigger the kill switch simulation while the runtime is active. Watch for `CLOSE_ALL_REQUESTED` and `KILL_SWITCH_ACTIVATED` logs.

## Grep verification
```bash
grep -R "SAFE_MODE_ENABLED\|KILL_SWITCH_ACTIVATED\|STALE_DATA_DETECTED\|HEALTH_RULE_TRIGGERED\|ENTRY_BLOCKED_SAFETY\|EXIT_ALLOWED_SAFETY" -n tradarbot
```
