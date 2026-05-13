from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

log = logging.getLogger("tradarbot.monitoring.heartbeat")


def now_ms() -> int:
    return int(time.time() * 1000)


class HeartbeatWriter:
    """Small wrapper around SQLiteStore heartbeat/status persistence."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, process_start_ts: Optional[float] = None):
        root = dict(cfg or {})
        mon = dict(root.get("monitoring", {}) or {})
        hb = dict(mon.get("heartbeat", {}) or {})
        self.enabled = bool(hb.get("enabled", mon.get("enabled", True)))
        self.process_start_ts = float(process_start_ts or time.time())
        self.pid = os.getpid()

    def write(self, ctx: Any, status: str = "OK", details: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        store = getattr(ctx, "store", None)
        if store is None or not hasattr(store, "insert_runtime_heartbeat"):
            return
        state = getattr(ctx, "state", None)
        cfg = getattr(ctx, "cfg", {}) or {}
        profile = (cfg.get("runtime", {}) or {}).get("profile") if isinstance(cfg, dict) else None
        profile = profile or os.environ.get("TRADAR_PROFILE", "paper")
        uptime = max(0.0, time.time() - self.process_start_ts)
        safe_mode = bool(getattr(state, "runtime_safe_mode", False))
        kill_switch = bool(getattr(state, "runtime_kill_switch", False))
        ts_ms = now_ms()
        store.insert_runtime_heartbeat(
            ts_ms=ts_ms,
            profile=str(profile),
            status=str(status),
            pid=self.pid,
            uptime_seconds=uptime,
            safe_mode=safe_mode,
            kill_switch=kill_switch,
            details=details or {},
        )
        if state is not None:
            state.last_heartbeat_ts_ms = ts_ms

    def event(self, ctx: Any, event_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        store = getattr(ctx, "store", None)
        if store is not None and hasattr(store, "insert_runtime_status_event"):
            store.insert_runtime_status_event(
                ts_ms=now_ms(),
                event_type=event_type,
                severity=severity,
                message=message,
                details=details or {},
            )
