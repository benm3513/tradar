#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

from tradarbot.storage.sqlite_store import SQLiteStore


def iso(ts_ms: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return "n/a"


def money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def load_status(db_path: str, tail_events: int, tail_fills: int) -> Dict[str, Any]:
    store = SQLiteStore(db_path)
    store.init_schema()
    heartbeat = store.latest_runtime_heartbeat()
    metrics = store.latest_runtime_metrics()
    safety_events = store.recent_safety_events(tail_events)
    runtime_events = store.recent_runtime_status_events(tail_events)
    fills = store.recent_fills(tail_fills)
    now_ms = int(time.time() * 1000)
    heartbeat_age_s = None
    if heartbeat and heartbeat.get("ts_ms"):
        heartbeat_age_s = max(0.0, (now_ms - int(heartbeat["ts_ms"])) / 1000.0)
    return {
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": heartbeat_age_s,
        "metrics": metrics,
        "recent_safety_events": safety_events,
        "recent_runtime_events": runtime_events,
        "recent_fills": fills,
    }


def print_human(payload: Dict[str, Any]) -> None:
    hb = payload.get("heartbeat") or {}
    metrics = payload.get("metrics") or {}
    runtime = metrics.get("runtime", {}) or {}
    execution = metrics.get("execution", {}) or {}
    md = metrics.get("market_data", {}) or {}
    ml = metrics.get("ml", {}) or {}
    safety = metrics.get("safety", {}) or {}
    system = metrics.get("system", {}) or {}

    print("TRADAR LIVE STATUS")
    print("==================")
    print(f"Heartbeat: {hb.get('status', 'UNKNOWN')} age={payload.get('heartbeat_age_seconds')}s last={iso(hb.get('ts_ms'))}")
    print(f"Profile:   {hb.get('profile') or runtime.get('runtime_profile', 'unknown')} pid={hb.get('pid') or runtime.get('pid')}")
    print(f"Uptime:    {hb.get('uptime_seconds') or runtime.get('uptime_seconds', 0):.1f}s")
    print(f"Health:    {safety.get('runtime_health_status', hb.get('status', 'UNKNOWN'))} safe_mode={bool(safety.get('safe_mode', hb.get('safe_mode', False)))} kill_switch={bool(safety.get('kill_switch', hb.get('kill_switch', False)))}")
    print()
    print("Market Data")
    print(f"  WS connected: {bool(md.get('ws_connected', False))} disconnects={int(md.get('ws_disconnect_count') or 0)}")
    print(f"  REST poll:    ok={int(md.get('poll_ok') or 0)} err={int(md.get('poll_err') or 0)} backoff={float(md.get('poll_backoff_s') or 0):.1f}s")
    print(f"  Symbols:      active={int(md.get('active_symbols') or 0)} ready={int(md.get('ready_symbols') or 0)} stale={md.get('stale_symbols') or []} global_stale={bool(md.get('stale_global', False))}")
    print()
    print("Execution")
    print(f"  Cash:         {money(execution.get('cash'))}")
    print(f"  Equity:       {money(execution.get('equity'))}")
    print(f"  Exposure:     {money(execution.get('exposure'))}")
    print(f"  PnL:          realized={money(execution.get('realized_pnl'))} unrealized={money(execution.get('unrealized_pnl'))}")
    print(f"  Positions:    {int(execution.get('open_positions') or 0)} open_orders={int(execution.get('open_orders') or 0)} rejected={int(execution.get('rejected_orders') or 0)}")
    print()
    print("ML Runtime")
    print(f"  Features:     {int(ml.get('feature_updates') or 0)} predictions={int(ml.get('prediction_count') or 0)} candidates={int(ml.get('candidate_count') or 0)} rankings={int(ml.get('ranking_count') or 0)} signals={int(ml.get('signal_count') or 0)}")
    print(f"  Failures:     inference={int(ml.get('inference_failures') or 0)} fallback={int(ml.get('fallback_predictions') or 0)}")
    print(f"  Top symbols:  {ml.get('top_symbols') or []}")
    print()
    print("System")
    print(f"  Memory:       {float(system.get('rss_memory_mb') or 0):.1f} MB")
    print(f"  Threads:      {int(system.get('thread_count') or 0)}")
    print(f"  SQLite size:  {float(system.get('sqlite_size_mb') or 0):.2f} MB")

    print("\nRecent fills")
    for f in payload.get("recent_fills", []):
        print(f"  {iso(f.get('ts_ms'))} {f.get('side')} {f.get('symbol')} qty={f.get('qty')} px={f.get('px')}")
    if not payload.get("recent_fills"):
        print("  none")

    print("\nRecent runtime events")
    for ev in payload.get("recent_runtime_events", []):
        print(f"  {iso(ev.get('ts_ms'))} {ev.get('severity')} {ev.get('event_type')}: {ev.get('message')}")
    if not payload.get("recent_runtime_events"):
        print("  none")

    print("\nRecent safety events")
    for ev in payload.get("recent_safety_events", []):
        print(f"  {iso(ev.get('ts_ms'))} {ev.get('severity')} {ev.get('event_type')} {ev.get('source') or ''}: {ev.get('message') or ''}")
    if not payload.get("recent_safety_events"):
        print("  none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print current Tradar runtime status from SQLite")
    parser.add_argument("--db-path", default="tradarbot.db")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tail-events", type=int, default=10)
    parser.add_argument("--tail-fills", type=int, default=10)
    args = parser.parse_args()
    payload = load_status(args.db_path, args.tail_events, args.tail_fills)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
