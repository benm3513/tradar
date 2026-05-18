#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.ml.shadow_eval import build_shadow_summary, compare_paper_live_execution, compare_shadow_to_fills
from tradarbot.ml.parity_checks import build_recent_window_parity_report


def parse_since(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = str(value).strip().lower()
    now = int(time.time() * 1000)
    if text.endswith("m"):
        return now - int(float(text[:-1]) * 60_000)
    if text.endswith("h"):
        return now - int(float(text[:-1]) * 3_600_000)
    if text.endswith("d"):
        return now - int(float(text[:-1]) * 86_400_000)
    return int(float(text))


def parse_symbols(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [s.strip() for s in str(value).split(",") if s.strip()]


def load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def write_output(path: str, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        rows = payload.get("recent_shadow_signals") or payload.get("recent_shadow_predictions") or []
        if not rows:
            rows = [payload.get("summary", payload)]
        fieldnames = sorted({k for row in rows for k in dict(row).keys()})
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list)) else v for k, v in dict(row).items()})
    else:
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def print_human(payload: Dict[str, Any]) -> None:
    cfg = payload.get("config", {}) or {}
    summary = payload.get("summary", {}) or {}
    print("TRADAR ML SHADOW REPORT")
    print("=======================")
    print(f"ml_live.mode: {cfg.get('ml_live_mode', 'unknown')} enabled={cfg.get('ml_live_enabled', 'unknown')}")
    print(f"Since:        {payload.get('since_ts_ms') or 'all'}")
    print(f"Predictions:  {summary.get('prediction_count', 0)}")
    print(f"Decisions:    {summary.get('decision_count', 0)} accepted={summary.get('accepted_count', 0)} rejected={summary.get('rejected_count', 0)}")
    print(f"Signals:      {summary.get('signal_count', 0)} would_trade={summary.get('would_trade_count', 0)}")
    print(f"Symbols:      {', '.join(summary.get('symbols', [])[:20]) or 'n/a'}")
    if payload.get("execution_comparison"):
        comp = payload["execution_comparison"]
        print()
        print(f"Execution comparison: {comp.get('status')} count={comp.get('count', len(comp.get('comparisons', [])))}")
        if comp.get("message"):
            print(f"  {comp.get('message')}")
    if payload.get("paper_live_comparison"):
        comp = payload["paper_live_comparison"]
        print()
        print(f"Paper/live comparison: {comp.get('status')} paper_fills={comp.get('paper_fills', 0)} live_fills={comp.get('live_fills', 0)}")
        if comp.get("message"):
            print(f"  {comp.get('message')}")
    if payload.get("parity_report"):
        pr = payload["parity_report"]
        print()
        print(f"Parity: {pr.get('status')} predictions={pr.get('metrics', {}).get('shadow_predictions', 0)} signals={pr.get('metrics', {}).get('shadow_signals', 0)}")
        for warning in pr.get("warnings", [])[:5]:
            print(f"  WARN: {warning}")
        for failure in pr.get("failures", [])[:5]:
            print(f"  FAIL: {failure}")
    recent = payload.get("recent_shadow_signals") or []
    if recent:
        print()
        print("Recent shadow signals")
        for row in recent[:10]:
            print(f"  {row.get('ts_ms')} {row.get('symbol')} {row.get('side')} qty={row.get('qty')} px={row.get('limit_px')} prob={row.get('prob')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Tradar ML shadow-mode evaluation from SQLite")
    parser.add_argument("--db-path", default="tradarbot.db")
    parser.add_argument("--config", default="config/tradar.yaml")
    parser.add_argument("--since", default=None, help="ts_ms or relative window like 30m, 6h, 2d")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--compare-execution", action="store_true")
    parser.add_argument("--parity-check", action="store_true")
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    since_ts_ms = parse_since(args.since)
    symbols = parse_symbols(args.symbols)
    cfg = load_config(args.config)
    ml_live = cfg.get("ml_live", {}) if isinstance(cfg, dict) else {}

    store = SQLiteStore(args.db_path)
    store.init_schema()

    payload: Dict[str, Any] = {
        "db_path": args.db_path,
        "since_ts_ms": since_ts_ms,
        "symbols": symbols,
        "config": {"ml_live_enabled": ml_live.get("enabled"), "ml_live_mode": ml_live.get("mode")},
        "summary": build_shadow_summary(store, since_ts_ms=since_ts_ms),
        "recent_shadow_predictions": store.recent_ml_shadow_predictions(limit=args.tail, since_ts_ms=since_ts_ms),
        "recent_shadow_signals": store.recent_ml_shadow_signals(limit=args.tail, since_ts_ms=since_ts_ms),
    }
    if symbols:
        allowed = set(symbols)
        payload["recent_shadow_predictions"] = [r for r in payload["recent_shadow_predictions"] if str(r.get("symbol")) in allowed]
        payload["recent_shadow_signals"] = [r for r in payload["recent_shadow_signals"] if str(r.get("symbol")) in allowed]

    if args.compare_execution:
        payload["execution_comparison"] = compare_shadow_to_fills(store, since_ts_ms=since_ts_ms)
        payload["paper_live_comparison"] = compare_paper_live_execution(store, since_ts_ms=since_ts_ms)
    if args.parity_check:
        payload["parity_report"] = build_recent_window_parity_report(store, symbols=symbols, since_ts_ms=since_ts_ms, write=True)

    if args.output:
        write_output(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
