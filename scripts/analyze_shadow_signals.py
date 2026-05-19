#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


def parse_since(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = str(value).strip().lower()
    now = int(time.time() * 1000)
    if text.endswith("m"):
        return now - int(float(text[:-1]) * 60 * 1000)
    if text.endswith("h"):
        return now - int(float(text[:-1]) * 3600 * 1000)
    if text.endswith("d"):
        return now - int(float(text[:-1]) * 86400 * 1000)
    return int(float(text))


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def fetch_rows(conn: sqlite3.Connection, table: str, since_ts_ms: Optional[int], symbol: Optional[str]) -> List[Dict[str, Any]]:
    if not table_exists(conn, table):
        return []
    clauses = []
    params: List[Any] = []
    if since_ts_ms is not None:
        clauses.append("ts_ms >= ?")
        params.append(int(since_ts_ms))
    if symbol:
        clauses.append("symbol = ?")
        params.append(str(symbol))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    cur = conn.execute(f'SELECT * FROM "{table}"{where} ORDER BY ts_ms ASC', params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"count": 0}
    def pct(q: float) -> float:
        idx = int(round((len(vals) - 1) * q))
        return vals[max(0, min(idx, len(vals) - 1))]
    return {
        "count": len(vals),
        "min": vals[0],
        "p25": pct(0.25),
        "median": pct(0.50),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "max": vals[-1],
        "avg": sum(vals) / len(vals),
    }


def build_report(db_path: str, since_ts_ms: Optional[int] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    preds = fetch_rows(conn, "ml_shadow_predictions", since_ts_ms, symbol)
    decisions = fetch_rows(conn, "ml_shadow_decisions", since_ts_ms, symbol)
    signals = fetch_rows(conn, "ml_shadow_signals", since_ts_ms, symbol)

    by_symbol = Counter(str(r.get("symbol")) for r in signals)
    key_counts = Counter(
        (
            int(r.get("ts_ms") or 0),
            str(r.get("symbol")),
            str(r.get("side")),
            round(float(r.get("qty") or 0.0), 8),
            round(float(r.get("limit_px") or 0.0), 2),
        )
        for r in signals
    )
    duplicate_groups = {str(k): v for k, v in key_counts.items() if v > 1}
    duplicate_rows = sum(v - 1 for v in key_counts.values() if v > 1)

    intervals: Dict[str, List[float]] = defaultdict(list)
    last_ts: Dict[str, int] = {}
    for row in signals:
        sym = str(row.get("symbol"))
        ts = int(row.get("ts_ms") or 0)
        if sym in last_ts:
            intervals[sym].append(max(0.0, (ts - last_ts[sym]) / 1000.0))
        last_ts[sym] = ts

    if signals:
        first_ts = min(int(r.get("ts_ms") or 0) for r in signals)
        last = max(int(r.get("ts_ms") or 0) for r in signals)
        hours = max((last - first_ts) / 3600000.0, 1 / 3600.0)
    else:
        first_ts = last = None
        hours = 0.0

    raw_probs = []
    calibrated_probs = []
    for row in preds:
        calibrated_probs.append(row.get("pred_prob", row.get("prob")))
        try:
            payload = json.loads(row.get("payload_json") or "{}")
            raw_probs.append(payload.get("raw_pred_prob", payload.get("raw_prob")))
        except Exception:
            pass

    return {
        "db_path": db_path,
        "since_ts_ms": since_ts_ms,
        "symbol_filter": symbol,
        "window": {"first_signal_ts_ms": first_ts, "last_signal_ts_ms": last, "hours": hours},
        "totals": {
            "predictions": len(preds),
            "decisions": len(decisions),
            "signals": len(signals),
            "accepted_decisions": sum(1 for r in decisions if int(r.get("accepted") or 0) == 1),
            "would_trade_decisions": sum(1 for r in decisions if int(r.get("would_trade") or 0) == 1),
        },
        "rates": {
            "signals_per_hour": (len(signals) / hours) if hours else 0.0,
            "duplicate_signal_rows": duplicate_rows,
            "duplicate_signal_rate": (duplicate_rows / len(signals)) if signals else 0.0,
        },
        "symbols": dict(by_symbol),
        "average_signal_interval_seconds": {
            sym: (sum(vals) / len(vals) if vals else None) for sym, vals in intervals.items()
        },
        "probability_stats": {
            "calibrated_or_current": stats(calibrated_probs),
            "raw_from_payload": stats(raw_probs),
        },
        "top_repeated_signals": sorted(duplicate_groups.items(), key=lambda kv: kv[1], reverse=True)[:20],
    }


def print_human(report: Dict[str, Any]) -> None:
    print("TRADAR SHADOW SIGNAL ANALYSIS")
    print("=============================")
    print(f"DB:       {report['db_path']}")
    print(f"Since:    {report.get('since_ts_ms') or 'all'}")
    print(f"Symbol:   {report.get('symbol_filter') or 'all'}")
    t = report["totals"]
    r = report["rates"]
    print(f"Predictions: {t['predictions']}")
    print(f"Decisions:   {t['decisions']} accepted={t['accepted_decisions']} would_trade={t['would_trade_decisions']}")
    print(f"Signals:     {t['signals']}")
    print(f"Signals/hr:  {r['signals_per_hour']:.2f}")
    print(f"Duplicates:  {r['duplicate_signal_rows']} rate={r['duplicate_signal_rate']:.2%}")
    print("Symbols:")
    for sym, count in sorted(report["symbols"].items()):
        avg = report["average_signal_interval_seconds"].get(sym)
        print(f"  {sym}: {count} avg_interval_s={avg}")
    print("Probability stats:")
    for name, payload in report["probability_stats"].items():
        print(f"  {name}: {payload}")
    if report["top_repeated_signals"]:
        print("Top repeated signals:")
        for key, count in report["top_repeated_signals"][:10]:
            print(f"  {count}x {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Tradar Phase 5.8/5.8.5 shadow signal quality")
    parser.add_argument("--db-path", default="tradarbot.db")
    parser.add_argument("--since", default=None, help="timestamp ms or relative window like 1h, 24h, 7d")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.db_path, parse_since(args.since), args.symbol)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
