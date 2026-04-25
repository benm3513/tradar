#!/usr/bin/env python3
"""
Phase 5.2 Alpaca paper metrics report.

Run from repo root after a controlled paper run:

    PYTHONPATH=. ./.venv/bin/python scripts/phase5_2_live_paper_report.py --db tradarbot.db

Optional:

    PYTHONPATH=. ./.venv/bin/python scripts/phase5_2_live_paper_report.py \
      --db tradarbot.db \
      --since-minutes 60
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def scalar(conn: sqlite3.Connection, query: str, params: Iterable[Any] = ()) -> Any:
    row = conn.execute(query, tuple(params)).fetchone()
    return None if row is None else row[0]


def rows(conn: sqlite3.Connection, query: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(query, tuple(params)).fetchall())


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_kv(key: str, value: Any) -> None:
    print(f"{key:36s} {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="tradarbot.db")
    parser.add_argument("--since-minutes", type=float, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    now_ms = int(time.time() * 1000)
    since_ms: Optional[int] = None
    if args.since_minutes is not None:
        since_ms = now_ms - int(args.since_minutes * 60 * 1000)

    def where_ts(alias: str = "ts_ms") -> tuple[str, list[Any]]:
        if since_ms is None:
            return "", []
        return f" WHERE {alias} >= ? ", [since_ms]

    report: dict[str, Any] = {}

    print_section("PHASE 5.2 LIVE/PAPER DATA HEALTH")

    if table_exists(conn, "candles"):
        where, params = where_ts("ts_ms")
        candle_count = scalar(conn, f"SELECT COUNT(*) FROM candles {where}", params)
        symbol_count = scalar(conn, f"SELECT COUNT(DISTINCT symbol) FROM candles {where}", params)
        latest_candle = scalar(conn, f"SELECT MAX(ts_ms) FROM candles {where}", params)
        report["candles"] = {
            "count": candle_count,
            "symbols": symbol_count,
            "latest_ts_ms": latest_candle,
        }
        print_kv("candles", candle_count)
        print_kv("candle_symbols", symbol_count)
        print_kv("latest_candle_ts_ms", latest_candle)

        recent = rows(
            conn,
            f"""
            SELECT symbol, COUNT(*) AS n, MIN(ts_ms) AS first_ts_ms, MAX(ts_ms) AS last_ts_ms
            FROM candles
            {where}
            GROUP BY symbol
            ORDER BY n DESC, symbol
            LIMIT 20
            """,
            params,
        )
        print("\nTop candle symbols:")
        for r in recent:
            print(f"  {r['symbol']:12s} n={r['n']:6d} first={r['first_ts_ms']} last={r['last_ts_ms']}")
    else:
        print_kv("candles", "missing table")

    if table_exists(conn, "book"):
        book_count = scalar(conn, "SELECT COUNT(*) FROM book")
        latest_book = scalar(conn, "SELECT MAX(ts_ms) FROM book")
        report["book"] = {"rows": book_count, "latest_ts_ms": latest_book}
        print_kv("book_rows", book_count)
        print_kv("latest_book_ts_ms", latest_book)

    print_section("EXECUTION / ALPACA PAPER ORDER HEALTH")

    if table_exists(conn, "orders"):
        where, params = where_ts("submitted_ts_ms")
        order_count = scalar(conn, f"SELECT COUNT(*) FROM orders {where}", params)
        report["orders_total"] = order_count
        print_kv("orders_total", order_count)

        by_status = rows(
            conn,
            f"""
            SELECT status, COUNT(*) AS n
            FROM orders
            {where}
            GROUP BY status
            ORDER BY n DESC
            """,
            params,
        )
        print("\nOrders by status:")
        report["orders_by_status"] = {r["status"]: r["n"] for r in by_status}
        for r in by_status:
            print(f"  {r['status']:18s} {r['n']}")

        rejected = rows(
            conn,
            f"""
            SELECT submitted_ts_ms, symbol, side, qty, limit_px, status, error_code, error_message
            FROM orders
            {where}
            AND status IN ('REJECTED','ERROR')
            ORDER BY submitted_ts_ms DESC
            LIMIT 10
            """ if where else """
            SELECT submitted_ts_ms, symbol, side, qty, limit_px, status, error_code, error_message
            FROM orders
            WHERE status IN ('REJECTED','ERROR')
            ORDER BY submitted_ts_ms DESC
            LIMIT 10
            """,
            params,
        )
        if rejected:
            print("\nRecent rejected/error orders:")
            for r in rejected:
                print(
                    f"  {r['submitted_ts_ms']} {r['symbol']} {r['side']} "
                    f"qty={r['qty']} px={r['limit_px']} status={r['status']} "
                    f"code={r['error_code']} msg={r['error_message']}"
                )
        else:
            print("\nRecent rejected/error orders: none")
    else:
        print_kv("orders", "missing table")

    if table_exists(conn, "order_fills"):
        where, params = where_ts("ts_ms")
        fill_count = scalar(conn, f"SELECT COUNT(*) FROM order_fills {where}", params)
        buy_qty = scalar(conn, f"SELECT COALESCE(SUM(qty),0) FROM order_fills {where} {'AND' if where else 'WHERE'} UPPER(side)='BUY'", params)
        sell_qty = scalar(conn, f"SELECT COALESCE(SUM(qty),0) FROM order_fills {where} {'AND' if where else 'WHERE'} UPPER(side)='SELL'", params)
        report["fills"] = {"count": fill_count, "buy_qty": buy_qty, "sell_qty": sell_qty}
        print_kv("fills_total", fill_count)
        print_kv("buy_qty", buy_qty)
        print_kv("sell_qty", sell_qty)

        recent_fills = rows(
            conn,
            f"""
            SELECT ts_ms, symbol, side, qty, px, fee, fee_asset
            FROM order_fills
            {where}
            ORDER BY ts_ms DESC
            LIMIT 10
            """,
            params,
        )
        print("\nRecent fills:")
        if not recent_fills:
            print("  none")
        for r in recent_fills:
            print(f"  {r['ts_ms']} {r['symbol']} {r['side']} qty={r['qty']} px={r['px']} fee={r['fee']} {r['fee_asset']}")
    else:
        print_kv("order_fills", "missing table")

    if table_exists(conn, "execution_events"):
        where, params = where_ts("ts_ms")
        event_count = scalar(conn, f"SELECT COUNT(*) FROM execution_events {where}", params)
        report["execution_events_total"] = event_count
        print_kv("execution_events_total", event_count)

        by_type = rows(
            conn,
            f"""
            SELECT event_type, COUNT(*) AS n
            FROM execution_events
            {where}
            GROUP BY event_type
            ORDER BY n DESC
            """,
            params,
        )
        print("\nExecution events by type:")
        for r in by_type:
            print(f"  {r['event_type']:24s} {r['n']}")

    print_section("PASS / FAIL CHECKS")
    checks = []

    candles_ok = table_exists(conn, "candles") and (report.get("candles", {}).get("count") or 0) > 0
    checks.append(("candles_written", candles_ok))

    symbols_ok = (report.get("candles", {}).get("symbols") or 0) >= 1
    checks.append(("active_symbols_have_candles", symbols_ok))

    no_order_errors = True
    if table_exists(conn, "orders"):
        where, params = where_ts("submitted_ts_ms")
        err_count = scalar(
            conn,
            f"SELECT COUNT(*) FROM orders {where} {'AND' if where else 'WHERE'} status IN ('REJECTED','ERROR')",
            params,
        )
        no_order_errors = int(err_count or 0) == 0
    checks.append(("no_rejected_or_error_orders", no_order_errors))

    for name, passed in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {name}")
    report["checks"] = {name: bool(passed) for name, passed in checks}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote JSON report -> {args.json_out}")

    return 0 if all(p for _, p in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
