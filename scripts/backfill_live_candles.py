#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Iterable


INTERVAL_TO_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def ensure_candles_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            interval_s INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, ts_ms, interval_s)
        )
        """
    )
    conn.commit()


def fetch_klines(
    base_url: str,
    symbol: str,
    interval: str,
    limit: int,
) -> list:
    qs = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
    )
    url = f"{base_url.rstrip('/')}/v3/klines?{qs}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


def insert_klines(
    conn: sqlite3.Connection,
    symbol: str,
    klines: Iterable,
    interval_s: int,
) -> int:
    rows = []
    for k in klines:
        # Binance kline format:
        # 0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close_time, ...
        ts_ms = int(k[0])
        rows.append(
            (
                symbol,
                ts_ms,
                interval_s,
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO candles
        (symbol, ts_ms, interval_s, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default="tradarbot.db")
    p.add_argument("--base-url", default="https://api.binance.us/api")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--limit", type=int, default=168)
    p.add_argument("--sleep-s", type=float, default=0.25)
    args = p.parse_args()

    if args.interval not in INTERVAL_TO_MS:
        raise SystemExit(f"Unsupported interval: {args.interval}")

    interval_s = INTERVAL_TO_MS[args.interval] // 1000

    conn = sqlite3.connect(args.db_path)
    ensure_candles_table(conn)

    for symbol in args.symbols:
        symbol = symbol.upper().strip()
        print(f"BACKFILL start symbol={symbol} interval={args.interval} limit={args.limit}")
        klines = fetch_klines(args.base_url, symbol, args.interval, args.limit)
        inserted = insert_klines(conn, symbol, klines, interval_s)
        print(f"BACKFILL done symbol={symbol} rows={inserted}")
        time.sleep(args.sleep_s)

    print("BACKFILL complete")


if __name__ == "__main__":
    main()
