import sqlite3
from tradarbot.core.events import CandleEvent

class SQLiteStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

    def init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS candles (
          symbol TEXT NOT NULL,
          interval_s INTEGER NOT NULL,
          ts_ms INTEGER NOT NULL,
          open REAL NOT NULL,
          high REAL NOT NULL,
          low REAL NOT NULL,
          close REAL NOT NULL,
          volume REAL NOT NULL,
          PRIMARY KEY(symbol, interval_s, ts_ms)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL DEFAULT (strftime('%s','now')*1000),
          symbol TEXT NOT NULL,
          side TEXT NOT NULL,
          qty REAL NOT NULL,
          px REAL NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_curve (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          cash REAL NOT NULL,
          realized_pnl REAL NOT NULL,
          unrealized_pnl REAL NOT NULL,
          equity REAL NOT NULL,
          mode TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS book (
          symbol TEXT PRIMARY KEY,
          ts_ms INTEGER NOT NULL,
          bid REAL NOT NULL,
          ask REAL NOT NULL
        );
        """)
        self.conn.commit()

    def insert_candle(self, ev: CandleEvent) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO candles(symbol, interval_s, ts_ms, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
            (ev.symbol, ev.interval_s, ev.ts_ms, ev.open, ev.high, ev.low, ev.close, ev.volume),
        )
        self.conn.commit()

    def insert_fill(self, ts_ms: int, symbol: str, side: str, qty: float, px: float) -> None:
        self.conn.execute(
            "INSERT INTO fills(ts_ms, symbol, side, qty, px) VALUES (?,?,?,?,?)",
            (ts_ms, symbol, side, qty, px),
        )
        self.conn.commit()

    def upsert_book(self, symbol: str, ts_ms: int, bid: float, ask: float) -> None:
        self.conn.execute(
            "INSERT INTO book(symbol, ts_ms, bid, ask) VALUES (?,?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET ts_ms=excluded.ts_ms, bid=excluded.bid, ask=excluded.ask",
            (symbol, ts_ms, bid, ask),
        )
        self.conn.commit()

    def insert_equity_snapshot(
        self,
        ts_ms: int,
        cash: float,
        realized_pnl: float,
        unrealized_pnl: float,
        equity: float,
        mode: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO equity_curve(ts_ms, cash, realized_pnl, unrealized_pnl, equity, mode)
            VALUES (?,?,?,?,?,?)
            """,
            (ts_ms, cash, realized_pnl, unrealized_pnl, equity, mode),
        )
        self.conn.commit()
