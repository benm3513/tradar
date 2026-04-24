import json
import sqlite3
from typing import Any, Dict, Optional

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
          ts_ms INTEGER NOT NULL,
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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          client_order_id TEXT UNIQUE,
          exchange_order_id TEXT,
          symbol TEXT NOT NULL,
          side TEXT NOT NULL,
          order_type TEXT NOT NULL,
          tif TEXT,
          qty REAL NOT NULL,
          limit_px REAL,
          status TEXT NOT NULL,
          broker_mode TEXT NOT NULL,
          strategy_name TEXT,
          submitted_ts_ms INTEGER NOT NULL,
          updated_ts_ms INTEGER NOT NULL,
          error_code TEXT,
          error_message TEXT,
          metadata_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS order_fills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          client_order_id TEXT,
          exchange_order_id TEXT,
          trade_id TEXT,
          symbol TEXT NOT NULL,
          side TEXT NOT NULL,
          qty REAL NOT NULL,
          px REAL NOT NULL,
          fee REAL DEFAULT 0.0,
          fee_asset TEXT,
          is_maker INTEGER,
          ts_ms INTEGER NOT NULL,
          metadata_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS execution_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          event_type TEXT NOT NULL,
          broker_mode TEXT NOT NULL,
          client_order_id TEXT,
          exchange_order_id TEXT,
          details_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions_live (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          qty REAL NOT NULL,
          avg_px REAL NOT NULL,
          mark_px REAL,
          unrealized_pnl REAL,
          broker_mode TEXT NOT NULL,
          details_json TEXT
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

    def insert_order(
        self,
        *,
        client_order_id: Optional[str],
        exchange_order_id: Optional[str],
        symbol: str,
        side: str,
        order_type: str,
        tif: Optional[str],
        qty: float,
        limit_px: Optional[float],
        status: str,
        broker_mode: str,
        strategy_name: Optional[str],
        submitted_ts_ms: int,
        updated_ts_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO orders(
              client_order_id, exchange_order_id, symbol, side, order_type, tif, qty, limit_px, status,
              broker_mode, strategy_name, submitted_ts_ms, updated_ts_ms, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(client_order_id) DO UPDATE SET
              exchange_order_id=excluded.exchange_order_id,
              status=excluded.status,
              updated_ts_ms=excluded.updated_ts_ms,
              metadata_json=excluded.metadata_json
            """,
            (
                client_order_id, exchange_order_id, symbol, side, order_type, tif, qty, limit_px, status,
                broker_mode, strategy_name, submitted_ts_ms, updated_ts_ms, self._json(metadata),
            ),
        )
        self.conn.commit()

    def update_order_status(
        self,
        *,
        client_order_id: Optional[str],
        status: str,
        updated_ts_ms: int,
        exchange_order_id: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not client_order_id:
            return
        self.conn.execute(
            """
            UPDATE orders
               SET status=?, updated_ts_ms=?, exchange_order_id=COALESCE(?, exchange_order_id),
                   error_code=COALESCE(?, error_code), error_message=COALESCE(?, error_message),
                   metadata_json=COALESCE(?, metadata_json)
             WHERE client_order_id=?
            """,
            (status, updated_ts_ms, exchange_order_id, error_code, error_message, self._json(metadata), client_order_id),
        )
        self.conn.commit()

    def insert_order_fill(
        self,
        *,
        client_order_id: Optional[str],
        exchange_order_id: Optional[str],
        symbol: str,
        side: str,
        qty: float,
        px: float,
        fee: float,
        fee_asset: Optional[str],
        ts_ms: int,
        trade_id: Optional[str],
        is_maker: Optional[bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO order_fills(
              client_order_id, exchange_order_id, trade_id, symbol, side, qty, px, fee, fee_asset, is_maker, ts_ms, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (client_order_id, exchange_order_id, trade_id, symbol, side, qty, px, fee, fee_asset, self._bool_int(is_maker), ts_ms, self._json(metadata)),
        )
        self.conn.commit()

    def insert_execution_event(
        self,
        *,
        ts_ms: int,
        symbol: str,
        event_type: str,
        broker_mode: str,
        client_order_id: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO execution_events(ts_ms, symbol, event_type, broker_mode, client_order_id, exchange_order_id, details_json)
            VALUES (?,?,?,?,?,?,?)
            """,
            (ts_ms, symbol, event_type, broker_mode, client_order_id, exchange_order_id, self._json(details)),
        )
        self.conn.commit()

    def insert_position_snapshot(
        self,
        *,
        ts_ms: int,
        symbol: str,
        qty: float,
        avg_px: float,
        broker_mode: str,
        mark_px: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO positions_live(ts_ms, symbol, qty, avg_px, mark_px, unrealized_pnl, broker_mode, details_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (ts_ms, symbol, qty, avg_px, mark_px, unrealized_pnl, broker_mode, self._json(details)),
        )
        self.conn.commit()

    @staticmethod
    def _json(value: Optional[Dict[str, Any]]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _bool_int(value: Optional[bool]) -> Optional[int]:
        if value is None:
            return None
        return int(bool(value))
