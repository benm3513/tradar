import json
import sqlite3
import time
from typing import Any, Dict, Optional

from tradarbot.core.events import CandleEvent
from tradarbot.portfolio.positions import LivePositionState, PortfolioSnapshot, PositionOwner


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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS position_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          venue_symbol TEXT,
          qty REAL NOT NULL,
          avg_px REAL NOT NULL,
          current_price REAL,
          unrealized_pnl REAL,
          realized_pnl REAL,
          peak_price REAL,
          trailing_stop_price REAL,
          partial_exit_taken INTEGER,
          strategy_name TEXT,
          signal_id TEXT,
          model_name TEXT,
          prediction_source TEXT,
          metadata_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          cash REAL NOT NULL,
          equity REAL NOT NULL,
          realized_pnl REAL NOT NULL,
          unrealized_pnl REAL NOT NULL,
          total_exposure REAL NOT NULL,
          broker_mode TEXT,
          metadata_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS position_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          event_type TEXT NOT NULL,
          qty REAL,
          px REAL,
          reason TEXT,
          strategy_name TEXT,
          client_order_id TEXT,
          exchange_order_id TEXT,
          metadata_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reconciliation_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          broker_mode TEXT,
          provider TEXT,
          status TEXT NOT NULL,
          result_json TEXT,
          warnings_json TEXT,
          errors_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS safety_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          event_type TEXT NOT NULL,
          severity TEXT NOT NULL,
          source TEXT,
          symbol TEXT,
          message TEXT,
          details_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runtime_health (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          status TEXT NOT NULL,
          safe_mode INTEGER NOT NULL,
          kill_switch INTEGER NOT NULL,
          stale_symbols TEXT,
          api_errors INTEGER DEFAULT 0,
          ws_disconnects INTEGER DEFAULT 0,
          prediction_errors INTEGER DEFAULT 0,
          details_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS safety_state_snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          safe_mode INTEGER NOT NULL,
          kill_switch INTEGER NOT NULL,
          reasons_json TEXT,
          metadata_json TEXT
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
        self.conn.execute("INSERT INTO fills(ts_ms, symbol, side, qty, px) VALUES (?,?,?,?,?)", (ts_ms, symbol, side, qty, px))
        self.conn.commit()

    def upsert_book(self, symbol: str, ts_ms: int, bid: float, ask: float) -> None:
        self.conn.execute(
            "INSERT INTO book(symbol, ts_ms, bid, ask) VALUES (?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET ts_ms=excluded.ts_ms, bid=excluded.bid, ask=excluded.ask",
            (symbol, ts_ms, bid, ask),
        )
        self.conn.commit()

    def insert_equity_snapshot(self, ts_ms: int, cash: float, realized_pnl: float, unrealized_pnl: float, equity: float, mode: str) -> None:
        self.conn.execute(
            "INSERT INTO equity_curve(ts_ms, cash, realized_pnl, unrealized_pnl, equity, mode) VALUES (?,?,?,?,?,?)",
            (ts_ms, cash, realized_pnl, unrealized_pnl, equity, mode),
        )
        self.conn.commit()

    def insert_order(self, *, client_order_id: Optional[str], exchange_order_id: Optional[str], symbol: str, side: str, order_type: str, tif: Optional[str], qty: float, limit_px: Optional[float], status: str, broker_mode: str, strategy_name: Optional[str], submitted_ts_ms: int, updated_ts_ms: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO orders(client_order_id, exchange_order_id, symbol, side, order_type, tif, qty, limit_px, status, broker_mode, strategy_name, submitted_ts_ms, updated_ts_ms, metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(client_order_id) DO UPDATE SET exchange_order_id=excluded.exchange_order_id, status=excluded.status, updated_ts_ms=excluded.updated_ts_ms, metadata_json=excluded.metadata_json
            """,
            (client_order_id, exchange_order_id, symbol, side, order_type, tif, qty, limit_px, status, broker_mode, strategy_name, submitted_ts_ms, updated_ts_ms, self._json(metadata)),
        )
        self.conn.commit()

    def update_order_status(self, *, client_order_id: Optional[str], status: str, updated_ts_ms: int, exchange_order_id: Optional[str] = None, error_code: Optional[str] = None, error_message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not client_order_id:
            return
        self.conn.execute(
            """
            UPDATE orders
               SET status=?, updated_ts_ms=?, exchange_order_id=COALESCE(?, exchange_order_id),
                   error_code=COALESCE(?, error_code), error_message=COALESCE(?, error_message), metadata_json=COALESCE(?, metadata_json)
             WHERE client_order_id=?
            """,
            (status, updated_ts_ms, exchange_order_id, error_code, error_message, self._json(metadata), client_order_id),
        )
        self.conn.commit()

    def insert_order_fill(self, *, client_order_id: Optional[str], exchange_order_id: Optional[str], symbol: str, side: str, qty: float, px: float, fee: float, fee_asset: Optional[str], ts_ms: int, trade_id: Optional[str], is_maker: Optional[bool], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            "INSERT INTO order_fills(client_order_id, exchange_order_id, trade_id, symbol, side, qty, px, fee, fee_asset, is_maker, ts_ms, metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (client_order_id, exchange_order_id, trade_id, symbol, side, qty, px, fee, fee_asset, self._bool_int(is_maker), ts_ms, self._json(metadata)),
        )
        self.conn.commit()

    def insert_execution_event(self, *, ts_ms: int, symbol: str, event_type: str, broker_mode: str, client_order_id: Optional[str] = None, exchange_order_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            "INSERT INTO execution_events(ts_ms, symbol, event_type, broker_mode, client_order_id, exchange_order_id, details_json) VALUES (?,?,?,?,?,?,?)",
            (ts_ms, symbol, event_type, broker_mode, client_order_id, exchange_order_id, self._json(details)),
        )
        self.conn.commit()

    def insert_position_snapshot(self, position=None, ts_ms: Optional[int] = None, **kwargs) -> None:
        if position is not None and isinstance(position, LivePositionState):
            owner = position.owner or PositionOwner()
            ts = int(ts_ms if ts_ms is not None else (position.last_update_ts_ms or 0))
            self.conn.execute(
                """
                INSERT INTO position_snapshots(ts_ms, symbol, venue_symbol, qty, avg_px, current_price, unrealized_pnl, realized_pnl, peak_price, trailing_stop_price, partial_exit_taken, strategy_name, signal_id, model_name, prediction_source, metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (ts, position.symbol, position.venue_symbol, position.qty, position.avg_px, position.current_price, position.unrealized_pnl, position.realized_pnl, position.peak_price, position.trailing_stop_price, self._bool_int(position.partial_exit_taken), owner.strategy_name, owner.signal_id, owner.model_name, owner.prediction_source, self._json(position.metadata)),
            )
            self.conn.commit()
            return
        # Backward-compatible legacy positions_live path.
        self.conn.execute(
            "INSERT INTO positions_live(ts_ms, symbol, qty, avg_px, mark_px, unrealized_pnl, broker_mode, details_json) VALUES (?,?,?,?,?,?,?,?)",
            (kwargs.get("ts_ms", ts_ms), kwargs["symbol"], kwargs["qty"], kwargs["avg_px"], kwargs.get("mark_px"), kwargs.get("unrealized_pnl"), kwargs.get("broker_mode", "unknown"), self._json(kwargs.get("details"))),
        )
        self.conn.commit()

    def insert_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self.conn.execute(
            "INSERT INTO portfolio_snapshots(ts_ms, cash, equity, realized_pnl, unrealized_pnl, total_exposure, broker_mode, metadata_json) VALUES (?,?,?,?,?,?,?,?)",
            (snapshot.ts_ms, snapshot.cash, snapshot.equity, snapshot.realized_pnl, snapshot.unrealized_pnl, snapshot.total_exposure, snapshot.broker_mode, self._json(snapshot.metadata)),
        )
        for pos in dict(snapshot.positions or {}).values():
            self.insert_position_snapshot(pos, ts_ms=snapshot.ts_ms)
        self.conn.commit()

    def insert_position_event(self, *, ts_ms: int, symbol: str, event_type: str, qty: Optional[float] = None, px: Optional[float] = None, reason: Optional[str] = None, strategy_name: Optional[str] = None, client_order_id: Optional[str] = None, exchange_order_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            "INSERT INTO position_events(ts_ms, symbol, event_type, qty, px, reason, strategy_name, client_order_id, exchange_order_id, metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts_ms, symbol, event_type, qty, px, reason, strategy_name, client_order_id, exchange_order_id, self._json(metadata)),
        )
        self.conn.commit()

    def insert_reconciliation_run(self, result) -> None:
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
        self.conn.execute(
            "INSERT INTO reconciliation_runs(ts_ms, broker_mode, provider, status, result_json, warnings_json, errors_json) VALUES (?,?,?,?,?,?,?)",
            (data.get("ts_ms"), data.get("broker_mode"), data.get("provider"), data.get("status", "unknown"), self._json(data), self._json(data.get("warnings", [])), self._json(data.get("errors", []))),
        )
        self.conn.commit()

    def load_latest_position_snapshots(self) -> Dict[str, LivePositionState]:
        rows = self.conn.execute(
            """
            SELECT ps.* FROM position_snapshots ps
            JOIN (SELECT symbol, MAX(id) AS max_id FROM position_snapshots GROUP BY symbol) latest
              ON ps.symbol = latest.symbol AND ps.id = latest.max_id
            WHERE ps.qty > 0
            """
        ).fetchall()
        out: Dict[str, LivePositionState] = {}
        for r in rows:
            d = dict(zip([c[0] for c in self.conn.execute("SELECT * FROM position_snapshots LIMIT 0").description], r))
            meta = self._loads(d.get("metadata_json")) or {}
            owner = PositionOwner(strategy_name=d.get("strategy_name"), signal_id=d.get("signal_id"), model_name=d.get("model_name"), prediction_source=d.get("prediction_source"))
            out[d["symbol"]] = LivePositionState(
                symbol=d["symbol"], venue_symbol=d.get("venue_symbol"), qty=float(d.get("qty") or 0.0), avg_px=float(d.get("avg_px") or 0.0), current_price=d.get("current_price"), unrealized_pnl=float(d.get("unrealized_pnl") or 0.0), realized_pnl=float(d.get("realized_pnl") or 0.0), peak_price=d.get("peak_price"), trailing_stop_price=d.get("trailing_stop_price"), partial_exit_taken=bool(d.get("partial_exit_taken")), owner=owner, metadata=meta, last_update_ts_ms=d.get("ts_ms")
            )
        return out

    def load_open_orders(self):
        cur = self.conn.execute("SELECT client_order_id, exchange_order_id, symbol, side, order_type, tif, qty, limit_px, status, broker_mode, strategy_name, submitted_ts_ms, updated_ts_ms, metadata_json FROM orders WHERE status NOT IN ('FILLED','CANCELED','REJECTED','EXPIRED','ERROR')")
        cols = [c[0] for c in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["metadata"] = self._loads(d.pop("metadata_json", None))
            rows.append(d)
        return rows

    def load_recent_position_events(self, limit: int = 100):
        cur = self.conn.execute("SELECT * FROM position_events ORDER BY id DESC LIMIT ?", (int(limit),))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    @staticmethod
    def _json(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _loads(value: Optional[str]) -> Any:
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    @staticmethod
    def _bool_int(value: Optional[bool]) -> Optional[int]:
        if value is None:
            return None
        return int(bool(value))


    def insert_safety_event(self, event_type: str, severity: str, source: str = None, symbol: str = None, message: str = None, details: Optional[Dict[str, Any]] = None, ts_ms: Optional[int] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO safety_events (ts_ms, event_type, severity, source, symbol, message, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(ts_ms or time.time() * 1000),
                str(event_type),
                str(severity),
                source,
                symbol,
                message,
                json.dumps(details or {}, sort_keys=True),
            ),
        )
        self.conn.commit()

    def insert_runtime_health(self, status: str, safe_mode: bool, kill_switch: bool, stale_symbols=None, api_errors: int = 0, ws_disconnects: int = 0, prediction_errors: int = 0, details: Optional[Dict[str, Any]] = None, ts_ms: Optional[int] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO runtime_health (ts_ms, status, safe_mode, kill_switch, stale_symbols, api_errors, ws_disconnects, prediction_errors, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(ts_ms or time.time() * 1000),
                str(status),
                1 if safe_mode else 0,
                1 if kill_switch else 0,
                json.dumps(list(stale_symbols or []), sort_keys=True),
                int(api_errors or 0),
                int(ws_disconnects or 0),
                int(prediction_errors or 0),
                json.dumps(details or {}, sort_keys=True),
            ),
        )
        self.conn.commit()

    def insert_safety_snapshot(self, safe_mode: bool, kill_switch: bool, reasons=None, metadata: Optional[Dict[str, Any]] = None, ts_ms: Optional[int] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO safety_state_snapshots (ts_ms, safe_mode, kill_switch, reasons_json, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(ts_ms or time.time() * 1000),
                1 if safe_mode else 0,
                1 if kill_switch else 0,
                json.dumps(list(reasons or []), sort_keys=True),
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        self.conn.commit()

    def latest_safety_snapshot(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT ts_ms, safe_mode, kill_switch, reasons_json, metadata_json FROM safety_state_snapshots ORDER BY ts_ms DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "ts_ms": row[0],
            "safe_mode": bool(row[1]),
            "kill_switch": bool(row[2]),
            "reasons": json.loads(row[3] or "[]"),
            "metadata": json.loads(row[4] or "{}"),
        }
