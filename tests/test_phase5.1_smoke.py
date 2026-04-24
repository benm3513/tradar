#!/usr/bin/env python3
from __future__ import annotations

"""Phase 5.1 smoke test for Tradar live-broker execution layer.

What this checks:
1) Phase 5.1 modules import cleanly
2) SQLiteStore initializes new live execution tables
3) OrderRouter rounds/routs an OrderIntent correctly
4) LiveBroker can submit a BUY against a mocked Binance client
5) Fill reconciliation writes orders/fills/execution events to SQLite
6) close_all(...) can flatten the mocked live position with a SELL

Run:
    PYTHONPATH=. ./.venv/bin/python tests/test_phase5.1_smoke.py

Optional:
    PYTHONPATH=. ./.venv/bin/python tests/test_phase5.1_smoke.py --config config/tradar.yaml
"""

import argparse
import os
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


class FakeClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.orders: Dict[str, Dict[str, Any]] = {}
        self._next_order_id = 1000

    def get_account(self) -> Dict[str, Any]:
        return {"balances": [{"asset": "USDT", "free": "10000.00", "locked": "0"}]}

    def get_exchange_info(self) -> Dict[str, Any]:
        return {"symbols": []}

    def get_open_orders(self, symbol=None):
        rows = [o for o in self.orders.values() if o["status"] not in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}]
        if symbol:
            rows = [o for o in rows if o["symbol"] == symbol]
        return rows

    def place_limit_order(self, symbol, side, quantity, price, tif="IOC", client_order_id=None):
        self._next_order_id += 1
        oid = str(self._next_order_id)
        qty = float(quantity)
        px = float(price)
        payload = {
            "symbol": symbol,
            "side": str(side).upper(),
            "status": "FILLED",
            "type": "LIMIT",
            "timeInForce": tif,
            "orderId": oid,
            "clientOrderId": client_order_id,
            "origQty": str(qty),
            "executedQty": str(qty),
            "cummulativeQuoteQty": str(qty * px),
            "price": str(px),
            "transactTime": 1713900000000,
            "updateTime": 1713900000000,
            "fills": [
                {
                    "price": str(px),
                    "qty": str(qty),
                    "commission": "0.0",
                    "commissionAsset": "USDT",
                    "tradeId": oid,
                    "time": 1713900000000,
                    "isMaker": False,
                }
            ],
        }
        self.orders[client_order_id] = payload
        return payload

    def place_market_order(self, symbol, side, quantity, client_order_id=None):
        return self.place_limit_order(symbol, side, quantity, 100.0, tif="IOC", client_order_id=client_order_id)

    def get_order(self, symbol, order_id=None, client_order_id=None):
        if client_order_id and client_order_id in self.orders:
            return self.orders[client_order_id]
        for row in self.orders.values():
            if row["symbol"] == symbol and (order_id is None or str(row["orderId"]) == str(order_id)):
                return row
        raise KeyError(f"order not found: symbol={symbol} order_id={order_id} client_order_id={client_order_id}")

    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        row = self.get_order(symbol, order_id=order_id, client_order_id=client_order_id)
        row = dict(row)
        row["status"] = "CANCELED"
        self.orders[row["clientOrderId"]] = row
        return row


class MiniCtx:
    def __init__(self, cfg, state, store, broker, risk):
        self.cfg = cfg
        self.state = state
        self.store = store
        self.broker = broker
        self.risk = risk


def build_cfg(config_path: str | None = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "execution": {
            "fee_bps": 10.0,
            "max_spread_pct": 0.01,
            "entry_slippage_cap_pct": 0.01,
            "exit_slippage_cap_pct": 0.01,
        },
        "execution_live": {
            "enabled": True,
            "broker": "live",
            "mode": "testnet",
            "default_order_type": "limit",
            "default_tif": "IOC",
            "starting_cash": 10000.0,
            "max_order_polls": 0,
            "order_poll_s": 0.0,
            "client_order_id_prefix": "smoke",
            "symbol_rules": {
                "ETHUSDT": {
                    "qty_step": 0.0001,
                    "price_tick": 0.01,
                    "min_qty": 0.0001,
                }
            },
        },
        "risk": {
            "enabled": True,
            "max_positions": 3,
            "max_total_exposure_usd": 5000.0,
            "max_exposure_per_symbol_usd": 2500.0,
            "cooldown_minutes_per_symbol": 0.0,
            "min_notional_per_trade": 10.0,
        },
        "ml_live": {
            "max_total_exposure_usd": 5000.0,
            "max_exposure_per_symbol_usd": 2500.0,
            "min_notional_per_trade": 10.0,
        },
    }

    if config_path:
        try:
            import yaml

            path = Path(config_path)
            if path.exists():
                loaded = yaml.safe_load(path.read_text()) or {}
                for key, value in loaded.items():
                    if isinstance(cfg.get(key), dict) and isinstance(value, dict):
                        merged = dict(cfg[key])
                        merged.update(value)
                        cfg[key] = merged
                    else:
                        cfg[key] = value
        except Exception:
            pass

    cfg.setdefault("execution", {})
    cfg["execution"].setdefault("fee_bps", 10.0)
    cfg["execution"].setdefault("max_spread_pct", 0.01)
    cfg["execution"].setdefault("entry_slippage_cap_pct", 0.01)
    cfg["execution"].setdefault("exit_slippage_cap_pct", 0.01)
    cfg.setdefault("execution_live", {})
    cfg["execution_live"]["enabled"] = True
    cfg["execution_live"]["broker"] = "live"
    cfg["execution_live"].setdefault("max_order_polls", 0)
    cfg["execution_live"].setdefault("order_poll_s", 0.0)
    cfg["execution_live"].setdefault("client_order_id_prefix", "smoke")
    return cfg


def count_rows(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    try:
        from tradarbot.core.events import OrderIntent
        from tradarbot.core.state import State, MarketState
        from tradarbot.execution.fill_reconciler import FillReconciler
        from tradarbot.execution.order_router import OrderRouter
        import tradarbot.execution.live_broker as live_broker_module
        from tradarbot.execution.slippage import validate_execution_bounds
        from tradarbot.risk.risk_manager import RiskManager
        from tradarbot.storage.sqlite_store import SQLiteStore
        ok("Imported Phase 5.1 modules")

        cfg = build_cfg(args.config)
        router = OrderRouter(cfg)
        routed = router.route_intent(OrderIntent(side="BUY", symbol="ETHUSDT", qty=0.123456, limit_px=2500.127, tif="IOC"))
        assert abs(routed.quantity - 0.1234) < 1e-12, routed
        assert abs(routed.price - 2500.12) < 1e-12, routed
        assert routed.order_type == "LIMIT", routed
        ok("OrderRouter rounded qty/price and produced a client_order_id")

        check = validate_execution_bounds(
            side="BUY",
            intended_px=2500.12,
            actual_px=2500.12,
            bid=2499.90,
            ask=2500.10,
            max_slippage_pct=0.01,
            max_spread_pct=0.01,
        )
        assert check.ok, check
        ok("Slippage/spread guard accepted a normal testnet-like book")

        reconciler = FillReconciler()
        normalized = reconciler.normalize_order(
            {
                "symbol": "ETHUSDT",
                "side": "BUY",
                "status": "FILLED",
                "orderId": "123",
                "clientOrderId": "abc",
                "origQty": "0.1",
                "executedQty": "0.1",
                "cummulativeQuoteQty": "250.0",
                "updateTime": 1713900000000,
                "fills": [{"price": "2500.0", "qty": "0.1", "commission": "0.0", "time": 1713900000000}],
            }
        )
        assert normalized.status == "FILLED"
        assert len(normalized.fills) == 1
        ok("FillReconciler normalized a FILLED order payload")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "phase5_1_smoke.sqlite")
            store = SQLiteStore(db_path)
            store.init_schema()
            ok("SQLiteStore initialized live execution schema")

            state = State()
            state.current_event_ts_ms = 1713900000000
            state.market["ETHUSDT"] = MarketState(bid=2499.90, ask=2500.10, last=2500.00, last_ts_ms=1713900000000)

            risk = RiskManager(cfg)
            risk.update_equity(10000.0)
            risk.update_positions(exposure_by_symbol={}, total_exposure=0.0, unrealized_pnl_total=0.0)

            original_client = live_broker_module.BinanceClient
            live_broker_module.BinanceClient = FakeClient
            try:
                broker = live_broker_module.LiveBroker(cfg=cfg, store=store, starting_cash=10000.0)
            finally:
                live_broker_module.BinanceClient = original_client

            ctx = MiniCtx(cfg=cfg, state=state, store=store, broker=broker, risk=risk)

            buy_intent = OrderIntent(side="BUY", symbol="ETHUSDT", qty=0.10009, limit_px=2500.12, tif="IOC")
            decision = risk.check(buy_intent, ctx, strat_name="phase5_1_smoke")
            assert decision["approved"], decision
            broker.execute_intent(decision["intent"], ctx)

            assert "ETHUSDT" in broker.positions, broker.positions_snapshot()
            assert broker.positions["ETHUSDT"].qty > 0.0, broker.positions_snapshot()
            ok("LiveBroker executed mocked BUY and opened a position")

            assert count_rows(db_path, "orders") >= 1
            assert count_rows(db_path, "order_fills") >= 1
            assert count_rows(db_path, "execution_events") >= 1
            conn = sqlite3.connect(db_path)
            try:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                conn.close()
            assert "positions_live" in tables, tables
            ok("SQLite persistence wrote orders, fills, events, and created the positions_live table")

            broker.close_all(ctx, reason="smoke_test")
            remaining_qty = broker.positions.get("ETHUSDT").qty if broker.positions.get("ETHUSDT") else 0.0
            assert remaining_qty <= 1e-12, broker.positions_snapshot()
            ok("close_all flattened the mocked live position")

            print("\nRow counts:")
            for table in ["orders", "order_fills", "execution_events"]:
                print(f"  {table}: {count_rows(db_path, table)}")
            print("  positions_live: table present")

        print("\nPASS: Phase 5.1 smoke test completed successfully.")
        return 0

    except Exception as exc:
        fail(str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
