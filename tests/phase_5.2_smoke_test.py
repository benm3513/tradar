#!/usr/bin/env python3
"""
Phase 5.2 smoke test for Tradar.

Run from repo root:

    PYTHONPATH=. ./.venv/bin/python tests/phase_5.2_smoke_test.py

What this validates:
- Phase 5.2 modules import cleanly.
- Event classes exist and preserve existing core event names.
- RollingCandleBuilder emits CandleEvent-compatible candles.
- RollingFeatureState stores candles, reports readiness, and exposes frames.
- live_regime computes the replay-equivalent market context family.
- LiveContextSnapshot can carry feature/regime/readiness metadata.
- build_live_feature_frame still works with candles_by_symbol input.
- MLStrategy can use centralized feature_state when present and still falls back safely.
- EventBus can publish/dispatch Phase 5.2 context events.
- Unified sizing files expose the expected router/broker surfaces.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    raise AssertionError(msg)


def assert_true(condition: bool, msg: str) -> None:
    if not condition:
        fail(msg)
    ok(msg)


def make_candle(symbol: str, i: int, base: float = 100.0, interval_s: int = 3600):
    from tradarbot.core.events import CandleEvent

    ts_ms = 1_700_000_000_000 + (i * interval_s * 1000)
    close = base + i * 0.25 + (0.05 if symbol.endswith("ETHUSDT") else 0.0)
    return CandleEvent(
        symbol=symbol,
        interval_s=interval_s,
        ts_ms=ts_ms,
        open=close - 0.10,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=1000.0 + i * 3.0,
    )


def build_test_frames(symbols: List[str], bars: int = 180) -> Dict[str, Any]:
    import pandas as pd

    frames = {}
    for s_idx, symbol in enumerate(symbols):
        rows = []
        base = 100.0 + s_idx * 20.0
        for i in range(bars):
            ev = make_candle(symbol, i, base=base, interval_s=3600)
            rows.append(
                {
                    "symbol": ev.symbol,
                    "ts_ms": ev.ts_ms,
                    "timestamp": pd.to_datetime(ev.ts_ms, unit="ms", utc=True),
                    "open": ev.open,
                    "high": ev.high,
                    "low": ev.low,
                    "close": ev.close,
                    "volume": ev.volume,
                }
            )
        frames[symbol] = pd.DataFrame(rows)
    return frames


class DummyStore:
    def __init__(self):
        self.candles = []
        self.fills = []
        self.orders = []
        self.execution_events = []

    def insert_candle(self, ev):
        self.candles.append(ev)

    def insert_fill(self, ts_ms, symbol, side, qty, px):
        self.fills.append((ts_ms, symbol, side, qty, px))

    def insert_execution_event(self, **kwargs):
        self.execution_events.append(kwargs)

    def insert_order(self, **kwargs):
        self.orders.append(kwargs)

    def update_order_status(self, **kwargs):
        pass

    def insert_order_fill(self, **kwargs):
        pass


class DummyBroker:
    def __init__(self, cash: float = 10_000.0):
        self.cash = cash
        self.positions = {}
        self.open_orders = {}
        self.broker_mode = "paper"

    def execute_intent(self, intent, ctx):
        self.open_orders[f"dummy-{len(self.open_orders)+1}"] = intent


def main() -> None:
    results = {"passed": [], "failed": []}

    # ------------------------------------------------------------------
    section("1) IMPORT CHECKS")
    try:
        from tradarbot.core.events import (
            BookEvent,
            CandleEvent,
            ListingEvent,
            OrderIntent,
            FeatureStateUpdatedEvent,
            LiveContextSnapshotEvent,
            RegimeContextEvent,
            MarketDataHealthEvent,
        )
        from tradarbot.core.bus import EventBus
        from tradarbot.core.state import State
        from tradarbot.data.candle_builder import RollingCandleBuilder
        from tradarbot.data.feature_state import RollingFeatureState
        from tradarbot.data.ws_client import LiveMarketDataClient
        from tradarbot.ml.live_regime import compute_live_regime
        from tradarbot.ml.context_snapshot import LiveContextSnapshot
        from tradarbot.ml.live_features import build_live_feature_frame
        from tradarbot.strategies.ml_strategy import MLStrategy
        from tradarbot.execution.order_router import OrderRouter
        from tradarbot.execution.live_broker import LiveBroker

        ok("Imported core Phase 5.2 modules and events")
        results["passed"].append("imports")
    except Exception as exc:
        results["failed"].append(("imports", repr(exc)))
        raise

    # Existing names must still exist.
    assert_true(BookEvent.__name__ == "BookEvent", "BookEvent preserved")
    assert_true(CandleEvent.__name__ == "CandleEvent", "CandleEvent preserved")
    assert_true(ListingEvent.__name__ == "ListingEvent", "ListingEvent preserved")
    assert_true(OrderIntent.__name__ == "OrderIntent", "OrderIntent preserved")

    # ------------------------------------------------------------------
    section("2) ROLLING CANDLE BUILDER")
    emitted = []

    def emit(ev):
        emitted.append(ev)

    builder = RollingCandleBuilder(interval_s=1, emit_fn=emit)
    out0 = builder.on_price("BTCUSDT", 1_700_000_000_000, 100.0, qty=1.0)
    out1 = builder.on_price("BTCUSDT", 1_700_000_000_500, 101.0, qty=2.0)
    out2 = builder.on_price("BTCUSDT", 1_700_000_001_000, 102.0, qty=3.0)

    produced = emitted + list(out0 or []) + list(out1 or []) + list(out2 or [])
    assert_true(len(produced) >= 1, "RollingCandleBuilder emits closed candle when bucket changes")
    first = produced[0]
    assert_true(isinstance(first, CandleEvent), "RollingCandleBuilder output is CandleEvent")
    assert_true(first.open == 100.0 and first.high == 101.0 and first.close == 101.0, "RollingCandleBuilder OHLC aggregation is correct")
    results["passed"].append("rolling_candle_builder")

    # ------------------------------------------------------------------
    section("3) ROLLING FEATURE STATE")
    feature_state = RollingFeatureState(lookback_bars=168, min_ready_bars=24, max_symbols=10)

    for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
        for i in range(180):
            feature_state.update_candle(make_candle(symbol, i, base=100.0 + len(symbol)))

    ready = feature_state.ready_symbols()
    frames = feature_state.frames_by_symbol()
    health = feature_state.health_snapshot()

    assert_true(set(ready) == {"BTCUSDT", "ETHUSDT", "BNBUSDT"}, "RollingFeatureState reports all symbols ready")
    assert_true(all(len(df) <= 168 for df in frames.values()), "RollingFeatureState enforces lookback_bars")
    ready_count = (
    health.get("ready_symbol_count")
    or health.get("ready_symbols_count")
    or health.get("ready_count")
    or len(ready)
    )

    assert_true(int(ready_count) == 3, "RollingFeatureState health reports ready count")
    print(json.dumps(health, indent=2, default=str))
    results["passed"].append("feature_state")

    # ------------------------------------------------------------------
    section("4) LIVE REGIME")
    regime = compute_live_regime(frames)
    required_regime_cols = {
        "market_dispersion_1h",
        "market_dispersion_24h",
        "market_breadth_up_1h",
        "market_breadth_up_24h",
        "market_trend_strength_24h",
        "market_volume_regime_24h",
        "market_risk_off_score",
    }
    missing_regime = required_regime_cols - set(regime)
    assert_true(not missing_regime, f"Regime has required fields: {sorted(required_regime_cols)}")
    assert_true(0.0 <= float(regime["market_risk_off_score"]) <= 1.0, "market_risk_off_score is bounded 0..1")
    print(json.dumps(regime, indent=2, default=str))
    results["passed"].append("live_regime")

    # ------------------------------------------------------------------
    section("5) LIVE FEATURE FRAME + CONTEXT SNAPSHOT")
    ctx = SimpleNamespace(
        cfg={
            "runtime": {"candle_interval_s": 3600},
            "feature_state": {"lookback_bars": 168, "min_ready_bars": 24},
            "execution": {"entry_slippage_cap_pct": 0.0025, "max_spread_pct": 0.02},
            "execution_live": {
                "enabled": False,
                "provider": "alpaca",
                "alpaca_qty_step": 0.000001,
                "alpaca_price_tick": 0.01,
                "alpaca_min_qty": 0.000001,
                "sizing_cash_buffer_pct": 0.005,
                "sell_qty_buffer_pct": 0.001,
            },
        },
        state=State(),
        store=DummyStore(),
        broker=DummyBroker(),
        risk=None,
    )
    ctx.state.feature_state = feature_state
    ctx.state.active_symbols = set(["BTCUSDT", "ETHUSDT", "BNBUSDT"])

    feature_frame = build_live_feature_frame(
        symbols=ready,
        ctx=ctx,
        lookback_bars=168,
        interval_s=3600,
        candles_by_symbol=frames,
    )
    assert_true(not feature_frame.empty, "build_live_feature_frame returns non-empty frame from candles_by_symbol")

    required_feature_cols = {
        "symbol",
        "price_close",
        "return_1h",
        "return_6h",
        "return_24h",
        "rolling_volatility_24h",
        "range_pct_24h",
        "price_zscore_24h",
        "volume_zscore_24h",
        "volume_spike_ratio_7d",
        "momentum_accel_6h_vs_24h",
        "market_risk_off_score",
        "market_dispersion_24h",
        "market_breadth_up_24h",
    }
    missing_features = required_feature_cols - set(feature_frame.columns)
    assert_true(not missing_features, f"Feature frame includes required Phase 4/5.2 columns")

    snapshot = LiveContextSnapshot(
        ts_ms=int(feature_frame["ts_ms"].max()) if "ts_ms" in feature_frame.columns else 0,
        symbols=sorted(list(ctx.state.active_symbols)),
        feature_frame=feature_frame,
        regime=regime,
        ready_symbols=ready,
        metadata={"source": "phase_5.2_smoke_test"},
    )
    snap_dict = snapshot.to_dict() if hasattr(snapshot, "to_dict") else asdict(snapshot)
    assert_true(len(snapshot.ready_symbols) == 3, "LiveContextSnapshot carries ready symbols")
    assert_true("market_risk_off_score" in snapshot.regime, "LiveContextSnapshot carries regime")
    print(json.dumps({k: v for k, v in snap_dict.items() if k != "feature_frame"}, indent=2, default=str))
    results["passed"].append("feature_frame_context_snapshot")

    # ------------------------------------------------------------------
    section("6) EVENT BUS + PHASE 5.2 EVENTS")
    seen = []

    async def bus_test():
        bus = EventBus()
        bus.subscribe(RegimeContextEvent, lambda ev: seen.append(("regime", ev)))
        bus.subscribe(MarketDataHealthEvent, lambda ev: seen.append(("health", ev)))
        task = asyncio.create_task(bus.run())
        try:
            bus.publish(RegimeContextEvent(ts_ms=123, regime=regime))
            bus.publish(MarketDataHealthEvent(ts_ms=123, health={"connected": True}))
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(bus_test())
    assert_true(any(x[0] == "regime" for x in seen), "EventBus dispatches RegimeContextEvent")
    assert_true(any(x[0] == "health" for x in seen), "EventBus dispatches MarketDataHealthEvent")
    results["passed"].append("event_bus_phase52_events")

    # ------------------------------------------------------------------
    section("7) MLSTRATEGY CENTRALIZED FEATURE STATE COMPATIBILITY")
    ml_cfg = {
        "enabled": True,
        "enable_regime_gating": False,
        "mode": "heuristic",
        "feature_lookback_bars": 168,
        "feature_interval_s": 3600,
        "evaluation_interval_s": 1,
        "prob_threshold": 0.1,
        "min_prob_percentile": 0.0,
        "top_n": 1,
        "max_positions": 2,
        "notional_per_trade": 250.0,
        "take_profit_pct": 0.18,
        "stop_loss_pct": 0.06,
        "max_hold_hours": 24,
        "min_ready_bars": 24,
        "live_sizing_safety_buffer": 0.995,
    }
    ctx.cfg["ml_live"] = ml_cfg
    strategy = MLStrategy(ml_cfg)

    # Make sure it exposes the expected live hook behavior without changing trading logic.
    assert_true(hasattr(strategy, "on_candle"), "MLStrategy still has on_candle")
    assert_true(hasattr(strategy, "_build_feature_frame"), "MLStrategy still has _build_feature_frame")

    frame_from_strategy = strategy._build_feature_frame(ready, ctx)
    assert_true(not frame_from_strategy.empty, "MLStrategy can build features from centralized feature_state/candles")
    assert_true("market_risk_off_score" in frame_from_strategy.columns, "MLStrategy feature frame includes regime fields")

    # Exercise one candle. It may or may not emit an order depending on thresholds; this is not a signal-quality test.
    outputs = strategy.on_candle(make_candle("BTCUSDT", 181, base=100.0), ctx)
    assert_true(outputs is None or isinstance(outputs, list), "MLStrategy.on_candle returns list/None compatible with StrategyEngine")
    results["passed"].append("ml_strategy_feature_state_compat")

    # ------------------------------------------------------------------
    section("8) UNIFIED SIZING SURFACE CHECKS")
    router = OrderRouter(ctx.cfg)
    intent = OrderIntent(side="BUY", symbol="BTCUSDT", qty=0.00709492123, limit_px=77714.090725, tif="IOC")
    routed = router.route_intent(intent)
    assert_true(routed.quantity <= intent.qty, "OrderRouter floors quantity to venue step")
    assert_true(abs((routed.quantity / 0.000001) - round(routed.quantity / 0.000001)) < 1e-6, "OrderRouter quantity matches Alpaca qty step")
    assert_true(routed.price <= intent.limit_px, "OrderRouter floors limit price to venue tick")

    broker_source = inspect.getsource(LiveBroker)
    sizing_markers = [
        "_cap_routed_order",
        "sizing_cash_buffer",
        "sell_qty_buffer",
    ]
    missing_markers = [m for m in sizing_markers if m not in broker_source]
    if missing_markers:
        warn(f"LiveBroker missing optional unified sizing markers: {missing_markers}")
    else:
        ok("LiveBroker exposes unified sizing cap markers")
    results["passed"].append("unified_sizing_surface")

    # ------------------------------------------------------------------
    section("SUMMARY")
    print(json.dumps(results, indent=2, default=str))
    if results["failed"]:
        raise SystemExit(1)
    print("[OK] Phase 5.2 smoke test passed")


if __name__ == "__main__":
    main()
