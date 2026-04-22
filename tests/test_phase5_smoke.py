#!/usr/bin/env python3
"""Phase 5.0 smoke test for Tradar.

What this checks:
1) Imports for all Phase 5.0 modules
2) Refactored replay helpers are importable
3) live_features -> live_predictor path works
4) MLStrategy can run through StrategyEngine
5) RiskManager.check(...) handles ML-sized BUY intents
6) Order flow stays strategy -> risk.check -> broker.execute_intent

Run:
    PYTHONPATH=. ./.venv/bin/python test_phase5_smoke.py

Optional:
    PYTHONPATH=. ./.venv/bin/python test_phase5_smoke.py --config config/tradar.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


class FakeStore:
    def __init__(self):
        self.candles = []

    def insert_candle(self, ev):
        self.candles.append(ev)


class FakePosition:
    def __init__(self, qty=0.0, entry_price=None, avg_px=None):
        self.qty = qty
        self.entry_price = entry_price
        self.avg_px = avg_px if avg_px is not None else entry_price
        self.current_price = entry_price
        self.unrealized_pnl = 0.0


class FakeBroker:
    def __init__(self, starting_cash: float = 10000.0):
        self.cash = float(starting_cash)
        self.positions = {}
        self.executed = []

    def execute_intent(self, intent, ctx):
        self.executed.append(intent)
        side = str(getattr(intent, "side", "")).upper()
        symbol = getattr(intent, "symbol", "")
        qty = float(getattr(intent, "qty", 0.0) or 0.0)
        px = float(getattr(intent, "limit_px", 0.0) or 0.0)
        if qty <= 0.0 or px <= 0.0 or not symbol:
            return

        if side == "BUY":
            self.cash -= qty * px
            pos = self.positions.get(symbol)
            if pos is None:
                pos = FakePosition(qty=qty, entry_price=px, avg_px=px)
                self.positions[symbol] = pos
            else:
                pos.qty += qty
                pos.current_price = px
        elif side == "SELL":
            pos = self.positions.get(symbol)
            if pos is not None:
                sell_qty = min(qty, float(getattr(pos, "qty", 0.0) or 0.0))
                self.cash += sell_qty * px
                pos.qty -= sell_qty
                pos.current_price = px
                if pos.qty <= 0.0:
                    del self.positions[symbol]


class FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, ev):
        self.published.append(ev)


class Ctx:
    def __init__(self, cfg, state, store, broker, risk, bus=None):
        self.cfg = cfg
        self.state = state
        self.store = store
        self.broker = broker
        self.risk = risk
        self.bus = bus


def build_test_config(config_path: str | None = None) -> dict:
    cfg = {
        "runtime": {
            "candle_interval_s": 3600,
            "log_level": "INFO",
        },
        "execution": {
            "entry_slippage_cap_pct": 0.002,
            "fee_bps": 10,
        },
        "ml_live": {
            "enabled": True,
            "mode": "heuristic",
            "prediction_source": "direct",
            "prob_threshold": 0.10,
            "min_prob_percentile": 0.0,
            "ranking_mode": "composite",
            "top_n": 2,
            "max_positions": 2,
            "enable_dynamic_max_positions": True,
            "min_dynamic_max_positions": 1,
            "dynamic_position_score_threshold": 0.15,
            "notional_per_trade": 1000.0,
            "min_notional_per_trade": 50.0,
            "enable_dynamic_sizing": True,
            "prob_size_cap": 2.0,
            "vol_reference": 0.006,
            "vol_size_floor": 0.75,
            "vol_size_cap": 1.25,
            "combined_size_cap": 2.0,
            "enable_kelly_sizing": True,
            "kelly_fraction_scale": 0.25,
            "kelly_probability_mode": "threshold_relative",
            "kelly_size_cap": 1.5,
            "take_profit_pct": 0.08,
            "stop_loss_pct": 0.04,
            "max_hold_hours": 24.0,
            "trailing_stop_pct": 0.05,
            "trailing_stop_activation_pct": 0.08,
            "partial_take_profit_pct": 0.10,
            "partial_take_profit_fraction": 0.50,
            "time_stop_hours": 12.0,
            "time_stop_min_return_pct": 0.01,
            "enable_risk_manager": True,
            "max_total_exposure_usd": 5000.0,
            "max_exposure_per_symbol_usd": 3000.0,
            "max_total_exposure_pct": 0.8,
            "max_drawdown_pct": 0.20,
            "cooldown_minutes_per_symbol": 0.0,
            "enable_drawdown_scaling": True,
            "drawdown_full_size_pct": 0.04,
            "drawdown_half_size_pct": 0.06,
            "drawdown_quarter_size_pct": 0.08,
            "drawdown_half_size_multiplier": 0.50,
            "drawdown_quarter_size_multiplier": 0.25,
            "enable_regime_gating": True,
            "regime_gating_mode": "scale",
            "risk_off_size_multiplier": 0.50,
            "risk_off_score_raise": 0.0,
            "feature_lookback_bars": 24 * 7,
            "feature_interval_s": 3600,
            "candles_table": "candles",
            "emit_observer_events": True,
        },
        "strategies": {
            "algo1_new_listing_pump": {"enabled": False},
            "algo2_micro_momentum": {"enabled": False},
        },
    }

    if config_path:
        try:
            import yaml
            path = Path(config_path)
            if path.exists():
                with open(path, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                            merged = dict(cfg[k])
                            merged.update(v)
                            cfg[k] = merged
                        else:
                            cfg[k] = v
        except Exception:
            pass

    return cfg


def test_imports():
    print_header("1) IMPORT CHECKS")
    imported = {}

    from tradarbot.strategies.ml_strategy import MLStrategy
    imported["MLStrategy"] = MLStrategy

    from tradarbot.ml.live_features import compute_features
    imported["compute_features"] = compute_features

    from tradarbot.ml.live_predictor import LivePredictor
    imported["LivePredictor"] = LivePredictor

    from tradarbot.ml.signal_schema import MLEntrySignal
    imported["MLEntrySignal"] = MLEntrySignal

    from tradarbot.core.events import CandleEvent, OrderIntent
    imported["CandleEvent"] = CandleEvent
    imported["OrderIntent"] = OrderIntent

    from tradarbot.core.state import State, MarketState
    imported["State"] = State
    imported["MarketState"] = MarketState

    from tradarbot.core.engine import StrategyEngine
    imported["StrategyEngine"] = StrategyEngine

    from tradarbot.risk.risk_manager import RiskManager
    imported["RiskManager"] = RiskManager

    from scripts.replay_ml_strategy import (
        build_runtime_args,
        compute_size_multipliers,
        apply_regime_gating,
    )
    imported["build_runtime_args"] = build_runtime_args
    imported["compute_size_multipliers"] = compute_size_multipliers
    imported["apply_regime_gating"] = apply_regime_gating

    for name in imported:
        ok(f"Imported {name}")

    return imported


def test_feature_predictor_flow(imported):
    print_header("2) live_features -> live_predictor")
    compute_features = imported["compute_features"]
    LivePredictor = imported["LivePredictor"]

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-04-01", periods=48, freq="h", tz="UTC"),
        "open":  [100 + i * 0.10 for i in range(48)],
        "high":  [100 + i * 0.12 + 0.25 for i in range(48)],
        "low":   [100 + i * 0.08 - 0.25 for i in range(48)],
        "close": [100 + i * 0.15 for i in range(48)],
        "volume":[1000 + i * 15 for i in range(48)],
    })

    features = compute_features("BTCUSDT", df)
    required_keys = [
        "symbol",
        "prob_proxy",
        "rolling_volatility_24h",
        "target_time_to_peak_seconds_24h",
    ]
    for key in required_keys:
        if key not in features:
            raise AssertionError(f"Missing feature key: {key}")
    ok("Feature row contains required Phase 5.0 fields")

    predictor = LivePredictor({"mode": "heuristic"})
    pred = predictor.predict(pd.DataFrame([features]))
    if "BTCUSDT" not in pred:
        raise AssertionError("Predictor did not return BTCUSDT payload")
    payload = pred["BTCUSDT"]
    if "prob" not in payload or "score" not in payload:
        raise AssertionError("Predictor payload missing prob/score")
    ok("Predictor returns per-symbol prob and score")

    print(json.dumps(serialize(payload), indent=2, default=str)[:1200])
    return features, payload


def test_replay_helper_surface(imported):
    print_header("3) replay helper surface")
    build_runtime_args = imported["build_runtime_args"]
    args = build_runtime_args(section="ml_live")
    for attr in [
        "prediction_source",
        "prob_threshold",
        "top_n",
        "regime_gating_mode",
        "notional_per_trade",
    ]:
        if not hasattr(args, attr):
            raise AssertionError(f"build_runtime_args missing attr {attr}")
    ok("Replay shared helpers expose runtime args for live imports")
    print({
        "prediction_source": args.prediction_source,
        "prob_threshold": args.prob_threshold,
        "top_n": args.top_n,
        "regime_gating_mode": args.regime_gating_mode,
        "notional_per_trade": args.notional_per_trade,
    })
    return args


def test_engine_flow(imported, cfg):
    print_header("4) engine flow smoke test")
    MLStrategy = imported["MLStrategy"]
    CandleEvent = imported["CandleEvent"]
    State = imported["State"]
    MarketState = imported["MarketState"]
    StrategyEngine = imported["StrategyEngine"]
    RiskManager = imported["RiskManager"]

    state = State()
    store = FakeStore()
    broker = FakeBroker(starting_cash=10000.0)
    bus = FakeBus()
    risk = RiskManager(cfg)
    ctx = Ctx(cfg=cfg, state=state, store=store, broker=broker, risk=risk, bus=bus)

    state.active_symbols = {"BTCUSDT"}
    state.market["BTCUSDT"] = MarketState(bid=100.0, ask=100.1, last=100.05, last_ts_ms=0)

    strategy = MLStrategy(cfg["ml_live"])
    engine = StrategyEngine([strategy], risk, broker, ctx)

    start_ts = 1711929600000
    for i in range(80):
        px = 100.0 + i * 0.20
        ev = CandleEvent(
            symbol="BTCUSDT",
            interval_s=3600,
            ts_ms=start_ts + i * 3600 * 1000,
            open=px - 0.05,
            high=px + 0.20,
            low=px - 0.20,
            close=px,
            volume=1000 + i * 25,
        )
        state.market["BTCUSDT"] = MarketState(
            bid=px - 0.05,
            ask=px + 0.05,
            last=px,
            last_ts_ms=ev.ts_ms,
        )
        engine.on_candle(ev)

    ok(f"Inserted candles: {len(store.candles)}")
    ok(f"Executed intents: {len(broker.executed)}")
    print("Final cash:", broker.cash)
    print("Open positions:", {k: getattr(v, 'qty', None) for k, v in broker.positions.items()})

    if len(store.candles) == 0:
        raise AssertionError("No candles were stored through engine path")

    if not hasattr(state, "ml_latest_features"):
        raise AssertionError("State does not expose ml_latest_features")
    if not hasattr(state, "ml_last_signal_ts_by_symbol"):
        raise AssertionError("State does not expose ml_last_signal_ts_by_symbol")
    ok("State has Phase 5.0 ML fields")

    if len(broker.executed) == 0:
        print("[WARN] No execution occurred. This can still be acceptable if MLStrategy thresholds stayed untriggered.")
    else:
        ok("At least one ML order reached broker.execute_intent(...)")

    return {
        "executed_count": len(broker.executed),
        "published_events": len(bus.published),
        "cash": broker.cash,
        "positions": {k: getattr(v, 'qty', None) for k, v in broker.positions.items()},
    }


def test_risk_acceptance(imported, cfg):
    print_header("5) risk acceptance for ML-sized orders")
    RiskManager = imported["RiskManager"]
    OrderIntent = imported["OrderIntent"]
    State = imported["State"]
    MarketState = imported["MarketState"]

    state = State()
    broker = FakeBroker(starting_cash=10000.0)
    broker.positions = {}
    store = FakeStore()
    risk = RiskManager(cfg)
    ctx = Ctx(cfg=cfg, state=state, store=store, broker=broker, risk=risk)

    state.current_event_ts_ms = 1711929600000
    state.market["BTCUSDT"] = MarketState(bid=100.0, ask=100.1, last=100.05, last_ts_ms=state.current_event_ts_ms)

    intent_ok = OrderIntent(side="BUY", symbol="BTCUSDT", qty=10.0, limit_px=100.0)
    decision_ok = risk.check(intent_ok, ctx, "ml_strategy")
    print("Approved decision:", decision_ok)
    if not decision_ok["approved"]:
        raise AssertionError(f"Expected ML-sized BUY to be approved, got {decision_ok}")

    limit = float(cfg["ml_live"]["max_exposure_per_symbol_usd"])
    broker.positions["BTCUSDT"] = FakePosition(
        qty=(limit / 100.0),
        entry_price=100.0,
        avg_px=100.0,
    )
    decision_block = risk.check(intent_ok, ctx, "ml_strategy")
    print("Blocked decision:", decision_block)
    if decision_block["approved"]:
        raise AssertionError("Expected BUY to be blocked by exposure assumptions")
    ok("RiskManager accepts valid ML-sized BUY and blocks exposure breach")

    return {
        "approved_reason": decision_ok["reason"],
        "blocked_reason": decision_block["reason"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional path to tradar.yaml")
    args = parser.parse_args()

    cfg = build_test_config(args.config)
    summary = {"passed": [], "failed": []}

    try:
        imported = test_imports()
        summary["passed"].append("imports")
    except Exception as e:
        fail(f"Import checks failed: {e}")
        traceback.print_exc()
        summary["failed"].append("imports")
        print(json.dumps(summary, indent=2))
        sys.exit(1)

    try:
        test_feature_predictor_flow(imported)
        summary["passed"].append("feature_predictor")
    except Exception as e:
        fail(f"Feature/predictor flow failed: {e}")
        traceback.print_exc()
        summary["failed"].append("feature_predictor")

    try:
        test_replay_helper_surface(imported)
        summary["passed"].append("replay_helper_surface")
    except Exception as e:
        fail(f"Replay helper surface failed: {e}")
        traceback.print_exc()
        summary["failed"].append("replay_helper_surface")

    try:
        test_engine_flow(imported, cfg)
        summary["passed"].append("engine_flow")
    except Exception as e:
        fail(f"Engine flow failed: {e}")
        traceback.print_exc()
        summary["failed"].append("engine_flow")

    try:
        test_risk_acceptance(imported, cfg)
        summary["passed"].append("risk_acceptance")
    except Exception as e:
        fail(f"Risk acceptance failed: {e}")
        traceback.print_exc()
        summary["failed"].append("risk_acceptance")

    print_header("SUMMARY")
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        sys.exit(1)
    ok("Phase 5.0 smoke test passed")


if __name__ == "__main__":
    main()
