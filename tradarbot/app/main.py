import asyncio
import logging
import os
import re
import time
import httpx
from typing import Any, Dict, Set

import yaml
import pandas as pd
import warnings

from tradarbot.app.context import Ctx
from tradarbot.core.bus import EventBus
from tradarbot.core.engine import StrategyEngine
from tradarbot.core.events import CandleEvent, ListingEvent, BookEvent, OrderIntent, FeatureStateUpdatedEvent, RegimeContextEvent, LiveContextSnapshotEvent, MarketDataHealthEvent
from tradarbot.core.state import State
from tradarbot.data.candles import CandleBuilder1s
from tradarbot.data.candle_builder import RollingCandleBuilder
from tradarbot.data.feature_state import RollingFeatureState
from tradarbot.data.ws_client import LiveMarketDataClient
from tradarbot.data.symbol_registry import SymbolRegistry
from tradarbot.exchange.binance.rest import BinanceRestClient
from tradarbot.exchange.binance.ws_manager import BinanceWSManager
from tradarbot.execution.paper_broker import PaperBroker
from tradarbot.execution.live_broker import LiveBroker
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.strategies.algo2_micro_momentum import Algo2MicroMomentum
from tradarbot.strategies.algo1_new_listing_pump import Algo1NewListingPump
from tradarbot.strategies.ml_strategy import MLStrategy
from tradarbot.safety.kill_switch import KillSwitchManager, KillSwitchReason
from tradarbot.safety.safe_mode import SafeModeManager
from tradarbot.safety.stale_data_guard import StaleDataGuard
from tradarbot.safety.health_rules import HealthMonitor, STATUS_KILL_SWITCH, STATUS_SAFE_MODE


pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna",
    category=FutureWarning,
)

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _expand_env_vars(raw: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(name, default)
    return _ENV_PATTERN.sub(repl, raw)


def load_runtime_config() -> Dict[str, Any]:
    config_path = os.environ.get("TRADAR_CONFIG", "config/tradar.yaml")
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(_expand_env_vars(fh.read())) or {}

    runtime = cfg.setdefault("runtime", {})
    runtime["profile"] = os.environ.get("TRADAR_PROFILE", runtime.get("profile", "paper"))
    if os.environ.get("TRADAR_LOG_LEVEL"):
        runtime["log_level"] = os.environ["TRADAR_LOG_LEVEL"]
    if os.environ.get("TRADAR_DB_PATH"):
        runtime["db_path"] = os.environ["TRADAR_DB_PATH"]
    if os.environ.get("TRADAR_DATA_DIR"):
        runtime["data_dir"] = os.environ["TRADAR_DATA_DIR"]
    if os.environ.get("TRADAR_ARTIFACT_DIR"):
        runtime["artifact_dir"] = os.environ["TRADAR_ARTIFACT_DIR"]
    return cfg

def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def main() -> None:
    cfg: Dict[str, Any] = load_runtime_config()
    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
    log = logging.getLogger("tradarbot")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("Booting...")

    # Core components
    state = State()
    store = SQLiteStore(str(cfg.get("runtime", {}).get("db_path", "tradarbot.db")))
    store.init_schema()

    exec_live_cfg = cfg.get("execution_live", {})
    broker_mode = str(exec_live_cfg.get("broker", "paper") or "paper").lower()
    if bool(exec_live_cfg.get("enabled", False)) and broker_mode in {"live", "dry_run_live", "dry-run-live", "dryrun"}:
        broker = LiveBroker(cfg=cfg, store=store, starting_cash=float(exec_live_cfg.get("starting_cash", 0.0) or 0.0))
    else:
        broker = PaperBroker(
            fee_bps=float(cfg["execution"]["fee_bps"]),
            starting_cash=float(exec_live_cfg.get("starting_cash", 10_000.0) or 10_000.0),
        )
    risk = RiskManager(cfg)

    # Phase 5.6 safety layer
    kill_switch = KillSwitchManager(cfg, store=store)
    safe_mode = SafeModeManager(cfg, store=store)
    stale_guard = StaleDataGuard(cfg, store=store)
    health_monitor = HealthMonitor(cfg, store=store, stale_guard=stale_guard)
    risk.attach_safety(
        kill_switch_manager=kill_switch,
        safe_mode_manager=safe_mode,
        health_monitor=health_monitor,
        stale_data_guard=stale_guard,
    )

    ctx = Ctx(cfg=cfg, state=state, store=store, broker=broker, risk=risk)
    ctx.kill_switch = kill_switch
    ctx.safe_mode = safe_mode
    ctx.stale_guard = stale_guard
    ctx.health_monitor = health_monitor

    previous_safety = store.latest_safety_snapshot() if hasattr(store, "latest_safety_snapshot") else None
    if previous_safety and bool(previous_safety.get("kill_switch")):
        kill_switch.activate(KillSwitchReason.STARTUP_FAIL_CLOSED, message="previous runtime ended with kill switch active", metadata=previous_safety)
    elif previous_safety and bool(previous_safety.get("safe_mode")):
        safe_mode.activate("startup_previous_safe_mode", message="previous runtime ended in safe mode", metadata=previous_safety)

    state.set_runtime_safety_snapshot({
        "safe_mode": safe_mode.is_active(),
        "kill_switch": kill_switch.is_active(),
        "health_status": "OK",
        "health_messages": [],
    })

    feature_cfg = cfg.get("feature_state", {}) or {}
    if bool(feature_cfg.get("enabled", True)):
        ctx.state.feature_state = RollingFeatureState(
            lookback_bars=int(feature_cfg.get("lookback_bars", cfg.get("ml_live", {}).get("feature_lookback_bars", 168))),
            min_ready_bars=int(feature_cfg.get("min_ready_bars", 24)),
            max_symbols=feature_cfg.get("max_symbols"),
            interval_s=int(feature_cfg.get("candle_interval_s", cfg.get("runtime", {}).get("candle_interval_s", 1))),
        )
    else:
        ctx.state.feature_state = None

    ctx.state.active_symbols = set()

    ctx.state._poll_ok = 0
    ctx.state._poll_err = 0
    ctx.state._poll_backoff_s = 0.0


    ctx.state._candle_count = 0
    ctx.state._last_candle_log = time.time()
    bus = EventBus()

    # Strategies
    strategies = []

    algo1_cfg = cfg.get("strategies", {}).get("algo1_new_listing_pump", {})
    if algo1_cfg.get("enabled", False):
        strategies.append(Algo1NewListingPump(algo1_cfg))

    algo2_cfg = cfg.get("strategies", {}).get("algo2_micro_momentum", {})
    if algo2_cfg.get("enabled", False):
        strategies.append(Algo2MicroMomentum(algo2_cfg))
    
    ml_cfg = cfg.get("ml_live", {})
    if ml_cfg.get("enabled", False):
        strategies.append(MLStrategy(ml_cfg))

    log.info("loaded strategies=%s", [s.name for s in strategies])

    engine = StrategyEngine(strategies=strategies, risk=risk, broker=broker, ctx=ctx)

    # Route events
    def on_candle_event(ev: CandleEvent):
        _on_candle_for_feature_state(ctx, bus, ev)
        return engine.on_candle(ev)

    bus.subscribe(CandleEvent, on_candle_event)
    bus.subscribe(ListingEvent, engine.on_listing)

    # Candle builder emits CandleEvents onto the bus
    def emit_candle(symbol: str, close_ts_ms: int, candle) -> None:
        ev = CandleEvent(
            symbol=symbol,
            interval_s=int(cfg["runtime"]["candle_interval_s"]),
            ts_ms=close_ts_ms,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )
        bus.publish(ev)
        ctx.state._candle_count += 1
        now = time.time()
        if now - ctx.state._last_candle_log >= 10:
            log.info("candles emitted in last 10s=%d", ctx.state._candle_count)
            ctx.state._candle_count = 0
            ctx.state._last_candle_log = now

    candle_builder = RollingCandleBuilder(
        interval_s=int(cfg.get("feature_state", {}).get("candle_interval_s", cfg["runtime"]["candle_interval_s"])),
        emit_fn=lambda ev: emit_candle(ev.symbol, ev.ts_ms, ev),
    )

    # Exchange clients
    bcfg = cfg.get("binance", {})
    rest = BinanceRestClient(
        data_rest_base_url=bcfg.get("data_rest_base_url", "https://api.binance.us/api"),
        exec_rest_base_url=bcfg.get("exec_rest_base_url", "https://testnet.binance.vision/api"),
        exchange_info_url=bcfg.get("exchange_info_url", "https://api.binance.us/api/v3/exchangeInfo"),
    )
    log.info("DATA base_url=%s EXEC base_url=%s", rest.data_rest_base_url, rest.exec_rest_base_url)
    ctx.state._book_count = 0
    ctx.state._last_book_log = time.time()
    # Symbol registry -> updates WS subscriptions
    registry_cfg = cfg.get("symbol_registry", {})
    symbol_registry = SymbolRegistry(rest=rest, cfg=registry_cfg, bus=bus)
    market_data_cfg = cfg.get("market_data", {}) or {}
    ws_client = LiveMarketDataClient(cfg=cfg, bus=bus, on_book=lambda ev: _on_book(ctx, candle_builder, ev))


    force_cfg = cfg.get("execution_live", {})
    forced_test_order_sent = False
    forced_test_entry_ts_s = None
    forced_test_exit_sent = False

    async def rest_poll_loop():
        nonlocal forced_test_order_sent, forced_test_entry_ts_s, forced_test_exit_sent
        interval = float(cfg.get("rest_poll", {}).get("interval_s", 0.5))
        endpoint = cfg.get("rest_poll", {}).get("endpoint", "ticker_book")
        symbols: Set[str] = set()

        async def updater():
            nonlocal symbols
            async for symset in symbol_registry.run():
                symbols = set(symset)
                ctx.state.active_symbols = set(symset)
                if str(market_data_cfg.get("mode", cfg.get("data_source", "rest_poll"))).lower() in {"ws", "hybrid"}:
                    await ws_client.set_symbols(set(symset))
                log.info("REST poll universe symbols=%d", len(symbols))

        asyncio.create_task(updater(), name="symbol_updater")

        last_err_log = 0.0
        backoff = 0.0

        while True:
            if not symbols:
                await asyncio.sleep(1.0)
                continue

            if backoff > 0:
                await asyncio.sleep(backoff)

            any_429 = False

            for s in list(symbols):
                try:
                    ts_ms = int(time.time() * 1000)

                    if endpoint == "ticker_price":
                        data = await rest.ticker_price(s)
                        px = float(data["price"])
                        bid = px
                        ask = px
                    else:
                        data = await rest.ticker_book(s)
                        bid = float(data["bidPrice"])
                        ask = float(data["askPrice"])

                    be = BookEvent(symbol=s, ts_ms=ts_ms, bid=bid, ask=ask)
                    _on_book(ctx, candle_builder, be)
                    ctx.state._poll_ok += 1

                    if bool(force_cfg.get("force_test_order_enabled", False)) and not forced_test_order_sent:
                        test_symbol = str(force_cfg.get("force_test_order_symbol", "ETHUSDT") or "ETHUSDT")
                        if s == test_symbol:
                            ms = ctx.state.market.get(test_symbol)
                            if ms and ms.bid is not None and ms.ask is not None:
                                forced_test_order_sent = True
                                qty = float(force_cfg.get("force_test_order_qty", 0.001) or 0.001)
                                premium_bps = float(force_cfg.get("force_test_order_premium_bps", 10.0) or 10.0)
                                limit_px = float(ms.ask) * (1.0 + premium_bps / 10000.0)
                                test_intent = OrderIntent(
                                    side="BUY",
                                    symbol=test_symbol,
                                    qty=qty,
                                    limit_px=limit_px,
                                    tif=str(force_cfg.get("default_tif", "IOC") or "IOC"),
                                )
                                log.info(
                                    "FORCE TEST ORDER routing_through_engine symbol=%s qty=%.8f limit_px=%.8f premium_bps=%.2f tif=%s",
                                    test_symbol,
                                    qty,
                                    limit_px,
                                    premium_bps,
                                    test_intent.tif,
                                )
                                engine._handle_strategy_outputs([test_intent], "force_test_order", trigger_event=be)
                                forced_test_entry_ts_s = time.time()

                    if (
                        bool(force_cfg.get("force_test_auto_exit_enabled", False))
                        and forced_test_order_sent
                        and not forced_test_exit_sent
                        and forced_test_entry_ts_s is not None
                    ):
                        exit_delay_s = float(force_cfg.get("force_test_auto_exit_delay_s", 15.0) or 15.0)
                        if time.time() - float(forced_test_entry_ts_s) >= exit_delay_s:
                            test_symbol = str(force_cfg.get("force_test_order_symbol", "ETHUSDT") or "ETHUSDT")
                            venue_symbol = test_symbol
                            router = getattr(broker, "router", None)
                            if router is not None and hasattr(router, "to_venue_symbol"):
                                try:
                                    venue_symbol = router.to_venue_symbol(test_symbol)
                                except Exception:
                                    venue_symbol = test_symbol

                            positions = getattr(broker, "positions", {}) or {}
                            pos = positions.get(venue_symbol) or positions.get(test_symbol)
                            pos_qty = float(getattr(pos, "qty", 0.0) or 0.0) if pos is not None else 0.0
                            if pos_qty > 0.0:
                                ms = ctx.state.market.get(test_symbol)
                                if ms and ms.bid is not None and ms.bid > 0:
                                    exit_discount_bps = float(
                                        force_cfg.get(
                                            "force_test_auto_exit_discount_bps",
                                            force_cfg.get("force_test_order_premium_bps", 10.0),
                                        ) or 10.0
                                    )
                                    exit_limit_px = float(ms.bid) * (1.0 - exit_discount_bps / 10_000.0)
                                    forced_test_exit_sent = True
                                    exit_intent = OrderIntent(
                                        side="SELL",
                                        symbol=venue_symbol,
                                        qty=pos_qty,
                                        limit_px=exit_limit_px,
                                        tif=str(force_cfg.get("default_tif", "IOC") or "IOC"),
                                    )
                                    log.info(
                                        "FORCE TEST AUTO EXIT routing_through_engine source_symbol=%s venue_symbol=%s qty=%.8f limit_px=%.8f discount_bps=%.2f tif=%s",
                                        test_symbol,
                                        venue_symbol,
                                        pos_qty,
                                        exit_limit_px,
                                        exit_discount_bps,
                                        exit_intent.tif,
                                    )
                                    engine._handle_strategy_outputs([exit_intent], "force_test_auto_exit", trigger_event=be)

                except httpx.HTTPStatusError as e:
                    ctx.state._poll_err += 1
                    ctx.state.api_error_counts = int(getattr(ctx.state, "api_error_counts", 0) or 0) + 1
                    status = e.response.status_code if e.response is not None else None
                    if status in (418, 429):
                        any_429 = True
                    now = time.time()
                    if now - last_err_log > 5:
                        log.warning("REST poll HTTP error symbol=%s status=%s", s, status)
                        last_err_log = now
                    continue

                except Exception as e:
                    ctx.state._poll_err += 1
                    ctx.state.api_error_counts = int(getattr(ctx.state, "api_error_counts", 0) or 0) + 1
                    now = time.time()
                    if now - last_err_log > 5:
                        log.warning("REST poll failed symbol=%s err=%s", s, e)
                        last_err_log = now
                    continue

            # Backoff policy
            if any_429:
                backoff = min(10.0, backoff * 2.0 if backoff > 0 else 1.0)
            else:
                backoff = max(0.0, backoff - 0.5)

            ctx.state._poll_backoff_s = backoff
            await asyncio.sleep(interval)


    async def status_loop():
        while True:
            sym_count = len(getattr(ctx.state, "active_symbols", set()))

            unrealized = broker.unrealized_pnl(ctx.state)
            equity = broker.equity(ctx.state)

            m = broker.metrics_snapshot()
            log.info(
                "cash=%.2f equity=%.2f pnl=%.2f trades=%d W/L=%d/%d avg_hold=%.1fs symbols=%d poll_ok=%d poll_err=%d backoff=%.1fs",
                broker.cash,
                equity,
                m["realized_pnl"],
                m["trades"],
                m["wins"],
                m["losses"],
                m["avg_hold_s"],
                sym_count,
                getattr(ctx.state, "_poll_ok", 0),
                getattr(ctx.state, "_poll_err", 0),
                getattr(ctx.state, "_poll_backoff_s", 0.0),
            )
            ctx.store.insert_equity_snapshot(
                ts_ms=int(time.time() * 1000),
                cash=broker.cash,
                realized_pnl=broker.realized_pnl,
                unrealized_pnl=unrealized,
                equity=equity,
                mode="live",
            )
            await asyncio.sleep(5)


    async def safety_loop():
        interval_s = float(cfg.get("safety", {}).get("health_loop_interval_s", 5.0) or 5.0)
        while True:
            try:
                symbols = sorted(getattr(ctx.state, "active_symbols", set()) or [])
                stale_snapshot = stale_guard.snapshot(symbols=symbols)
                ctx.state.stale_symbols = list(stale_snapshot.stale_symbols)
                ctx.state.stale_global = bool(stale_snapshot.stale_global)

                results = health_monitor.evaluate(ctx)
                worst = health_monitor.worst_status(results)
                if worst == STATUS_KILL_SWITCH:
                    kill_switch.activate(KillSwitchReason.HEALTH_RULE, metadata={"health": [r.to_dict() for r in results]})
                elif worst == STATUS_SAFE_MODE:
                    safe_mode.activate("health_rule", metadata={"health": [r.to_dict() for r in results]})
                else:
                    safe_mode.maybe_auto_recover()

                if kill_switch.is_active() and kill_switch.flatten_positions_on_trigger and hasattr(broker, "close_all"):
                    try:
                        broker.close_all(ctx, reason="kill_switch")
                    except TypeError:
                        broker.close_all(ctx)

                ctx.state.set_runtime_safety_snapshot({
                    "safe_mode": safe_mode.is_active(),
                    "kill_switch": kill_switch.is_active(),
                    "health_status": worst,
                    "health_messages": health_monitor.messages(results),
                    "stale_symbols": stale_snapshot.stale_symbols,
                    "stale_global": stale_snapshot.stale_global,
                })
                store.insert_safety_snapshot(
                    safe_mode=safe_mode.is_active(),
                    kill_switch=kill_switch.is_active(),
                    reasons=list(kill_switch.active_reasons()) + list(safe_mode.state.reasons),
                    metadata={"health_status": worst, "stale_symbols": stale_snapshot.stale_symbols},
                )
            except Exception:
                log.exception("SAFETY_LOOP_FAILED")
                safe_mode.activate("safety_loop_failed")
            await asyncio.sleep(interval_s)

    tasks = [
        asyncio.create_task(bus.run(), name="event_bus"),
        asyncio.create_task(status_loop(), name="status"),
        asyncio.create_task(safety_loop(), name="safety"),
    ]

    md_mode = str(market_data_cfg.get("mode", cfg.get("data_source", "rest_poll")) or "rest_poll").lower()
    rest_fallback = bool(market_data_cfg.get("rest_fallback", True))
    if md_mode in {"rest_poll", "hybrid"} or rest_fallback:
        tasks.append(asyncio.create_task(rest_poll_loop(), name="rest_poll"))
    if md_mode in {"ws", "hybrid"}:
        tasks.append(asyncio.create_task(ws_client.run_forever(), name="ws_market_data"))

    shutdown_started = False

    async def _shutdown(reason: str) -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True

        log.warning("SHUTDOWN_START reason=%s flattening_positions", reason)

        flatten_on_shutdown = bool(
            cfg.get("execution_live", {}).get("flatten_on_shutdown", True)
        )

        if flatten_on_shutdown and hasattr(broker, "close_all"):
            try:
                positions = getattr(broker, "positions", {}) or {}
                log.warning(
                    "CLOSE_ALL_REQUESTED reason=%s open_positions=%s",
                    reason,
                    list(positions.keys()),
                )
                try:
                    broker.close_all(ctx, reason=reason)
                except TypeError:
                    try:
                        broker.close_all(ctx)
                    except TypeError:
                        broker.close_all()

                log.warning("CLOSE_ALL_COMPLETE reason=%s", reason)

            except Exception:
                    log.exception("CLOSE_ALL_FAILED reason=%s", reason)
        else:
            log.warning(
                "CLOSE_ALL_SKIPPED reason=%s flatten_on_shutdown=%s has_close_all=%s",
                reason,
                flatten_on_shutdown,
                hasattr(broker, "close_all"),
            )

        for t in tasks:
            if not t.done():
                t.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        log.warning("SHUTDOWN_COMPLETE reason=%s", reason)

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        await _shutdown("KEYBOARD_INTERRUPT")
    except asyncio.CancelledError:
        await _shutdown("CANCELLED")
        raise
    finally:
        # In asyncio.run(), Ctrl+C commonly cancels the main task before
        # KeyboardInterrupt is visible inside this coroutine. The finally block
        # is the reliable place to flatten positions for manual-stop tests.
        if not shutdown_started:
            await _shutdown("FINALLY")




def _on_candle_for_feature_state(ctx: Ctx, bus: EventBus, ev: CandleEvent) -> None:
    feature_state = getattr(ctx.state, "feature_state", None)
    if feature_state is None:
        return

    feature_state.update_candle(ev)
    stale_guard = getattr(ctx, "stale_guard", None)
    if stale_guard is not None:
        stale_guard.update_candle(ev.symbol, int(ev.ts_ms))
        stale_guard.update_feature(ev.symbol, int(ev.ts_ms))
    ctx.state.last_candle_ts_ms = int(ev.ts_ms)
    ctx.state.last_feature_update_ts_ms = int(ev.ts_ms)
    health = feature_state.health_snapshot()
    ready = feature_state.ready_symbols()
    health["ready_symbol_list"] = ready

    if hasattr(ctx.state, "set_feature_state_health"):
        ctx.state.set_feature_state_health(health)
    else:
        ctx.state.feature_state_health = health
        ctx.state.rolling_ready_symbols = ready

    regime = feature_state.compute_regime(ready_only=True)
    if hasattr(ctx.state, "set_live_regime_snapshot"):
        ctx.state.set_live_regime_snapshot(regime)
    else:
        ctx.state.live_regime_snapshot = regime

    snapshot_metadata = {
        "ts_ms": int(ev.ts_ms),
        "symbols": feature_state.symbols(),
        "ready_symbols": ready,
        "feature_state": health,
    }
    if hasattr(ctx.state, "set_live_context_snapshot_metadata"):
        ctx.state.set_live_context_snapshot_metadata(snapshot_metadata)
    else:
        ctx.state.latest_context_snapshot_metadata = snapshot_metadata

    bus.publish(FeatureStateUpdatedEvent(ts_ms=int(ev.ts_ms), ready_symbols=ready, health=health))
    bus.publish(RegimeContextEvent(ts_ms=int(ev.ts_ms), regime=regime))
    bus.publish(
        LiveContextSnapshotEvent(
            ts_ms=int(ev.ts_ms),
            ready_symbols=ready,
            feature_rows=len(ready),
            regime=regime,
            metadata=snapshot_metadata,
        )
    )

def _on_book(ctx: Ctx, candle_builder, book_event) -> None:
    """
    book_event provides best bid/ask. We'll build 1s candles from mid.
    """
    ms = ctx.state.market.setdefault(book_event.symbol, ctx.state.market_state_factory())
    ms.bid = book_event.bid
    ms.ask = book_event.ask
    ms.last_ts_ms = book_event.ts_ms

    mid = (book_event.bid + book_event.ask) / 2.0
    ms.last = mid

    # Feed candle builder (volume=0 for bookTicker-based candles)
    candle_builder.on_price(book_event.symbol, book_event.ts_ms, mid, qty=0.0)

    # Persist latest book snapshot (optional)
    ctx.store.upsert_book(book_event.symbol, book_event.ts_ms, book_event.bid, book_event.ask)

    stale_guard = getattr(ctx, "stale_guard", None)
    if stale_guard is not None:
        stale_guard.update_book(book_event.symbol, int(book_event.ts_ms))
        stale_guard.update_ws_heartbeat(int(book_event.ts_ms))
    ctx.state.last_market_data_ts_ms = int(book_event.ts_ms)
    ctx.state.last_book_ts_ms = int(book_event.ts_ms)

    ctx.state.market_data_health = {
        "last_book_ts_ms": int(book_event.ts_ms),
        "last_symbol": book_event.symbol,
        "book_count": getattr(ctx.state, "_book_count", 0) + 1,
    }

    ctx.state._book_count += 1
    now = time.time()
    if now - ctx.state._last_book_log >= 5:
        logging.getLogger("tradarbot").info(
            "book events=%d last=%s bid=%s ask=%s",
            ctx.state._book_count,
            book_event.symbol,
            book_event.bid,
            book_event.ask,
        )
        ctx.state._book_count = 0
        ctx.state._last_book_log = now


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
