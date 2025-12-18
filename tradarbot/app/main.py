import asyncio
import logging
from typing import Any, Dict, Set

import yaml

from tradarbot.app.context import Ctx
from tradarbot.core.bus import EventBus
from tradarbot.core.engine import StrategyEngine
from tradarbot.core.events import CandleEvent, ListingEvent
from tradarbot.core.state import State
from tradarbot.data.candles import CandleBuilder1s
from tradarbot.data.symbol_registry import SymbolRegistry
from tradarbot.exchange.binance.rest import BinanceRestClient
from tradarbot.exchange.binance.ws_manager import BinanceWSManager
from tradarbot.execution.paper_broker import PaperBroker
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.strategies.algo2_micro_momentum import Algo2MicroMomentum


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def main() -> None:
    cfg: Dict[str, Any] = yaml.safe_load(open("config/tradar.yaml", "r"))
    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
    log = logging.getLogger("tradarbot")
    log.info("Booting...")

    # Core components
    state = State()
    store = SQLiteStore("tradarbot.db")
    store.init_schema()

    broker = PaperBroker(
        fee_bps=float(cfg["execution"]["fee_bps"]),
        starting_cash=10_000.0
    )
    risk = RiskManager(cfg)

    ctx = Ctx(cfg=cfg, state=state, store=store, broker=broker, risk=risk)

    bus = EventBus()

    # Strategies
    strategies = []
    s_cfg = cfg.get("strategies", {}).get("algo2_micro_momentum", {})
    if s_cfg.get("enabled", False):
        strategies.append(Algo2MicroMomentum(s_cfg))

    engine = StrategyEngine(strategies=strategies, risk=risk, broker=broker, ctx=ctx)

    # Route events
    bus.subscribe(CandleEvent, engine.on_candle)
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

    candle_builder = CandleBuilder1s(emit_fn=emit_candle)

    # Exchange clients
    rest = BinanceRestClient()
    ws_mgr = BinanceWSManager(cfg=cfg, on_book=lambda be: _on_book(ctx, candle_builder, be))

    # Symbol registry -> updates WS subscriptions
    registry_cfg = cfg.get("symbol_registry", {})
    symbol_registry = SymbolRegistry(rest=rest, cfg=registry_cfg, bus=bus)

    async def registry_loop():
        async for symbols in symbol_registry.run():
            await ws_mgr.set_symbols(symbols)

    async def ws_loop():
        await ws_mgr.run_forever()

    async def status_loop():
        while True:
            log.info("cash=%.2f positions=%s symbols=%d",
                     broker.cash, broker.positions_snapshot(), len(ws_mgr.current_symbols))
            await asyncio.sleep(5)

    tasks = [
        asyncio.create_task(bus.run(), name="event_bus"),
        asyncio.create_task(registry_loop(), name="symbol_registry"),
        asyncio.create_task(ws_loop(), name="ws_manager"),
        asyncio.create_task(status_loop(), name="status"),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        log.warning("Shutting down...")
        for t in tasks:
            t.cancel()


def _on_book(ctx: Ctx, candle_builder: CandleBuilder1s, book_event) -> None:
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
