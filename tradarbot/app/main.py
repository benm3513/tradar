import asyncio
import logging
import time
import httpx
from typing import Any, Dict, Set

import yaml

from tradarbot.app.context import Ctx
from tradarbot.core.bus import EventBus
from tradarbot.core.engine import StrategyEngine
from tradarbot.core.events import CandleEvent, ListingEvent, BookEvent
from tradarbot.core.state import State
from tradarbot.data.candles import CandleBuilder1s
from tradarbot.data.symbol_registry import SymbolRegistry
from tradarbot.exchange.binance.rest import BinanceRestClient
from tradarbot.exchange.binance.ws_manager import BinanceWSManager
from tradarbot.execution.paper_broker import PaperBroker
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.strategies.algo2_micro_momentum import Algo2MicroMomentum
from tradarbot.strategies.algo1_new_listing_pump import Algo1NewListingPump


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
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
        ctx.state._candle_count += 1
        now = time.time()
        if now - ctx.state._last_candle_log >= 10:
            log.info("candles emitted in last 10s=%d", ctx.state._candle_count)
            ctx.state._candle_count = 0
            ctx.state._last_candle_log = now

    candle_builder = CandleBuilder1s(emit_fn=emit_candle)

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


    async def rest_poll_loop():
        interval = float(cfg.get("rest_poll", {}).get("interval_s", 0.5))
        endpoint = cfg.get("rest_poll", {}).get("endpoint", "ticker_book")
        symbols: Set[str] = set()

        async def updater():
            nonlocal symbols
            async for symset in symbol_registry.run():
                symbols = set(symset)
                ctx.state.active_symbols = set(symset)
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

                except httpx.HTTPStatusError as e:
                    ctx.state._poll_err += 1
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


    tasks = [
        asyncio.create_task(bus.run(), name="event_bus"),
        asyncio.create_task(rest_poll_loop(), name="rest_poll"),
        asyncio.create_task(status_loop(), name="status"),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        log.warning("Shutting down... flattening positions")
        broker.close_all(ctx, reason="SHUTDOWN")

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)



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
