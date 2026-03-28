import argparse
import heapq
import sqlite3
import time
from typing import Dict, Iterator, List, Tuple

import yaml

from tradarbot.app.context import Ctx
from tradarbot.core.engine import StrategyEngine
from tradarbot.core.events import CandleEvent, ListingEvent
from tradarbot.core.state import State
from tradarbot.execution.paper_broker import PaperBroker
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.strategies.algo1_new_listing_pump import Algo1NewListingPump
from tradarbot.strategies.algo2_micro_momentum import Algo2MicroMomentum


def list_symbols_in_db(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")
    out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


def iter_candles(
    db_path: str,
    symbol: str,
    interval_s: int,
    start_ms: int,
    end_ms: int,
) -> Iterator[CandleEvent]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts_ms, open, high, low, close, volume
        FROM candles
        WHERE symbol=? AND interval_s=? AND ts_ms BETWEEN ? AND ?
        ORDER BY ts_ms ASC
        """,
        (symbol, interval_s, start_ms, end_ms),
    )
    rows = cur.fetchall()
    conn.close()

    for ts_ms, o, h, l, c, v in rows:
        yield CandleEvent(
            symbol=symbol,
            interval_s=interval_s,
            ts_ms=ts_ms,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
        )


def merge_streams(streams: Dict[str, Iterator[CandleEvent]]) -> Iterator[CandleEvent]:
    """
    K-way merge on ts_ms (ties broken by symbol).
    """
    heap: List[Tuple[int, str, CandleEvent]] = []
    for sym, it in streams.items():
        try:
            ev = next(it)
            heapq.heappush(heap, (int(ev.ts_ms), sym, ev))
        except StopIteration:
            pass

    while heap:
        _, sym, ev = heapq.heappop(heap)
        yield ev
        it = streams[sym]
        try:
            nxt = next(it)
            heapq.heappush(heap, (int(nxt.ts_ms), sym, nxt))
        except StopIteration:
            pass


def run_replay(
    config_path: str,
    db_path: str,
    symbols: List[str],
    start_ms: int,
    end_ms: int,
    liquidate_end: bool = False,
    synthetic_spread_bps: float = 1.0,
    persist_equity: bool = False,
):
    cfg = yaml.safe_load(open(config_path, "r"))

    state = State()
    store = SQLiteStore(db_path)
    store.init_schema()

    exec_cfg = cfg.get("execution", {}) or {}
    fee_bps = exec_cfg.get("fee_bps", None)
    if fee_bps is None:
        fee_bps = (exec_cfg.get("fees", {}) or {}).get("fee_bps", None)
    if fee_bps is None:
        fee_bps = 10

    broker = PaperBroker(fee_bps=float(fee_bps), starting_cash=10_000.0)
    risk = RiskManager(cfg)
    ctx = Ctx(cfg=cfg, state=state, store=store, broker=broker, risk=risk)

    strategies = []

    algo1_cfg = cfg.get("strategies", {}).get("algo1_new_listing_pump", {})
    if algo1_cfg.get("enabled", False):
        strategies.append(Algo1NewListingPump(algo1_cfg))

    algo2_cfg = cfg.get("strategies", {}).get("algo2_micro_momentum", {})
    if algo2_cfg.get("enabled", False):
        strategies.append(Algo2MicroMomentum(algo2_cfg))

    engine = StrategyEngine(strategies=strategies, risk=risk, broker=broker, ctx=ctx)
    interval_s = int(cfg["runtime"]["candle_interval_s"])

    streams = {
        sym: iter_candles(db_path, sym, interval_s, start_ms, end_ms)
        for sym in symbols
    }

    start_equity = broker.cash
    peak_equity = start_equity
    max_drawdown = 0.0
    candles = 0
    last_ts_ms = None
    equity_points = []

    t0 = time.time()
    for ev in merge_streams(streams):
        last_ts_ms = int(ev.ts_ms)
        candles += 1

        ms = state.market.setdefault(ev.symbol, state.market_state_factory())
        mid = float(ev.close)
        half = (synthetic_spread_bps / 10_000.0) * mid / 2.0
        ms.bid = mid - half
        ms.ask = mid + half
        ms.last = mid
        ms.last_ts_ms = int(ev.ts_ms)

        if ev.symbol not in state.listings:
            engine.on_listing(ListingEvent(symbol=ev.symbol, ts_ms=int(ev.ts_ms)))

        engine.on_candle(ev)

        unrealized = broker.unrealized_pnl(state)
        equity = broker.equity(state)
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)

        point = {
            "ts_ms": int(ev.ts_ms),
            "cash": broker.cash,
            "realized_pnl": broker.realized_pnl,
            "unrealized_pnl": unrealized,
            "equity": equity,
            "drawdown": drawdown,
        }
        equity_points.append(point)

        if persist_equity:
            store.insert_equity_snapshot(
                ts_ms=point["ts_ms"],
                cash=point["cash"],
                realized_pnl=point["realized_pnl"],
                unrealized_pnl=point["unrealized_pnl"],
                equity=point["equity"],
                mode="replay",
            )

    if liquidate_end and last_ts_ms is not None:
        broker.close_all(ctx, reason="REPLAY_END")

        unrealized = broker.unrealized_pnl(state)
        equity = broker.equity(state)
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)

        point = {
            "ts_ms": last_ts_ms,
            "cash": broker.cash,
            "realized_pnl": broker.realized_pnl,
            "unrealized_pnl": unrealized,
            "equity": equity,
            "drawdown": drawdown,
        }
        equity_points.append(point)

        if persist_equity:
            store.insert_equity_snapshot(
                ts_ms=point["ts_ms"],
                cash=point["cash"],
                realized_pnl=point["realized_pnl"],
                unrealized_pnl=point["unrealized_pnl"],
                equity=point["equity"],
                mode="replay",
            )

    elapsed = time.time() - t0
    end_equity = broker.equity(state)
    m = broker.metrics_snapshot()

    summary = {
        "symbols": symbols,
        "candles": candles,
        "elapsed_s": elapsed,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "pnl": end_equity - start_equity,
        "return_pct": ((end_equity / start_equity) - 1.0) * 100.0 if start_equity > 0 else 0.0,
        "realized_pnl": m["realized_pnl"],
        "trades": m["trades"],
        "wins": m["wins"],
        "losses": m["losses"],
        "win_rate": m["win_rate"],
        "avg_hold_s": m["avg_hold_s"],
        "worst_trade_streak": m["worst_losing_streak"],
        "max_drawdown": max_drawdown,
        "positions": broker.positions_snapshot(),
        "equity_points": equity_points,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tradar.yaml")
    ap.add_argument("--db", default="tradarbot.db")
    ap.add_argument("--symbols", default=None, help="Comma list e.g. BTCUSDT,ETHUSDT (overrides --all)")
    ap.add_argument("--all", action="store_true", help="Replay all symbols in DB")
    ap.add_argument("--start_ms", type=int, required=True)
    ap.add_argument("--end_ms", type=int, required=True)
    ap.add_argument("--liquidate_end", action="store_true", help="Flatten all positions at end of replay")
    ap.add_argument("--synthetic_spread_bps", type=float, default=1.0, help="Bid/ask synthetic spread for replay")
    ap.add_argument("--persist_equity", action="store_true", help="Persist replay equity snapshots")
    args = ap.parse_args()

    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.all:
        syms = list_symbols_in_db(args.db)
    else:
        raise SystemExit("Provide --symbols or --all")

    summary = run_replay(
        config_path=args.config,
        db_path=args.db,
        symbols=syms,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        liquidate_end=args.liquidate_end,
        synthetic_spread_bps=args.synthetic_spread_bps,
        persist_equity=args.persist_equity,
    )

    print("----- REPLAY SUMMARY -----")
    print(f"symbols: {summary['symbols']}")
    print(f"candles: {summary['candles']}")
    print(f"elapsed: {summary['elapsed_s']:.3f}s")
    print(f"start_equity: {summary['start_equity']:.2f}")
    print(f"end_equity:   {summary['end_equity']:.2f}")
    print(f"pnl:          {summary['pnl']:.2f}")
    print(f"return_pct:   {summary['return_pct']:.2f}%")
    print(f"realized_pnl: {summary['realized_pnl']:.2f}")
    print(f"max_drawdown: {summary['max_drawdown']:.2f}")
    print(
        f"trades:       {summary['trades']}  "
        f"W/L: {summary['wins']}/{summary['losses']}  "
        f"win_rate: {summary['win_rate'] * 100:.1f}%  "
        f"avg_hold_s: {summary['avg_hold_s']:.1f}"
    )
    print(f"worst_trade_streak: {summary['worst_trade_streak']}")
    print(f"positions:    {summary['positions']}")


if __name__ == "__main__":
    main()