import argparse
import sqlite3
import time
import yaml

from tradarbot.app.context import Ctx
from tradarbot.core.engine import StrategyEngine
from tradarbot.core.events import CandleEvent
from tradarbot.core.state import State
from tradarbot.execution.paper_broker import PaperBroker
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore
from tradarbot.strategies.algo2_micro_momentum import Algo2MicroMomentum


def load_candles(db_path: str, symbol: str, interval_s: int, start_ms: int, end_ms: int):
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
    for row in cur.fetchall():
        ts_ms, o, h, l, c, v = row
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


def mark_to_market(broker: PaperBroker, state: State):
    equity = broker.cash
    for sym, pos in broker.positions.items():
        ms = state.market.get(sym)
        if ms and ms.bid is not None and ms.ask is not None:
            mid = (ms.bid + ms.ask) / 2.0
            equity += pos.qty * mid
    return equity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tradar.yaml")
    ap.add_argument("--db", default="tradarbot.db")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start_ms", type=int, required=True)
    ap.add_argument("--end_ms", type=int, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    state = State()
    store = SQLiteStore(args.db)
    broker = PaperBroker(fee_bps=float(cfg["execution"]["fee_bps"]), starting_cash=10_000.0)
    risk = RiskManager(cfg)
    ctx = Ctx(cfg=cfg, state=state, store=store, broker=broker, risk=risk)

    # strategy selection (extend later)
    strategies = []
    s_cfg = cfg.get("strategies", {}).get("algo2_micro_momentum", {})
    if s_cfg.get("enabled", False):
        strategies.append(Algo2MicroMomentum(s_cfg))

    engine = StrategyEngine(strategies=strategies, risk=risk, broker=broker, ctx=ctx)

    interval_s = int(cfg["runtime"]["candle_interval_s"])

    # basic stats
    start_equity = broker.cash
    candles = 0

    t0 = time.time()
    for ev in load_candles(args.db, args.symbol, interval_s, args.start_ms, args.end_ms):
        # update "market" so broker can fill
        ms = state.market.setdefault(ev.symbol, state.market_state_factory())
        mid = ev.close
        spread_bps = 1.0
        half = (spread_bps / 10_000.0) * mid / 2.0
        ms.bid = mid - half
        ms.ask = mid + half     
        ms.last = ev.close
        ms.last_ts_ms = ev.ts_ms

        engine.on_candle(ev)
        candles += 1

    elapsed = time.time() - t0
    end_equity = mark_to_market(broker, state)

    print("----- REPLAY SUMMARY -----")
    print(f"symbol: {args.symbol}")
    print(f"candles: {candles}")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"start_equity: {start_equity:.2f}")
    print(f"end_equity:   {end_equity:.2f}")
    print(f"pnl:          {end_equity - start_equity:.2f}")
    print(f"positions:    {broker.positions_snapshot()}")


if __name__ == "__main__":
    main()
