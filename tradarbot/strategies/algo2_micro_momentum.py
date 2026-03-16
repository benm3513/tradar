import logging
from typing import Dict, List, Optional

from tradarbot.core.events import CandleEvent, OrderIntent

log = logging.getLogger("tradarbot.algo2")


class Algo2MicroMomentum:

    name = "algo2_micro_momentum"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.closes: Dict[str, List[float]] = {}
        self.last_close: Dict[str, float] = {}

        # approximate entry timestamp: generate BUY signal
        self.entry_ts_ms: Dict[str, int] = {}

    @staticmethod
    def _bps_to_frac(bps: float) -> float:
        return float(bps) / 10_000.0

    def on_listing(self, ev, ctx):
        return []

    def on_candle(self, e: CandleEvent, ctx):
        sym = e.symbol

        arr = self.closes.get(sym, [])
        arr.append(e.close)
        if len(arr) > 3:
            arr = arr[-3:]
        self.closes[sym] = arr

        pos = ctx.broker.positions.get(sym)
        if pos and pos.qty > 0:
            intents = self._manage_exit(sym, e, ctx, pos)
            self.last_close[sym] = e.close
            return intents

        # optional price cap
        price_cap = self.cfg.get("price_cap", None)
        if price_cap is not None and e.close >= float(price_cap):
            self.last_close[sym] = e.close
            return []

        if len(arr) < 3:
            self.last_close[sym] = e.close
            return []

        min_move_bps = float(self.cfg.get("min_move_bps", 2))
        min_move = max(1e-12, self._bps_to_frac(min_move_bps) * e.close)

        up1 = arr[1] > arr[0] + min_move
        up2 = arr[2] > arr[1] + min_move
        if not (up1 and up2):
            self.last_close[sym] = e.close
            return []

        ms = ctx.state.market.get(sym)
        if not ms or ms.ask is None:
            self.last_close[sym] = e.close
            return []

        notional_fraction = float(self.cfg.get("notional_fraction", 0.05))
        qty_usd = ctx.broker.cash * notional_fraction
        qty = qty_usd / max(ms.ask, 1e-12)

        limit_px = ms.ask * (1.0 + float(ctx.cfg["execution"]["entry_slippage_cap_pct"]))

        log.info("DEBUG ENTRY_OK %s closes=%s min_move_bps=%s",sym,arr,min_move_bps,)
        log.info("SIGNAL BUY %s close=%.2f ask=%.2f qty=%.8f", sym, e.close, ms.ask, qty)

        # track "entry time" at signal time (approx)
        self.entry_ts_ms[sym] = int(e.ts_ms)

        self.last_close[sym] = e.close
        log.info("DEBUG ENTRY_OK %s closes=%s min_move_bps=%s", sym, arr, min_move_bps)
        return [OrderIntent("BUY", sym, qty, limit_px)]

    def _manage_exit(self, sym, e: CandleEvent, ctx, pos):
        ms = ctx.state.market.get(sym)
        if not ms or ms.bid is None:
            return []

        tp_bps = float(self.cfg.get("take_profit_bps", 10))
        sl_bps = float(self.cfg.get("stop_bps", 10))
        tp = self._bps_to_frac(tp_bps)
        sl = self._bps_to_frac(sl_bps)

        min_move_bps = float(self.cfg.get("min_move_bps", 2))
        min_move = max(1e-12, self._bps_to_frac(min_move_bps) * e.close)

        prev = self.last_close.get(sym, None)

        # time stop
        max_hold_s = self.cfg.get("max_hold_s", None)
        if max_hold_s is not None:
            entry_ts = self.entry_ts_ms.get(sym)
            if entry_ts is not None:
                held_s = (int(e.ts_ms) - int(entry_ts)) / 1000.0
                if held_s >= float(max_hold_s):
                    return self._sell(sym, e, ctx, pos, f"TIME_{int(max_hold_s)}s")

        if e.close >= pos.avg_px * (1.0 + tp):
            return self._sell(sym, e, ctx, pos, f"TP_{tp_bps:.0f}bps")

        if e.close <= pos.avg_px * (1.0 - sl):
            return self._sell(sym, e, ctx, pos, f"SL_{sl_bps:.0f}bps")

        if prev is not None and e.close < prev - min_move:
            return self._sell(sym, e, ctx, pos, "REVERSAL_1TICK_DOWN")

        return []

    def _sell(self, sym, e: CandleEvent, ctx, pos, reason: str):
        ms = ctx.state.market.get(sym)
        limit_px = ms.bid * (1.0 - float(ctx.cfg["execution"]["exit_slippage_cap_pct"]))
        log.info(
            "SIGNAL SELL %s reason=%s close=%.2f bid=%.2f qty=%.8f",
            sym, reason, e.close, ms.bid, pos.qty
        )
        # cooldown on any exit
        ctx.risk.set_cooldown(sym, int(ctx.cfg["risk"]["cooldown_s"]))
        return [OrderIntent("SELL", sym, pos.qty, limit_px)]
