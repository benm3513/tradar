import logging
from typing import Dict, List

from tradarbot.core.events import CandleEvent, OrderIntent

log = logging.getLogger("tradarbot.algo2")


class Algo2MicroMomentum:
    """
    Tick definition: 1-second candle closes.
    Entry: 2 consecutive up-ticks (close rises by >= min_move_bps each second).
    Exit:
      - TP/SL in bps from avg entry
      - Reversal: 1 down-tick of >= min_move_bps after entry
    """
    name = "algo2_micro_momentum"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.closes: Dict[str, List[float]] = {}     # last 3 closes
        self.last_close: Dict[str, float] = {}       # previous close for reversal logic

    @staticmethod
    def _bps_to_frac(bps: float) -> float:
        return float(bps) / 10_000.0

    def on_listing(self, ev, ctx):
        return []

    def on_candle(self, e: CandleEvent, ctx):
        sym = e.symbol

        # Track last closes
        arr = self.closes.get(sym, [])
        arr.append(e.close)
        if len(arr) > 3:
            arr = arr[-3:]
        self.closes[sym] = arr

        # If position open, manage exit
        pos = ctx.broker.positions.get(sym)
        if pos and pos.qty > 0:
            intents = self._manage_exit(sym, e, ctx, pos)
            self.last_close[sym] = e.close
            return intents

        # Entry price filter (optional)
        price_cap = self.cfg.get("price_cap", None)
        if price_cap is not None and e.close >= float(price_cap):
            self.last_close[sym] = e.close
            return []

        if len(arr) < 3:
            self.last_close[sym] = e.close
            return []

        # min_move_bps is TRUE bps now
        min_move_frac = self._bps_to_frac(float(self.cfg.get("min_move_bps", 2)))
        min_move = max(1e-12, min_move_frac * e.close)

        up1 = arr[1] > arr[0] + min_move
        up2 = arr[2] > arr[1] + min_move
        if not (up1 and up2):
            self.last_close[sym] = e.close
            return []

        ms = ctx.state.market.get(sym)
        if not ms or ms.ask is None:
            self.last_close[sym] = e.close
            return []

        # Position sizing: fraction of cash
        notional_fraction = float(self.cfg.get("notional_fraction", 0.05))
        qty_usd = ctx.broker.cash * notional_fraction
        qty = qty_usd / max(ms.ask, 1e-12)

        limit_px = ms.ask * (1.0 + float(ctx.cfg["execution"]["entry_slippage_cap_pct"]))

        log.info("SIGNAL BUY %s close=%.2f ask=%.2f qty=%.8f", sym, e.close, ms.ask, qty)
        self.last_close[sym] = e.close
        return [OrderIntent("BUY", sym, qty, limit_px)]

    def _manage_exit(self, sym, e, ctx, pos):
        ms = ctx.state.market.get(sym)
        if not ms or ms.bid is None:
            return []

        # TP/SL in TRUE bps
        tp_bps = float(self.cfg.get("take_profit_bps", 10))
        sl_bps = float(self.cfg.get("stop_bps", 10))
        tp = self._bps_to_frac(tp_bps)
        sl = self._bps_to_frac(sl_bps)

        # reversal: 1 down-tick >= min_move_bps
        min_move_frac = self._bps_to_frac(float(self.cfg.get("min_move_bps", 2)))
        min_move = max(1e-12, min_move_frac * e.close)
        prev = self.last_close.get(sym, None)

        # helper: generate sell intent
        def sell(reason: str):
            limit_px = ms.bid * (1.0 - float(ctx.cfg["execution"]["exit_slippage_cap_pct"]))
            log.info("SIGNAL SELL %s reason=%s close=%.2f bid=%.2f qty=%.8f",
                     sym, reason, e.close, ms.bid, pos.qty)
            ctx.risk.set_cooldown(sym, int(ctx.cfg["risk"]["cooldown_s"]))
            return [OrderIntent("SELL", sym, pos.qty, limit_px)]

        if e.close >= pos.avg_px * (1.0 + tp):
            return sell(f"TP_{tp_bps:.0f}bps")

        if e.close <= pos.avg_px * (1.0 - sl):
            return sell(f"SL_{sl_bps:.0f}bps")

        if prev is not None and e.close < prev - min_move:
            return sell("REVERSAL_1TICK_DOWN")

        return []
