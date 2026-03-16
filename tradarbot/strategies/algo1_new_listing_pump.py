import logging
from typing import Dict, List

from tradarbot.core.events import CandleEvent, ListingEvent, OrderIntent

log = logging.getLogger("tradarbot.algo1")


class Algo1NewListingPump:
    name = "algo1_new_listing_pump"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.armed: Dict[str, int] = {}          # symbol -> listing ts_ms
        self.closes: Dict[str, List[float]] = {} # symbol -> recent closes
        self.entry_ts_ms: Dict[str, int] = {}

    @staticmethod
    def _bps_to_frac(bps: float) -> float:
        return float(bps) / 10_000.0

    def on_listing(self, ev: ListingEvent, ctx):
        self.armed[ev.symbol] = int(ev.ts_ms)
        log.info("ALGO1 ARM %s listing_ts=%d", ev.symbol, ev.ts_ms)
        return []

    def on_candle(self, e: CandleEvent, ctx):
        sym = e.symbol

        listing_ts = self.armed.get(sym)
        if listing_ts is None:
            return []

        max_listing_age_s = float(self.cfg.get("max_listing_age_s", 300))
        age_s = (int(e.ts_ms) - int(listing_ts)) / 1000.0
        if age_s < 0 or age_s > max_listing_age_s:
            self.armed.pop(sym, None)
            return []

        arr = self.closes.get(sym, [])
        arr.append(float(e.close))
        if len(arr) > 4:
            arr = arr[-4:]
        self.closes[sym] = arr

        pos = ctx.broker.positions.get(sym)
        if pos and pos.qty > 0:
            return self._manage_exit(sym, e, ctx, pos)

        ms = ctx.state.market.get(sym)
        if not ms or ms.bid is None or ms.ask is None:
            return []

        # spread guard
        spread_pct = (ms.ask - ms.bid) / max(ms.ask, 1e-12)
        max_spread_pct = float(
            self.cfg.get(
                "max_spread_pct",
                ctx.cfg.get("execution", {}).get("max_spread_pct", 0.02)
            )
        )
        if spread_pct > max_spread_pct:
            return []

        # optional price cap
        price_cap = self.cfg.get("price_cap", None)
        if price_cap is not None and e.close >= float(price_cap):
            return []

        if len(arr) < 3:
            return []

        pump_trigger_bps = float(self.cfg.get("pump_trigger_bps", 20))
        trigger_frac = self._bps_to_frac(pump_trigger_bps)

        up1 = arr[-2] > arr[-3]
        up2 = arr[-1] > arr[-2]
        cumulative_move = (arr[-1] / max(arr[-3], 1e-12)) - 1.0

        if not (up1 and up2 and cumulative_move >= trigger_frac):
            return []

        notional_fraction = float(self.cfg.get("notional_fraction", 0.10))
        qty_usd = ctx.broker.cash * notional_fraction
        qty = qty_usd / max(ms.ask, 1e-12)
        if qty <= 0:
            return []

        limit_px = ms.ask * (1.0 + float(ctx.cfg["execution"]["entry_slippage_cap_pct"]))
        self.entry_ts_ms[sym] = int(e.ts_ms)

        log.info(
            "ALGO1 BUY %s close=%.8f ask=%.8f age_s=%.1f cum_move=%.4f qty=%.8f",
            sym, e.close, ms.ask, age_s, cumulative_move, qty
        )
        return [OrderIntent("BUY", sym, qty, limit_px)]

    def _manage_exit(self, sym, e: CandleEvent, ctx, pos):
        ms = ctx.state.market.get(sym)
        if not ms or ms.bid is None:
            return []

        tp_bps = float(self.cfg.get("take_profit_bps", 40))
        sl_bps = float(self.cfg.get("stop_bps", 20))
        exhaustion_reversal_bps = float(self.cfg.get("exhaustion_reversal_bps", 10))
        max_hold_s = float(self.cfg.get("max_hold_s", 120))

        tp = self._bps_to_frac(tp_bps)
        sl = self._bps_to_frac(sl_bps)
        rev = self._bps_to_frac(exhaustion_reversal_bps)

        entry_ts = self.entry_ts_ms.get(sym)
        if entry_ts is not None:
            held_s = (int(e.ts_ms) - int(entry_ts)) / 1000.0
            if held_s >= max_hold_s:
                return self._sell(sym, ctx, pos, "TIME_STOP")

        if e.close >= pos.avg_px * (1.0 + tp):
            return self._sell(sym, ctx, pos, f"TP_{tp_bps:.0f}bps")

        if e.close <= pos.avg_px * (1.0 - sl):
            return self._sell(sym, ctx, pos, f"SL_{sl_bps:.0f}bps")

        arr = self.closes.get(sym, [])
        if len(arr) >= 2:
            peak_ref = max(arr)
            if peak_ref > 0 and e.close <= peak_ref * (1.0 - rev):
                return self._sell(sym, ctx, pos, f"EXHAUSTION_{exhaustion_reversal_bps:.0f}bps")

        return []

    def _sell(self, sym, ctx, pos, reason: str):
        ms = ctx.state.market.get(sym)
        if not ms or ms.bid is None:
            return []

        limit_px = ms.bid * (1.0 - float(ctx.cfg["execution"]["exit_slippage_cap_pct"]))
        log.info(
            "ALGO1 SELL %s reason=%s bid=%.8f qty=%.8f",
            sym, reason, ms.bid, pos.qty
        )
        ctx.risk.set_cooldown(sym, int(ctx.cfg["risk"]["cooldown_s"]))
        return [OrderIntent("SELL", sym, pos.qty, limit_px)]