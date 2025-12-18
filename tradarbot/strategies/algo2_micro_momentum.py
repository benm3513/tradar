from typing import Dict, List

from tradarbot.core.events import CandleEvent, OrderIntent

class Algo2MicroMomentum:
    name = "algo2_micro_momentum"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.closes: Dict[str, List[float]] = {}  # last 3 closes

    def on_listing(self, ev, ctx):
        return []

    def on_candle(self, e: CandleEvent, ctx):
        sym = e.symbol

        arr = self.closes.get(sym, [])
        arr.append(e.close)
        if len(arr) > 3:
            arr = arr[-3:]
        self.closes[sym] = arr

        # exit if position open
        pos = ctx.broker.positions.get(sym)
        if pos and pos.qty > 0:
            return self._manage_exit(sym, e, ctx, pos)

        # entry price filter
        price_cap = self.cfg.get("price_cap", None)
        if price_cap is not None and e.close >= float(price_cap):
            return []

        if len(arr) < 3:
            return []

        min_move = max(1e-12, float(self.cfg["min_move_bps"]) * e.close)
        up1 = arr[1] > arr[0] + min_move
        up2 = arr[2] > arr[1] + min_move
        if not (up1 and up2):
            return []

        ms = ctx.state.market.get(sym)
        if not ms or ms.ask is None:
            return []

        # position sizing (starter): fraction of cash
        notional_fraction = float(self.cfg.get("notional_fraction", 0.05))
        qty_usd = ctx.broker.cash * notional_fraction
        qty = qty_usd / max(ms.ask, 1e-12)

        limit_px = ms.ask * (1.0 + float(ctx.cfg["execution"]["entry_slippage_cap_pct"]))
        return [OrderIntent("BUY", sym, qty, limit_px)]

    def _manage_exit(self, sym, e, ctx, pos):
        ms = ctx.state.market.get(sym)
        if not ms or ms.bid is None:
            return []

        tp = float(self.cfg["take_profit_pct"])
        sl = float(self.cfg["stop_pct"])

        if e.close >= pos.avg_px * (1.0 + tp):
            limit_px = ms.bid * (1.0 - float(ctx.cfg["execution"]["exit_slippage_cap_pct"]))
            return [OrderIntent("SELL", sym, pos.qty, limit_px)]

        if e.close <= pos.avg_px * (1.0 - sl):
            limit_px = ms.bid * (1.0 - float(ctx.cfg["execution"]["exit_slippage_cap_pct"]))
            ctx.risk.set_cooldown(sym, int(ctx.cfg["risk"]["cooldown_s"]))
            return [OrderIntent("SELL", sym, pos.qty, limit_px)]

        return []
