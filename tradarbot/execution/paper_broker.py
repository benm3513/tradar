from dataclasses import dataclass
from typing import Dict

from tradarbot.core.events import OrderIntent

@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0

class PaperBroker:
    def __init__(self, fee_bps: float, starting_cash: float = 10_000.0):
        self.fee_bps = fee_bps
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}

    def positions_snapshot(self):
        return {k: {"qty": v.qty, "avg_px": v.avg_px} for k, v in self.positions.items()}

    def execute_intent(self, intent: OrderIntent, ctx) -> None:
        ms = ctx.state.market.get(intent.symbol)
        if not ms or ms.bid is None or ms.ask is None:
            return

        fee = self.fee_bps / 10_000.0

        if intent.side == "BUY":
            fill_px = ms.ask
            if intent.limit_px < fill_px:
                return
            cost = intent.qty * fill_px * (1.0 + fee)
            if cost > self.cash:
                return
            self.cash -= cost

            pos = self.positions.get(intent.symbol, Position())
            new_qty = pos.qty + intent.qty
            pos.avg_px = (pos.avg_px * pos.qty + fill_px * intent.qty) / max(new_qty, 1e-12)
            pos.qty = new_qty
            self.positions[intent.symbol] = pos
            ctx.store.insert_fill(intent.symbol, "BUY", intent.qty, fill_px)

        elif intent.side == "SELL":
            pos = self.positions.get(intent.symbol)
            if not pos or pos.qty <= 0:
                return
            fill_px = ms.bid
            if intent.limit_px > fill_px:
                return
            sell_qty = min(pos.qty, intent.qty)

            proceeds = sell_qty * fill_px * (1.0 - fee)
            self.cash += proceeds
            pos.qty -= sell_qty
            ctx.store.insert_fill(intent.symbol, "SELL", sell_qty, fill_px)

            if pos.qty <= 1e-12:
                self.positions.pop(intent.symbol, None)
            else:
                self.positions[intent.symbol] = pos
