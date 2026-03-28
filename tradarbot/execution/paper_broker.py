import logging
from dataclasses import dataclass
from typing import Dict, Optional

from tradarbot.core.events import OrderIntent

log = logging.getLogger("tradarbot.broker")


@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0
    entry_ts_ms: Optional[int] = None  # set on first entry


class PaperBroker:
    def __init__(self, fee_bps: float, starting_cash: float = 10_000.0):
        self.fee_bps = float(fee_bps)
        self.cash = float(starting_cash)
        self.positions: Dict[str, Position] = {}

        # metrics
        self.current_losing_streak: int = 0
        self.worst_losing_streak: int = 0
        self.realized_pnl: float = 0.0
        self.trades: int = 0           # number of SELL fills (round-trip count proxy)
        self.wins: int = 0
        self.losses: int = 0
        self.total_hold_s: float = 0.0

    def positions_snapshot(self):
        return {k: {"qty": v.qty, "avg_px": v.avg_px} for k, v in self.positions.items()}

    def metrics_snapshot(self):
        avg_hold = (self.total_hold_s / self.trades) if self.trades > 0 else 0.0
        win_rate = (self.wins / self.trades) if self.trades > 0 else 0.0
        return {
            "realized_pnl": self.realized_pnl,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "avg_hold_s": avg_hold,
            "current_losing_streak": self.current_losing_streak,
            "worst_losing_streak": self.worst_losing_streak,
        }
    
    def unrealized_pnl(self, state) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            ms = state.market.get(sym)
            if not ms or ms.bid is None or ms.ask is None:
                continue
            mid = (ms.bid + ms.ask) / 2.0
            total += (mid - pos.avg_px) * pos.qty
        return total
    
    def equity(self, state) -> float:
        return self.cash + sum(
            pos.qty * ((state.market[sym].bid + state.market[sym].ask) / 2.0)
            for sym, pos in self.positions.items()
            if sym in state.market and state.market[sym].bid is not None and state.market[sym].ask is not None        
        )

    def _resolve_fill_ts_ms(self, ctx, symbol: str) -> int:
        event_ts = getattr(ctx.state, "current_event_ts_ms", None)
        if event_ts is not None:
            return int(event_ts)

        ms = ctx.state.market.get(symbol)
        if ms and getattr(ms, "last_ts_ms", None) is not None:
            return int(ms.last_ts_ms)

        return 0
    
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

            fill_ts_ms = self._resolve_fill_ts_ms(ctx, intent.symbol)

            pos = self.positions.get(intent.symbol, Position())
            if pos.qty <= 1e-12:
                pos.entry_ts_ms = fill_ts_ms

            new_qty = pos.qty + intent.qty
            pos.avg_px = (pos.avg_px * pos.qty + fill_px * intent.qty) / max(new_qty, 1e-12)
            pos.qty = new_qty
            self.positions[intent.symbol] = pos

            ctx.store.insert_fill(fill_ts_ms, intent.symbol, "BUY", intent.qty, fill_px)
            
            log.info(
                "FILLED BUY %s qty=%.8f px=%.2f fee_bps=%.1f cash=%.2f",
                intent.symbol, intent.qty, fill_px, self.fee_bps, self.cash
            )

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

            # realized pnl on sold quantity (fees approximated via fill price delta; fee already applied to proceeds)
            gross_pnl = (fill_px - pos.avg_px) * sell_qty

            buy_fee_est = pos.avg_px * sell_qty * fee
            sell_fee = fill_px * sell_qty * fee
            net_pnl = gross_pnl - buy_fee_est - sell_fee

            self.realized_pnl += net_pnl

            # hold time metric (only if closing entire position)
            fill_ts_ms = self._resolve_fill_ts_ms(ctx, intent.symbol)

            if pos.entry_ts_ms is not None and fill_ts_ms > 0:
                hold_s = max(0.0, (fill_ts_ms - pos.entry_ts_ms) / 1000.0)
            else:
                hold_s = 0.0

            pos.qty -= sell_qty
            ctx.store.insert_fill(fill_ts_ms, intent.symbol, "SELL", sell_qty, fill_px)

            self.trades += 1
            if net_pnl >= 0:
                self.wins += 1
                self.current_losing_streak = 0
            else:
                self.losses += 1
                self.current_losing_streak += 1
                self.worst_losing_streak = max(self.worst_losing_streak, self.current_losing_streak)

            self.total_hold_s += hold_s

            log.info(
                "FILLED SELL %s qty=%.8f px=%.2f pnl=%.2f hold_s=%.1f fee_bps=%.1f cash=%.2f",
                intent.symbol, sell_qty, fill_px, net_pnl, hold_s, self.fee_bps, self.cash
            )

            if pos.qty <= 1e-12:
                self.positions.pop(intent.symbol, None)
            else:
                self.positions[intent.symbol] = pos

    def close_all(self, ctx, reason: str = "FLATTEN") -> None:
        """
        Flatten all positions immediately at bid (within slippage limit cap handled by caller intent limits if desired).
        This is a best-effort paper flatten for clean shutdown / replay end.
        """
        for sym, pos in list(self.positions.items()):
            ms = ctx.state.market.get(sym)
            if not ms or ms.bid is None:
                continue
            # sell full qty at current bid (paper)
            intent = OrderIntent("SELL", sym, pos.qty, ms.bid)
            log.warning("CLOSE_ALL %s reason=%s qty=%.8f bid=%.2f", sym, reason, pos.qty, ms.bid)
            self.execute_intent(intent, ctx)
