import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from tradarbot.core.events import OrderIntent
from tradarbot.portfolio.positions import LivePositionState, PortfolioSnapshot, PositionOwner, position_notional, position_unrealized_pnl

log = logging.getLogger("tradarbot.broker")


@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0
    entry_ts_ms: Optional[int] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    partial_exit_taken: bool = False


class PaperBroker:
    def __init__(self, fee_bps: float, starting_cash: float = 10_000.0):
        self.fee_bps = float(fee_bps)
        self.cash = float(starting_cash)
        self.account_equity = float(starting_cash)
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Dict] = {}
        self.broker_mode = "paper"
        self.current_losing_streak = 0
        self.worst_losing_streak = 0
        self.realized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_hold_s = 0.0

    def positions_snapshot(self):
        return {k: self._canonical_position(k, v).to_dict() for k, v in self.positions.items() if v.qty > 0.0}

    def metrics_snapshot(self):
        avg_hold = (self.total_hold_s / self.trades) if self.trades > 0 else 0.0
        win_rate = (self.wins / self.trades) if self.trades > 0 else 0.0
        return {"realized_pnl": self.realized_pnl, "trades": self.trades, "wins": self.wins, "losses": self.losses, "win_rate": win_rate, "avg_hold_s": avg_hold, "current_losing_streak": self.current_losing_streak, "worst_losing_streak": self.worst_losing_streak, "open_orders": len(self.open_orders), "broker_mode": self.broker_mode}

    def unrealized_pnl(self, state) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            ms = state.market.get(sym)
            if not ms or ms.bid is None or ms.ask is None:
                continue
            mid = (ms.bid + ms.ask) / 2.0
            pos.current_price = mid
            pos.unrealized_pnl = (mid - pos.avg_px) * pos.qty
            total += pos.unrealized_pnl
        return total

    def equity(self, state) -> float:
        eq = self.cash + sum(pos.qty * ((state.market[sym].bid + state.market[sym].ask) / 2.0) for sym, pos in self.positions.items() if sym in state.market and state.market[sym].bid is not None and state.market[sym].ask is not None)
        self.account_equity = eq
        return eq

    def canonical_positions(self, ctx=None):
        if ctx is not None:
            self.unrealized_pnl(ctx.state)
        return {sym: self._canonical_position(sym, pos) for sym, pos in self.positions.items() if pos.qty > 0.0}

    def sync_state_positions(self, ctx):
        if ctx is None:
            return
        positions = self.canonical_positions(ctx)
        ctx.state.live_positions = positions
        for sym, pos in positions.items():
            ctx.state.set_live_position(sym, pos)
        for sym in list(getattr(ctx.state, "live_positions", {}).keys()):
            if sym not in positions:
                ctx.state.remove_live_position(sym)

    def persist_position_snapshots(self, ctx):
        if ctx is None or not hasattr(ctx, "store"):
            return
        ts_ms = self._resolve_fill_ts_ms(ctx, "") or int(time.time() * 1000)
        positions = self.canonical_positions(ctx)
        unrealized = sum(p.unrealized_pnl for p in positions.values())
        total_exposure = sum(position_notional(p) for p in positions.values())
        snapshot = PortfolioSnapshot(ts_ms=ts_ms, cash=self.cash, equity=self.account_equity, realized_pnl=self.realized_pnl, unrealized_pnl=unrealized, total_exposure=total_exposure, positions=positions, broker_mode=self.broker_mode)
        ctx.state.set_portfolio_snapshot(snapshot)
        if hasattr(ctx.store, "insert_portfolio_snapshot"):
            ctx.store.insert_portfolio_snapshot(snapshot)
        log.info("PORTFOLIO_SNAPSHOT broker_mode=%s positions=%d equity=%.8f exposure=%.8f", self.broker_mode, len(positions), self.account_equity, total_exposure)

    def has_open_exit_order(self, symbol: str) -> bool:
        for order in dict(self.open_orders or {}).values():
            if str(order.get("symbol")) == str(symbol) and str(order.get("side", "")).upper() == "SELL" and str(order.get("status", "")).upper() not in {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "ERROR"}:
                return True
        return False

    def _canonical_position(self, symbol: str, pos: Position) -> LivePositionState:
        return LivePositionState(symbol=symbol, venue_symbol=symbol, qty=float(pos.qty or 0.0), avg_px=float(pos.avg_px or 0.0), entry_ts_ms=pos.entry_ts_ms, current_price=pos.current_price, unrealized_pnl=float(pos.unrealized_pnl or 0.0), realized_pnl=float(pos.realized_pnl or 0.0), peak_price=pos.peak_price, trailing_stop_price=pos.trailing_stop_price, partial_exit_taken=bool(pos.partial_exit_taken), owner=PositionOwner(strategy_name="paper"), metadata={"broker_mode": self.broker_mode})

    def _resolve_fill_ts_ms(self, ctx, symbol: str) -> int:
        event_ts = getattr(ctx.state, "current_event_ts_ms", None)
        if event_ts is not None:
            return int(event_ts)
        ms = ctx.state.market.get(symbol) if symbol else None
        if ms and getattr(ms, "last_ts_ms", None) is not None:
            return int(ms.last_ts_ms)
        return int(time.time() * 1000)

    def execute_intent(self, intent: OrderIntent, ctx) -> None:
        ms = ctx.state.market.get(intent.symbol)
        if not ms or ms.bid is None or ms.ask is None:
            return
        fill_ts_ms = self._resolve_fill_ts_ms(ctx, intent.symbol)
        fee = self.fee_bps / 10_000.0
        client_order_id = f"paper-{intent.symbol.lower()}-{intent.side.lower()}-{fill_ts_ms}"
        self.open_orders[client_order_id] = {"client_order_id": client_order_id, "symbol": intent.symbol, "side": intent.side, "qty": intent.qty, "limit_px": intent.limit_px, "status": "SUBMITTED"}
        if hasattr(ctx.store, "insert_order"):
            ctx.store.insert_order(client_order_id=client_order_id, exchange_order_id=None, symbol=intent.symbol, side=intent.side, order_type="LIMIT", tif=intent.tif, qty=intent.qty, limit_px=intent.limit_px, status="SUBMITTED", broker_mode=self.broker_mode, strategy_name=getattr(intent, "strategy_name", None), submitted_ts_ms=fill_ts_ms, updated_ts_ms=fill_ts_ms, metadata={"paper": True})

        if intent.side == "BUY":
            fill_px = ms.ask
            if intent.limit_px < fill_px:
                return
            cost = intent.qty * fill_px * (1.0 + fee)
            if cost > self.cash:
                return
            self.cash -= cost
            pos = self.positions.get(intent.symbol, Position())
            if pos.qty <= 1e-12:
                pos.entry_ts_ms = fill_ts_ms
            new_qty = pos.qty + intent.qty
            pos.avg_px = (pos.avg_px * pos.qty + fill_px * intent.qty) / max(new_qty, 1e-12)
            pos.qty = new_qty
            pos.current_price = fill_px
            self.positions[intent.symbol] = pos
            ctx.store.insert_fill(fill_ts_ms, intent.symbol, "BUY", intent.qty, fill_px)
            self._persist_fill(ctx, client_order_id, None, intent.symbol, "BUY", intent.qty, fill_px, intent.qty * fill_px * fee, fill_ts_ms, "ENTRY_FILL" if new_qty == intent.qty else "INCREASE_FILL")
            log.info("FILLED BUY %s qty=%.8f px=%.2f fee_bps=%.1f cash=%.2f", intent.symbol, intent.qty, fill_px, self.fee_bps, self.cash)

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
            gross_pnl = (fill_px - pos.avg_px) * sell_qty
            buy_fee_est = pos.avg_px * sell_qty * fee
            sell_fee = fill_px * sell_qty * fee
            net_pnl = gross_pnl - buy_fee_est - sell_fee
            self.realized_pnl += net_pnl
            pos.realized_pnl += net_pnl
            hold_s = max(0.0, (fill_ts_ms - pos.entry_ts_ms) / 1000.0) if pos.entry_ts_ms is not None and fill_ts_ms > 0 else 0.0
            pos.qty -= sell_qty
            pos.current_price = fill_px
            # Mark partial TP only once an actual partial sell has filled.
            if pos.qty > 1e-12 and sell_qty > 0.0:
                pos.partial_exit_taken = True
            ctx.store.insert_fill(fill_ts_ms, intent.symbol, "SELL", sell_qty, fill_px)
            event_type = "PARTIAL_EXIT_FILL" if pos.qty > 1e-12 else "EXIT_FILL"
            self._persist_fill(ctx, client_order_id, None, intent.symbol, "SELL", sell_qty, fill_px, sell_fee, fill_ts_ms, event_type, pnl=net_pnl)
            if pos.qty <= 1e-12:
                del self.positions[intent.symbol]
            else:
                self.positions[intent.symbol] = pos
            self.trades += 1
            self.total_hold_s += hold_s
            if net_pnl >= 0:
                self.wins += 1
                self.current_losing_streak = 0
            else:
                self.losses += 1
                self.current_losing_streak += 1
                self.worst_losing_streak = max(self.worst_losing_streak, self.current_losing_streak)
            log.info("FILLED SELL %s qty=%.8f px=%.2f pnl=%.2f cash=%.2f", intent.symbol, sell_qty, fill_px, net_pnl, self.cash)

        self.open_orders.pop(client_order_id, None)
        self.sync_state_positions(ctx)
        self.persist_position_snapshots(ctx)

    def _persist_fill(self, ctx, client_order_id, exchange_order_id, symbol, side, qty, px, fee, ts_ms, position_event, pnl=None):
        if hasattr(ctx.store, "update_order_status"):
            ctx.store.update_order_status(client_order_id=client_order_id, status="FILLED", updated_ts_ms=ts_ms, exchange_order_id=exchange_order_id, metadata={"fill_px": px, "paper": True, "position_event": position_event})
            ctx.store.insert_order_fill(client_order_id=client_order_id, exchange_order_id=exchange_order_id, symbol=symbol, side=side, qty=qty, px=px, fee=fee, fee_asset="USD", ts_ms=ts_ms, trade_id=None, is_maker=None, metadata={"paper": True})
        if hasattr(ctx.store, "insert_position_event"):
            ctx.store.insert_position_event(ts_ms=ts_ms, symbol=symbol, event_type=position_event, qty=qty, px=px, reason=position_event.lower(), client_order_id=client_order_id, exchange_order_id=exchange_order_id, metadata={"pnl": pnl, "fee": fee})
            log.info("POSITION_EVENT %s symbol=%s qty=%.8f px=%.8f", position_event, symbol, qty, px)

    def close_all(self, ctx, reason: str = "manual") -> None:
        for symbol, pos in list(self.positions.items()):
            if pos.qty <= 0:
                continue
            ms = ctx.state.market.get(symbol)
            if not ms or ms.bid is None:
                continue
            self.execute_intent(OrderIntent(side="SELL", symbol=symbol, qty=pos.qty, limit_px=ms.bid, tif="IOC"), ctx)
