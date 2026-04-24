from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from tradarbot.core.events import OrderIntent
from tradarbot.execution.binance_client import BinanceAPIError
from tradarbot.execution.exchange_factory import build_exchange_client
from tradarbot.execution.fill_reconciler import FillReconciler
from tradarbot.execution.order_router import OrderRouter
from tradarbot.execution.symbol_mapper import SymbolMapper
from tradarbot.execution.slippage import validate_execution_bounds

log = logging.getLogger("tradarbot.live_broker")


@dataclass
class LivePosition:
    qty: float = 0.0
    avg_px: float = 0.0
    entry_ts_ms: Optional[int] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0


class LiveBroker:
    def __init__(self, cfg, store, *, starting_cash: float = 0.0):
        self.cfg = dict(cfg or {})
        self.store = store
        self.exec_cfg = dict(self.cfg.get("execution_live", {}))
        self.fee_bps = float(self.cfg.get("execution", {}).get("fee_bps", 10.0) or 10.0)
        mode = str(self.exec_cfg.get("mode", "testnet") or "testnet").lower()
        broker = str(self.exec_cfg.get("broker", "live") or "live").lower()
        provider = str(self.exec_cfg.get("provider", self.exec_cfg.get("exchange", "binance")) or "binance").lower()
        self.provider = provider
        self.dry_run = broker in {"dry_run_live", "dry-run-live", "dryrun"} or mode in {"dry_run", "dry-run", "dryrun"}
        self.broker_mode = "dry_run_live" if self.dry_run else (f"live_{provider}_{mode}" if mode != "live" else f"live_{provider}")
        self.cash = float(starting_cash)
        self.account_equity = float(starting_cash)
        self.positions: Dict[str, LivePosition] = {}
        self.open_orders: Dict[str, Dict] = {}

        self.current_losing_streak = 0
        self.worst_losing_streak = 0
        self.realized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_hold_s = 0.0

        self.client = build_exchange_client(self.cfg)
        self.provider = getattr(self.client, "provider_name", self.provider)
        self.router = OrderRouter(self.cfg)
        self.symbol_mapper = SymbolMapper(self.cfg)
        self.reconciler = FillReconciler()
        self.order_poll_s = float(self.exec_cfg.get("order_poll_s", 1.0) or 1.0)
        self.max_order_polls = int(self.exec_cfg.get("max_order_polls", 1) or 1)

    def positions_snapshot(self):
        return {
            k: {
                "qty": v.qty,
                "avg_px": v.avg_px,
                "current_price": v.current_price,
                "unrealized_pnl": v.unrealized_pnl,
            }
            for k, v in self.positions.items()
        }

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
            "open_orders": len(self.open_orders),
            "broker_mode": self.broker_mode,
        }

    def unrealized_pnl(self, state) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.qty <= 0.0:
                continue
            price = self._resolve_mark_price(state, symbol)
            if price is None:
                price = pos.current_price
            if price is None:
                continue
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.avg_px) * pos.qty
            total += pos.unrealized_pnl
        return total

    def equity(self, state) -> float:
        mtm = 0.0
        for symbol, pos in self.positions.items():
            if pos.qty <= 0.0:
                continue
            price = self._resolve_mark_price(state, symbol)
            if price is None:
                price = pos.current_price or pos.avg_px
            mtm += pos.qty * price
        self.account_equity = self.cash + mtm
        return self.account_equity

    def refresh_account(self) -> None:
        try:
            account = self.client.get_account()
        except Exception:
            log.exception("LIVE_ACCOUNT_REFRESH_FAILED")
            return
        balances = account.get("balances", []) if isinstance(account, dict) else []
        for row in balances:
            if str(row.get("asset", "")).upper() in {"USDT", "USD"}:
                self.cash = float(row.get("free", 0.0) or 0.0)
                break

    def execute_intent(self, intent: OrderIntent, ctx) -> None:
        ts_ms = self._resolve_ts_ms(ctx)
        strategy_name = getattr(intent, "strategy_name", None)
        log.info(
            "LIVE_EXECUTE_INTENT_START symbol=%s side=%s qty=%.8f limit_px=%.8f tif=%s broker_mode=%s provider=%s",
            intent.symbol,
            intent.side,
            float(intent.qty),
            float(getattr(intent, "limit_px", 0.0) or 0.0),
            getattr(intent, "tif", "IOC"),
            self.broker_mode,
            self.provider,
        )

        try:
            routed = self.router.route_intent(intent)
        except Exception:
            log.exception(
                "LIVE_ROUTE_FAILED symbol=%s side=%s qty=%.8f limit_px=%.8f",
                intent.symbol,
                float(intent.qty),
                float(getattr(intent, "limit_px", 0.0) or 0.0),
            )
            self.store.insert_execution_event(
                ts_ms=ts_ms,
                symbol=intent.symbol,
                event_type="route_failed",
                broker_mode=self.broker_mode,
                details={"intent": getattr(intent, "__dict__", repr(intent))},
            )
            return

        log.info(
            "LIVE_ROUTED_ORDER symbol=%s side=%s order_type=%s tif=%s qty=%.8f price=%s client_order_id=%s",
            routed.symbol,
            routed.side,
            routed.order_type,
            routed.tif,
            routed.quantity,
            f"{float(routed.price):.8f}" if routed.price is not None else "MARKET",
            routed.client_order_id,
        )

        if routed.quantity <= 0.0:
            log.warning(
                "LIVE_ROUTE_REJECTED symbol=%s client_order_id=%s reason=non_positive_routed_qty",
                routed.symbol,
                routed.client_order_id,
            )
            self.store.insert_execution_event(
                ts_ms=ts_ms,
                symbol=routed.symbol,
                event_type="route_rejected",
                broker_mode=self.broker_mode,
                details={"reason": "non_positive_routed_qty", "intent": intent.__dict__},
            )
            return

        bid, ask = self._resolve_book(ctx, routed.symbol)
        log.info(
            "LIVE_PRETRADE_BOOK symbol=%s bid=%s ask=%s",
            routed.symbol,
            f"{float(bid):.8f}" if bid is not None else "None",
            f"{float(ask):.8f}" if ask is not None else "None",
        )
        max_slippage_pct = self._resolve_slippage_limit(routed.side)
        if routed.price is not None:
            check = validate_execution_bounds(
                side=routed.side,
                intended_px=float(getattr(intent, "limit_px", routed.price) or routed.price),
                actual_px=routed.price,
                bid=bid,
                ask=ask,
                max_slippage_pct=max_slippage_pct,
                max_spread_pct=self.cfg.get("execution", {}).get("max_spread_pct"),
            )
            if not check.ok:
                log.warning(
                    "LIVE_PRETRADE_BLOCKED symbol=%s side=%s reason=%s slippage_pct=%s spread_pct=%s",
                    routed.symbol,
                    routed.side,
                    check.reason,
                    check.slippage_pct,
                    check.spread_pct,
                )
                self.store.insert_execution_event(
                    ts_ms=ts_ms,
                    symbol=routed.symbol,
                    event_type="pretrade_blocked",
                    broker_mode=self.broker_mode,
                    details={"reason": check.reason, "slippage_pct": check.slippage_pct, "spread_pct": check.spread_pct},
                )
                return

        self.store.insert_order(
            client_order_id=routed.client_order_id,
            exchange_order_id=None,
            symbol=routed.symbol,
            side=routed.side,
            order_type=routed.order_type,
            tif=routed.tif,
            qty=routed.quantity,
            limit_px=routed.price,
            status="SUBMITTED",
            broker_mode=self.broker_mode,
            submitted_ts_ms=ts_ms,
            updated_ts_ms=ts_ms,
            strategy_name=strategy_name,
            metadata=routed.raw,
        )
        log.info(
            "LIVE_ORDER_DB_INSERTED symbol=%s client_order_id=%s status=SUBMITTED strategy=%s",
            routed.symbol,
            routed.client_order_id,
            strategy_name,
        )

        try:
            if routed.order_type == "MARKET":
                log.info(
                    "LIVE_ORDER_SUBMIT symbol=%s side=%s type=MARKET qty=%.8f client_order_id=%s base_url=%s",
                    routed.symbol,
                    routed.side,
                    routed.quantity,
                    routed.client_order_id,
                    self.client.base_url,
                )
                response = self.client.place_market_order(
                    symbol=routed.symbol,
                    side=routed.side,
                    quantity=routed.quantity,
                    client_order_id=routed.client_order_id,
                )
            else:
                log.info(
                    "LIVE_ORDER_SUBMIT symbol=%s side=%s type=LIMIT tif=%s qty=%.8f price=%.8f client_order_id=%s base_url=%s",
                    routed.symbol,
                    routed.side,
                    routed.tif,
                    routed.quantity,
                    float(routed.price or 0.0),
                    routed.client_order_id,
                    self.client.base_url,
                )
                response = self.client.place_limit_order(
                    symbol=routed.symbol,
                    side=routed.side,
                    quantity=routed.quantity,
                    price=float(routed.price or 0.0),
                    tif=str(routed.tif or "IOC"),
                    client_order_id=routed.client_order_id,
                )
        except BinanceAPIError as exc:
            self.store.update_order_status(
                client_order_id=routed.client_order_id,
                status="REJECTED",
                updated_ts_ms=ts_ms,
                exchange_order_id=None,
                error_code=str(exc.status_code) if exc.status_code is not None else None,
                error_message=str(exc),
                metadata={"payload": exc.payload},
            )
            self.store.insert_execution_event(
                ts_ms=ts_ms,
                symbol=routed.symbol,
                event_type="order_rejected",
                broker_mode=self.broker_mode,
                client_order_id=routed.client_order_id,
                details={"error": str(exc), "payload": exc.payload},
            )
            log.warning(
                "LIVE_ORDER_REJECTED symbol=%s side=%s client_order_id=%s status_code=%s error=%s payload=%r",
                routed.symbol,
                routed.side,
                routed.client_order_id,
                exc.status_code,
                exc,
                exc.payload,
            )
            return
        except Exception:
            log.exception(
                "LIVE_ORDER_SUBMIT_FAILED symbol=%s side=%s client_order_id=%s",
                routed.symbol,
                routed.side,
                routed.client_order_id,
            )
            self.store.update_order_status(
                client_order_id=routed.client_order_id,
                status="ERROR",
                updated_ts_ms=ts_ms,
                exchange_order_id=None,
                error_code=None,
                error_message="unexpected_submit_error",
                metadata=None,
            )
            self.store.insert_execution_event(
                ts_ms=ts_ms,
                symbol=routed.symbol,
                event_type="order_submit_failed",
                broker_mode=self.broker_mode,
                client_order_id=routed.client_order_id,
                details={"error": "unexpected_submit_error"},
            )
            return

        log.info(
            "LIVE_ORDER_RESPONSE symbol=%s client_order_id=%s response=%r",
            routed.symbol,
            routed.client_order_id,
            response,
        )

        state = self.reconciler.normalize_order(response)
        log.info(
            "LIVE_ORDER_NORMALIZED symbol=%s client_order_id=%s exchange_order_id=%s status=%s executed_qty=%.8f orig_qty=%.8f avg_px=%s fills=%d final=%s",
            state.symbol,
            state.client_order_id,
            state.exchange_order_id,
            state.status,
            state.executed_qty,
            state.orig_qty,
            f"{float(state.avg_px):.8f}" if state.avg_px is not None else "None",
            len(state.fills),
            state.is_final,
        )
        self.open_orders[routed.client_order_id] = {
            "client_order_id": routed.client_order_id,
            "symbol": routed.symbol,
            "side": routed.side,
            "orig_qty": routed.quantity,
            "status": state.status,
            "exchange_order_id": state.exchange_order_id,
        }
        self.store.update_order_status(
            client_order_id=routed.client_order_id,
            status=state.status,
            updated_ts_ms=state.update_ts_ms or ts_ms,
            exchange_order_id=state.exchange_order_id,
            metadata=state.raw,
        )
        self._apply_order_state(state, ctx)

        if not state.is_final:
            self._poll_until_terminal(symbol=routed.symbol, client_order_id=routed.client_order_id, exchange_order_id=state.exchange_order_id, ctx=ctx)

    def close_all(self, ctx, reason: str = "manual") -> None:
        for symbol, pos in list(self.positions.items()):
            if pos.qty <= 0.0:
                continue
            bid, _ = self._resolve_book(ctx, symbol)
            if bid is None or bid <= 0.0:
                continue
            log.info("LIVE_CLOSE_ALL_REQUEST symbol=%s qty=%.8f reason=%s", symbol, pos.qty, reason)
            self.execute_intent(OrderIntent(side="SELL", symbol=symbol, qty=pos.qty, limit_px=bid, tif="IOC"), ctx)
            self.store.insert_execution_event(
                ts_ms=self._resolve_ts_ms(ctx),
                symbol=symbol,
                event_type="close_all_requested",
                broker_mode=self.broker_mode,
                details={"reason": reason},
            )

    def _poll_until_terminal(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str], ctx) -> None:
        for poll_idx in range(max(0, self.max_order_polls)):
            try:
                log.info(
                    "LIVE_ORDER_POLL_START symbol=%s client_order_id=%s exchange_order_id=%s poll=%d/%d",
                    symbol,
                    client_order_id,
                    exchange_order_id,
                    poll_idx + 1,
                    self.max_order_polls,
                )
                payload = self.client.get_order(symbol=symbol, order_id=exchange_order_id, client_order_id=client_order_id)
            except Exception:
                log.exception("LIVE_ORDER_POLL_FAILED symbol=%s client_order_id=%s", symbol, client_order_id)
                return
            log.info(
                "LIVE_ORDER_POLL_RESPONSE symbol=%s client_order_id=%s payload=%r",
                symbol,
                client_order_id,
                payload,
            )
            state = self.reconciler.normalize_order(payload)
            log.info(
                "LIVE_ORDER_POLL_NORMALIZED symbol=%s client_order_id=%s status=%s executed_qty=%.8f avg_px=%s fills=%d final=%s",
                state.symbol,
                state.client_order_id,
                state.status,
                state.executed_qty,
                f"{float(state.avg_px):.8f}" if state.avg_px is not None else "None",
                len(state.fills),
                state.is_final,
            )
            self.store.update_order_status(
                client_order_id=client_order_id,
                status=state.status,
                updated_ts_ms=state.update_ts_ms or self._resolve_ts_ms(ctx),
                exchange_order_id=state.exchange_order_id,
                metadata=state.raw,
            )
            self._apply_order_state(state, ctx)
            if state.is_final:
                return

    def _apply_order_state(self, state, ctx) -> None:
        if state.exchange_order_id or state.client_order_id:
            key = state.client_order_id or state.exchange_order_id
            self.open_orders[key] = {
                "client_order_id": state.client_order_id,
                "exchange_order_id": state.exchange_order_id,
                "symbol": state.symbol,
                "side": state.side,
                "status": state.status,
                "executed_qty": state.executed_qty,
                "orig_qty": state.orig_qty,
            }
            if state.is_final:
                self.open_orders.pop(key, None)

        if state.fills:
            for fill in state.fills:
                log.info(
                    "LIVE_FILL_APPLY symbol=%s side=%s qty=%.8f px=%.8f fee=%s fee_asset=%s trade_id=%s",
                    fill.symbol,
                    fill.side,
                    fill.qty,
                    fill.px,
                    fill.fee,
                    fill.fee_asset,
                    fill.trade_id,
                )
                self.store.insert_order_fill(
                    client_order_id=state.client_order_id,
                    exchange_order_id=state.exchange_order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    qty=fill.qty,
                    px=fill.px,
                    fee=fill.fee,
                    fee_asset=fill.fee_asset,
                    ts_ms=fill.ts_ms,
                    trade_id=fill.trade_id,
                    is_maker=fill.is_maker,
                    metadata=fill.raw,
                )
                self.store.insert_fill(fill.ts_ms, fill.symbol, fill.side, fill.qty, fill.px)
                self._apply_fill(fill, ctx)
        elif state.status in {"FILLED", "PARTIALLY_FILLED"} and state.executed_qty > 0.0 and state.avg_px is not None:
            synthetic_fill_ts = state.update_ts_ms or self._resolve_ts_ms(ctx)
            log.info(
                "LIVE_SYNTHETIC_FILL_APPLY symbol=%s side=%s qty=%.8f px=%.8f status=%s",
                state.symbol,
                state.side,
                state.executed_qty,
                state.avg_px,
                state.status,
            )
            self.store.insert_order_fill(
                client_order_id=state.client_order_id,
                exchange_order_id=state.exchange_order_id,
                symbol=state.symbol,
                side=state.side,
                qty=state.executed_qty,
                px=state.avg_px,
                fee=0.0,
                fee_asset=None,
                ts_ms=synthetic_fill_ts,
                trade_id=None,
                is_maker=None,
                metadata={"synthetic": True, "status": state.status},
            )
            self.store.insert_fill(synthetic_fill_ts, state.symbol, state.side, state.executed_qty, state.avg_px)
            from tradarbot.execution.fill_reconciler import NormalizedFill
            self._apply_fill(NormalizedFill(symbol=state.symbol, side=state.side, qty=state.executed_qty, px=state.avg_px, ts_ms=synthetic_fill_ts), ctx)

        self.store.insert_execution_event(
            ts_ms=state.update_ts_ms or self._resolve_ts_ms(ctx),
            symbol=state.symbol,
            event_type=f"order_{state.status.lower()}",
            broker_mode=self.broker_mode,
            client_order_id=state.client_order_id,
            exchange_order_id=state.exchange_order_id,
            details={"executed_qty": state.executed_qty, "orig_qty": state.orig_qty, "avg_px": state.avg_px},
        )
        log.info(
            "LIVE_ORDER_STATE_APPLIED symbol=%s client_order_id=%s exchange_order_id=%s status=%s executed_qty=%.8f avg_px=%s open_orders=%d",
            state.symbol,
            state.client_order_id,
            state.exchange_order_id,
            state.status,
            state.executed_qty,
            f"{float(state.avg_px):.8f}" if state.avg_px is not None else "None",
            len(self.open_orders),
        )

    def _apply_fill(self, fill, ctx) -> None:
        pos = self.positions.get(fill.symbol, LivePosition())
        fill_fee = float(fill.fee or 0.0)
        if fill.side == "BUY":
            if pos.qty <= 1e-12:
                pos.entry_ts_ms = fill.ts_ms
            total_cost = pos.avg_px * pos.qty + fill.px * fill.qty
            pos.qty += fill.qty
            if pos.qty > 0:
                pos.avg_px = total_cost / pos.qty
            self.cash -= (fill.qty * fill.px) + fill_fee
        else:
            sell_qty = min(pos.qty, fill.qty)
            gross_pnl = (fill.px - pos.avg_px) * sell_qty
            self.realized_pnl += gross_pnl - fill_fee
            self.cash += (sell_qty * fill.px) - fill_fee
            if gross_pnl - fill_fee >= 0.0:
                self.wins += 1
                self.current_losing_streak = 0
            else:
                self.losses += 1
                self.current_losing_streak += 1
                self.worst_losing_streak = max(self.worst_losing_streak, self.current_losing_streak)
            self.trades += 1
            if pos.entry_ts_ms is not None and fill.ts_ms > 0:
                self.total_hold_s += max(0.0, (fill.ts_ms - pos.entry_ts_ms) / 1000.0)
            pos.qty = max(0.0, pos.qty - sell_qty)
            if pos.qty <= 1e-12:
                pos = LivePosition()
        self.positions[fill.symbol] = pos
        equity = self.equity(getattr(ctx, "state", None))
        log.info(
            "LIVE_POSITION_UPDATED symbol=%s side=%s qty=%.8f avg_px=%.8f cash=%.8f realized_pnl=%.8f equity=%.8f",
            fill.symbol,
            fill.side,
            self.positions[fill.symbol].qty,
            self.positions[fill.symbol].avg_px,
            self.cash,
            self.realized_pnl,
            equity,
        )

    def _resolve_book(self, ctx, symbol: str):
        """Resolve bid/ask using either venue or market-data symbol forms.

        Example: Alpaca execution uses ETH/USD, while the current REST poller
        stores Binance-style market data as ETHUSDT.
        """
        market = getattr(getattr(ctx, "state", None), "market", {}) or {}
        for candidate in self._market_symbol_candidates(symbol):
            ms = market.get(candidate)
            if ms:
                return getattr(ms, "bid", None), getattr(ms, "ask", None)
        return None, None

    def _resolve_mark_price(self, state, symbol: str):
        market = getattr(state, "market", {}) or {}
        for candidate in self._market_symbol_candidates(symbol):
            ms = market.get(candidate)
            if not ms:
                continue
            bid = getattr(ms, "bid", None)
            ask = getattr(ms, "ask", None)
            if bid is None and ask is None:
                continue
            if bid is None:
                return ask
            if ask is None:
                return bid
            return (bid + ask) / 2.0
        return None

    def _market_symbol_candidates(self, symbol: str):
        return self.symbol_mapper.market_symbol_candidates(symbol)


    def _resolve_slippage_limit(self, side: str) -> Optional[float]:
        exec_cfg = dict(self.cfg.get("execution", {}))
        if str(side).upper() == "BUY":
            return exec_cfg.get("entry_slippage_cap_pct")
        return exec_cfg.get("exit_slippage_cap_pct")

    @staticmethod
    def _resolve_ts_ms(ctx) -> int:
        ts_ms = getattr(getattr(ctx, "state", None), "current_event_ts_ms", None)
        if ts_ms is None:
            return 0
        return int(ts_ms)
