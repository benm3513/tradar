from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tradarbot.core.events import OrderIntent
from tradarbot.execution.binance_client import BinanceAPIError
try:
    from tradarbot.execution.alpaca_client import AlpacaAPIError
except Exception:  # keep Binance-only deployments import-safe
    AlpacaAPIError = RuntimeError
from tradarbot.execution.exchange_factory import build_exchange_client
from tradarbot.execution.fill_reconciler import FillReconciler
from tradarbot.execution.order_router import OrderRouter
from tradarbot.execution.symbol_mapper import SymbolMapper
from tradarbot.execution.slippage import validate_execution_bounds
from tradarbot.portfolio.positions import LivePositionState, PortfolioSnapshot, PositionOwner, position_notional

log = logging.getLogger("tradarbot.live_broker")


@dataclass
class LivePosition:
    qty: float = 0.0
    avg_px: float = 0.0
    entry_ts_ms: Optional[int] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    partial_exit_taken: bool = False


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
        self.cash_order_buffer = float(self.exec_cfg.get("cash_order_buffer", self.exec_cfg.get("order_notional_buffer", 0.995)) or 0.995)
        self.position_qty_buffer = float(self.exec_cfg.get("position_qty_buffer", 0.995) or 0.995)

        # Shutdown / emergency flatten behavior.
        # A close-all SELL at exactly bid can cancel if the quote moves before Alpaca
        # accepts the IOC. Use an aggressive limit below bid by default so the
        # liquidation order behaves marketable while still going through the
        # existing limit-order path.
        self.close_all_tif = str(self.exec_cfg.get("close_all_tif", "IOC") or "IOC")
        self.close_all_price_mode = str(self.exec_cfg.get("close_all_price_mode", "aggressive_limit") or "aggressive_limit").lower()
        self.close_all_slippage_pct = float(self.exec_cfg.get("close_all_slippage_pct", 0.01) or 0.01)

    def validate_startup(self) -> Dict[str, Any]:
        """Phase 5.9 live provider/account sanity check.

        Called by app/main.py only for true live startup. This intentionally
        raises on ambiguity so live mode fails closed before any strategy can
        submit an order.
        """
        profile = str((self.cfg.get("runtime", {}) or {}).get("profile", "paper") or "paper").lower()
        broker = str(self.exec_cfg.get("broker", "paper") or "paper").lower().replace("-", "_")
        mode = str(self.exec_cfg.get("mode", "paper") or "paper").lower().replace("-", "_")
        provider = str(self.provider or self.exec_cfg.get("provider", "") or "").lower()

        if profile != "live":
            return {"ok": True, "skipped": "not_live_profile"}
        if broker != "live" or mode != "live":
            raise RuntimeError(f"LIVE_PROVIDER_CHECK_FAILED reason=execution_not_live broker={broker} mode={mode}")

        required_env = [
            str(self.exec_cfg.get("api_key_env", "ALPACA_API_KEY") or "ALPACA_API_KEY"),
            str(self.exec_cfg.get("api_secret_env", "ALPACA_API_SECRET") or "ALPACA_API_SECRET"),
        ]
        missing = [name for name in required_env if not __import__("os").environ.get(name)]
        if missing:
            raise RuntimeError("LIVE_PROVIDER_CHECK_FAILED reason=missing_env vars=" + ",".join(missing))

        if bool(self.exec_cfg.get("startup_ping_required", True)):
            try:
                ping = self.client.ping() if hasattr(self.client, "ping") else {"ok": True, "provider": provider}
                log.warning("LIVE_PROVIDER_CHECK_OK provider=%s ping=%s", provider, ping)
            except Exception as exc:
                log.exception("LIVE_PROVIDER_CHECK_FAILED provider=%s", provider)
                raise RuntimeError(f"LIVE_PROVIDER_CHECK_FAILED reason=ping_failed error={exc}") from exc

        account = None
        if hasattr(self.client, "get_account"):
            try:
                account = self.client.get_account()
            except Exception as exc:
                log.exception("LIVE_ACCOUNT_VALIDATION_FAILED provider=%s", provider)
                raise RuntimeError(f"LIVE_ACCOUNT_VALIDATION_FAILED reason=get_account_failed error={exc}") from exc

        status = str((account or {}).get("status", "") or "").upper()
        required_statuses = [str(s).upper() for s in self.exec_cfg.get("account_status_required", ["ACTIVE"]) or []]
        if required_statuses and status and status not in required_statuses:
            raise RuntimeError(f"LIVE_ACCOUNT_VALIDATION_FAILED reason=bad_account_status status={status}")

        min_cash = float(self.exec_cfg.get("min_cash_usd", 0.0) or 0.0)
        cash = self._extract_account_cash(account)
        if min_cash > 0 and cash is not None and cash < min_cash:
            raise RuntimeError(f"LIVE_ACCOUNT_VALIDATION_FAILED reason=insufficient_cash cash={cash:.2f} min_cash={min_cash:.2f}")

        log.warning("LIVE_ACCOUNT_VALIDATED provider=%s status=%s cash=%s", provider, status or "unknown", cash)
        return {"ok": True, "provider": provider, "status": status, "cash": cash}

    @staticmethod
    def _extract_account_cash(account: Any) -> Optional[float]:
        if not isinstance(account, dict):
            return None
        for key in ("cash", "buying_power", "portfolio_value"):
            if account.get(key) is not None:
                try:
                    return float(account.get(key))
                except Exception:
                    pass
        balances = account.get("balances")
        if isinstance(balances, list):
            for row in balances:
                if str(row.get("asset", "")).upper() in {"USD", "USDT", "USDC"}:
                    for key in ("free", "cash", "available"):
                        if row.get(key) is not None:
                            try:
                                return float(row.get(key))
                            except Exception:
                                pass
        return None

    def positions_snapshot(self):
        return {k: v.to_dict() for k, v in self.canonical_positions().items()}

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

        routed = self._cap_routed_order_to_available(routed, ctx)
        if routed.quantity <= 0.0:
            log.warning(
                "LIVE_PRETRADE_BLOCKED symbol=%s side=%s reason=non_positive_after_balance_cap",
                routed.symbol,
                routed.side,
            )
            self.store.insert_execution_event(
                ts_ms=ts_ms,
                symbol=routed.symbol,
                event_type="pretrade_blocked",
                broker_mode=self.broker_mode,
                details={"reason": "non_positive_after_balance_cap"},
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
        except (BinanceAPIError, AlpacaAPIError) as exc:
            retry_response = self._retry_sell_with_available_from_error(routed=routed, exc=exc, ctx=ctx, ts_ms=ts_ms)
            if retry_response is not None:
                log.info(
                    "LIVE_ORDER_RETRY_RESPONSE symbol=%s client_order_id=%s response=%r",
                    routed.symbol,
                    routed.client_order_id,
                    retry_response,
                )
                state = self.reconciler.normalize_order(retry_response)
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
                return

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

    def _cap_routed_order_to_available(self, routed, ctx):
        """Final broker-side sizing guard.

        MLStrategy should already use OrderRouter for precision/cash-aware sizing,
        but the broker is the last line of defense before the exchange. This keeps
        strategy -> router -> broker quantity semantics identical.
        """
        side = str(routed.side).upper()
        if side == "BUY":
            price = float(routed.price or 0.0)
            if price <= 0.0:
                return routed
            capped_qty = self.router.clamp_buy_quantity_to_cash(
                symbol=routed.symbol,
                desired_qty=float(routed.quantity or 0.0),
                price=price,
                cash=float(self.cash or 0.0),
                fee_bps=float(self.fee_bps or 0.0),
                cash_buffer=float(self.cash_order_buffer or 1.0),
            )
            if capped_qty < float(routed.quantity or 0.0):
                log.info(
                    "LIVE_SIZE_CAPPED side=BUY symbol=%s old_qty=%.8f new_qty=%.8f cash=%.8f price=%.8f buffer=%.4f",
                    routed.symbol,
                    float(routed.quantity or 0.0),
                    float(capped_qty),
                    float(self.cash or 0.0),
                    price,
                    float(self.cash_order_buffer or 1.0),
                )
                routed.quantity = capped_qty
            return routed

        if side == "SELL":
            pos = self._local_position_for_symbol(routed.symbol)
            available_qty = float(getattr(pos, "qty", 0.0) or 0.0) if pos is not None else 0.0
            capped_qty = self.router.clamp_sell_quantity_to_position(
                symbol=routed.symbol,
                desired_qty=float(routed.quantity or 0.0),
                available_qty=available_qty,
                position_buffer=float(self.position_qty_buffer or 1.0),
            )
            if capped_qty < float(routed.quantity or 0.0):
                log.info(
                    "LIVE_SIZE_CAPPED side=SELL symbol=%s old_qty=%.8f new_qty=%.8f local_available=%.8f buffer=%.4f",
                    routed.symbol,
                    float(routed.quantity or 0.0),
                    float(capped_qty),
                    available_qty,
                    float(self.position_qty_buffer or 1.0),
                )
                routed.quantity = capped_qty
            return routed

        return routed

    def _local_position_for_symbol(self, symbol: str):
        for candidate in self._market_symbol_candidates(symbol):
            pos = self.positions.get(candidate)
            if pos is not None:
                return pos
        return self.positions.get(symbol)

    def _retry_sell_with_available_from_error(self, *, routed, exc, ctx, ts_ms: int):
        """Retry an Alpaca SELL once when the API reports a smaller available qty.

        Example error: "insufficient balance for BTC (requested: 0.007094,
        available: 0.007076265)". This protects shutdown flattening and live exits
        from tiny exchange/local-position drift.
        """
        if str(routed.side).upper() != "SELL":
            return None
        text = str(exc)
        try:
            payload = getattr(exc, "payload", None)
            if isinstance(payload, dict):
                text += " " + str(payload.get("message") or payload.get("error") or "")
        except Exception:
            pass

        import re
        match = re.search(r"available:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if not match:
            return None
        available = float(match.group(1))
        retry_qty = self.router.clamp_sell_quantity_to_position(
            symbol=routed.symbol,
            desired_qty=float(routed.quantity or 0.0),
            available_qty=available,
            position_buffer=float(self.position_qty_buffer or 1.0),
        )
        if retry_qty <= 0.0 or retry_qty >= float(routed.quantity or 0.0):
            return None

        old_qty = routed.quantity
        old_client_order_id = routed.client_order_id
        try:
            self.store.update_order_status(
                client_order_id=old_client_order_id,
                status="REJECTED",
                updated_ts_ms=ts_ms,
                exchange_order_id=None,
                error_code=str(getattr(exc, "status_code", "")) or None,
                error_message=str(exc),
                metadata={"retrying_with_available_qty": True, "available_qty": available},
            )
        except Exception:
            log.exception("LIVE_ORDER_RETRY_MARK_ORIGINAL_REJECTED_FAILED client_order_id=%s", old_client_order_id)

        routed.quantity = retry_qty
        retry_client_order_id = self.router._client_order_id(symbol=self.router.to_source_symbol(routed.symbol), side=routed.side)
        routed.client_order_id = retry_client_order_id
        log.warning(
            "LIVE_ORDER_RETRY_AVAILABLE_QTY symbol=%s side=SELL old_qty=%.8f available=%.8f retry_qty=%.8f client_order_id=%s",
            routed.symbol,
            float(old_qty or 0.0),
            available,
            retry_qty,
            retry_client_order_id,
        )
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
            strategy_name=None,
            metadata={**dict(routed.raw or {}), "retry_after_available_qty_error": True, "old_qty": old_qty, "available_qty": available},
        )
        try:
            if routed.order_type == "MARKET":
                response = self.client.place_market_order(
                    symbol=routed.symbol,
                    side=routed.side,
                    quantity=routed.quantity,
                    client_order_id=routed.client_order_id,
                )
            else:
                response = self.client.place_limit_order(
                    symbol=routed.symbol,
                    side=routed.side,
                    quantity=routed.quantity,
                    price=float(routed.price or 0.0),
                    tif=str(routed.tif or "IOC"),
                    client_order_id=routed.client_order_id,
                )
            return response
        except Exception:
            log.exception("LIVE_ORDER_RETRY_FAILED symbol=%s side=%s client_order_id=%s", routed.symbol, routed.side, routed.client_order_id)
            return None

    def close_all(self, ctx=None, reason: str = "manual") -> None:
        """Flatten all locally tracked live positions.

        This is intentionally broker-local and should be safe for shutdown:
        - iterates actual self.positions keys, usually venue symbols like ETH/USD
        - supports LivePosition dataclass objects, not dict-only positions
        - resolves market-data book from ctx when available
        - uses an aggressive SELL limit below bid/current price
        - sends through execute_intent() so routing, DB logging, fills, and sizing
          guards remain centralized
        """
        log.info("CLOSE_ALL_REQUESTED reason=%s", reason)
        log.info("CLOSE_ALL_START positions=%s", list(self.positions.keys()))

        for symbol, pos in list(self.positions.items()):
            qty = float(getattr(pos, "qty", 0.0) or 0.0)
            if qty <= 0.0:
                log.info("CLOSE_ALL_SKIP_EMPTY symbol=%s qty=%.8f", symbol, qty)
                continue

            bid, ask = self._resolve_book(ctx, symbol) if ctx is not None else (None, None)
            price = self._close_all_limit_price(symbol=symbol, pos=pos, bid=bid, ask=ask)

            if price is None or float(price) <= 0.0:
                log.warning(
                    "CLOSE_ALL_SKIPPED no_price symbol=%s bid=%s ask=%s current_price=%s avg_px=%s",
                    symbol,
                    bid,
                    ask,
                    getattr(pos, "current_price", None),
                    getattr(pos, "avg_px", None),
                )
                continue

            # Use the exact local qty here. execute_intent() will route/floor to venue
            # precision and _cap_routed_order_to_available() will apply the final
            # position buffer / availability cap before submission.
            log.info(
                "CLOSE_ALL_SUBMIT symbol=%s qty=%.8f limit_px=%.8f tif=%s bid=%s ask=%s reason=%s",
                symbol,
                qty,
                float(price),
                self.close_all_tif,
                bid,
                ask,
                reason,
            )

            intent = OrderIntent(
                side="SELL",
                symbol=symbol,
                qty=qty,
                limit_px=float(price),
                tif=self.close_all_tif,
            )

            self.execute_intent(intent, ctx)

    def _close_all_limit_price(self, *, symbol: str, pos: LivePosition, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        """Return the liquidation limit price used by close_all()."""
        reference = None
        if bid is not None and float(bid) > 0.0:
            reference = float(bid)
        elif getattr(pos, "current_price", None) is not None and float(pos.current_price) > 0.0:
            reference = float(pos.current_price)
        elif getattr(pos, "avg_px", None) is not None and float(pos.avg_px) > 0.0:
            reference = float(pos.avg_px)

        if reference is None or reference <= 0.0:
            return None

        mode = self.close_all_price_mode
        if mode in {"bid", "passive", "at_bid"}:
            raw_price = reference
        else:
            # For a SELL, lower limit price is more aggressive and more likely
            # to fill immediately as IOC. Bound slippage to avoid accidental zero.
            slip = max(0.0, min(float(self.close_all_slippage_pct or 0.0), 0.25))
            raw_price = reference * (1.0 - slip)

        try:
            venue_symbol = self.router.to_venue_symbol(symbol)
            return self.router._round_price(symbol=symbol, venue_symbol=venue_symbol, price=raw_price)
        except Exception:
            log.exception("LIVE_CLOSE_ALL_PRICE_ROUND_FAILED symbol=%s raw_price=%.8f", symbol, raw_price)
            return raw_price

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
        side = str(fill.side).upper()
        position_event = None
        realized_delta = 0.0

        if side == "BUY":
            if pos.qty <= 1e-12:
                pos.entry_ts_ms = fill.ts_ms
                pos.peak_price = None
                pos.trailing_stop_price = None
                pos.partial_exit_taken = False
                position_event = "ENTRY_FILL"
            else:
                position_event = "INCREASE_FILL"
            total_cost = pos.avg_px * pos.qty + fill.px * fill.qty
            pos.qty += fill.qty
            if pos.qty > 0:
                pos.avg_px = total_cost / pos.qty
            pos.current_price = fill.px
            self.cash -= (fill.qty * fill.px) + fill_fee
        else:
            sell_qty = min(pos.qty, fill.qty)
            gross_pnl = (fill.px - pos.avg_px) * sell_qty
            realized_delta = gross_pnl - fill_fee
            self.realized_pnl += realized_delta
            pos.realized_pnl += realized_delta
            self.cash += (sell_qty * fill.px) - fill_fee
            if realized_delta >= 0.0:
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
            pos.current_price = fill.px
            if pos.qty > 1e-12:
                pos.partial_exit_taken = True
                position_event = "PARTIAL_EXIT_FILL"
            else:
                position_event = "EXIT_FILL"
                pos = LivePosition()

        self.positions[fill.symbol] = pos
        if getattr(pos, "qty", 0.0) <= 1e-12:
            self.positions.pop(fill.symbol, None)

        try:
            self.store.insert_position_event(
                ts_ms=fill.ts_ms or self._resolve_ts_ms(ctx),
                symbol=fill.symbol,
                event_type=position_event or f"{side}_FILL",
                qty=fill.qty,
                px=fill.px,
                reason=(position_event or side).lower(),
                strategy_name=None,
                client_order_id=getattr(fill, "client_order_id", None),
                exchange_order_id=getattr(fill, "exchange_order_id", None),
                metadata={"fee": fill_fee, "realized_pnl_delta": realized_delta, "raw": getattr(fill, "raw", {})},
            )
            log.info("POSITION_EVENT %s symbol=%s qty=%.8f px=%.8f", position_event, fill.symbol, fill.qty, fill.px)
        except Exception:
            log.exception("POSITION_EVENT_PERSIST_FAILED symbol=%s", fill.symbol)

        equity = self.equity(getattr(ctx, "state", None))
        self.sync_state_positions(ctx)
        self.persist_position_snapshots(ctx)
        log.info(
            "LIVE_POSITION_UPDATED symbol=%s side=%s qty=%.8f avg_px=%.8f cash=%.8f realized_pnl=%.8f equity=%.8f",
            fill.symbol,
            fill.side,
            self.positions.get(fill.symbol, LivePosition()).qty,
            self.positions.get(fill.symbol, LivePosition()).avg_px,
            self.cash,
            self.realized_pnl,
            equity,
        )

    def canonical_positions(self, ctx=None) -> Dict[str, LivePositionState]:
        state = getattr(ctx, "state", None) if ctx is not None else None
        if state is not None:
            self.unrealized_pnl(state)
        out: Dict[str, LivePositionState] = {}
        for symbol, pos in dict(self.positions or {}).items():
            if float(getattr(pos, "qty", 0.0) or 0.0) <= 0.0:
                continue
            venue_symbol = symbol
            try:
                venue_symbol = self.router.to_venue_symbol(symbol)
            except Exception:
                pass
            out[symbol] = LivePositionState(
                symbol=symbol,
                venue_symbol=venue_symbol,
                qty=float(pos.qty or 0.0),
                avg_px=float(pos.avg_px or 0.0),
                entry_ts_ms=pos.entry_ts_ms,
                current_price=pos.current_price,
                unrealized_pnl=float(pos.unrealized_pnl or 0.0),
                realized_pnl=float(getattr(pos, "realized_pnl", 0.0) or 0.0),
                peak_price=getattr(pos, "peak_price", None),
                trailing_stop_price=getattr(pos, "trailing_stop_price", None),
                partial_exit_taken=bool(getattr(pos, "partial_exit_taken", False)),
                owner=PositionOwner(strategy_name="live_broker"),
                metadata={"broker_mode": self.broker_mode, "provider": self.provider},
            )
        return out

    def sync_state_positions(self, ctx) -> None:
        if ctx is None or not hasattr(ctx, "state"):
            return
        positions = self.canonical_positions(ctx)
        ctx.state.live_positions = positions
        for sym, pos in positions.items():
            if hasattr(ctx.state, "set_live_position"):
                ctx.state.set_live_position(sym, pos)
        for sym in list(getattr(ctx.state, "live_positions", {}).keys()):
            if sym not in positions and hasattr(ctx.state, "remove_live_position"):
                ctx.state.remove_live_position(sym)

    def persist_position_snapshots(self, ctx) -> None:
        if ctx is None or not hasattr(ctx, "store"):
            return
        ts_ms = self._resolve_ts_ms(ctx) or 0
        positions = self.canonical_positions(ctx)
        unrealized = sum(float(p.unrealized_pnl or 0.0) for p in positions.values())
        total_exposure = sum(position_notional(p) for p in positions.values())
        snapshot = PortfolioSnapshot(
            ts_ms=ts_ms, cash=float(self.cash or 0.0), equity=float(self.account_equity or 0.0),
            realized_pnl=float(self.realized_pnl or 0.0), unrealized_pnl=unrealized,
            total_exposure=total_exposure, positions=positions, broker_mode=self.broker_mode,
            metadata={"provider": self.provider},
        )
        if hasattr(ctx.state, "set_portfolio_snapshot"):
            ctx.state.set_portfolio_snapshot(snapshot)
        if hasattr(ctx.store, "insert_portfolio_snapshot"):
            ctx.store.insert_portfolio_snapshot(snapshot)
        log.info("PORTFOLIO_SNAPSHOT broker_mode=%s positions=%d equity=%.8f exposure=%.8f", self.broker_mode, len(positions), self.account_equity, total_exposure)

    def has_open_exit_order(self, symbol: str) -> bool:
        candidates = set(self._market_symbol_candidates(symbol))
        candidates.add(str(symbol))
        for order in dict(self.open_orders or {}).values():
            if str(order.get("side", "")).upper() != "SELL":
                continue
            if str(order.get("status", "")).upper() in {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "ERROR"}:
                continue
            if str(order.get("symbol")) in candidates:
                return True
        return False

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
