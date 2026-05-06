from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from tradarbot.portfolio.positions import LivePositionState

log = logging.getLogger("tradarbot.portfolio.reconcile")


@dataclass
class ReconciliationResult:
    ts_ms: int
    provider: Optional[str]
    broker_mode: Optional[str]
    local_positions_count: int = 0
    exchange_positions_count: int = 0
    local_open_orders_count: int = 0
    exchange_open_orders_count: int = 0
    adopted_positions: List[str] = field(default_factory=list)
    closed_local_positions: List[str] = field(default_factory=list)
    open_orders: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status: str = "ok"
    fail_closed_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioReconciler:
    def __init__(self, cfg: Dict[str, Any], store, broker, router=None):
        self.cfg = dict(cfg or {})
        self.store = store
        self.broker = broker
        self.router = router or getattr(broker, "router", None)
        self.portfolio_cfg = dict(self.cfg.get("portfolio", {}) or {})
        self.reconcile_cfg = dict(self.portfolio_cfg.get("reconciliation", {}) or {})

    def reconcile_startup(self, ctx=None) -> ReconciliationResult:
        ts_ms = int(time.time() * 1000)
        result = ReconciliationResult(
            ts_ms=ts_ms,
            provider=getattr(self.broker, "provider", None),
            broker_mode=getattr(self.broker, "broker_mode", None),
        )
        log.info("RECONCILE_START broker_mode=%s provider=%s", result.broker_mode, result.provider)

        if not bool(self.reconcile_cfg.get("enabled", True)):
            result.status = "disabled"
            result.warnings.append("reconciliation_disabled")
            self.persist_reconciliation_result(result)
            return result

        try:
            local_positions = self.load_local_positions()
            local_orders = self.load_local_open_orders()
            exchange_positions = self.load_exchange_positions()
            exchange_orders = self.load_exchange_open_orders()
            result.local_positions_count = len(local_positions)
            result.local_open_orders_count = len(local_orders)
            result.exchange_positions_count = len(exchange_positions)
            result.exchange_open_orders_count = len(exchange_orders)
            result.open_orders = exchange_orders or local_orders

            if exchange_positions and bool(self.reconcile_cfg.get("adopt_exchange_positions", True)):
                result.adopted_positions = self.adopt_exchange_positions_into_broker(exchange_positions, ctx)
            elif local_positions:
                self._adopt_local_positions(local_positions, ctx)
                result.adopted_positions = [p.symbol for p in local_positions]

            if result.errors:
                result.status = "error"
        except Exception as exc:
            log.exception("RECONCILE_FAILED")
            result.status = "error"
            result.errors.append(str(exc))

        if result.status == "error" and bool(self.reconcile_cfg.get("fail_closed_on_error", True)):
            result.fail_closed_active = True
            if ctx is not None and hasattr(ctx, "state"):
                setattr(ctx.state, "portfolio_fail_closed", True)

        if ctx is not None and hasattr(ctx, "state") and hasattr(ctx.state, "set_reconciliation_result"):
            ctx.state.set_reconciliation_result(result)
        self.persist_reconciliation_result(result)
        log.info(
            "RECONCILE_COMPLETE status=%s local_positions=%d exchange_positions=%d local_orders=%d exchange_orders=%d fail_closed=%s warnings=%s errors=%s",
            result.status,
            result.local_positions_count,
            result.exchange_positions_count,
            result.local_open_orders_count,
            result.exchange_open_orders_count,
            result.fail_closed_active,
            result.warnings,
            result.errors,
        )
        return result

    def load_local_positions(self) -> List[LivePositionState]:
        if hasattr(self.store, "load_latest_position_snapshots"):
            return list(self.store.load_latest_position_snapshots().values())
        return []

    def load_local_open_orders(self) -> List[Dict[str, Any]]:
        if hasattr(self.store, "load_open_orders"):
            return self.store.load_open_orders()
        return []

    def load_exchange_positions(self) -> List[LivePositionState]:
        client = getattr(self.broker, "client", None)
        if client is None or getattr(self.broker, "dry_run", False):
            return []
        raw_positions = []
        try:
            if hasattr(client, "get_positions"):
                raw_positions = client.get_positions()
            elif hasattr(client, "list_positions"):
                raw_positions = client.list_positions()
            else:
                return []
        except Exception as exc:
            if self._is_live_mode():
                raise
            log.warning("RECONCILE_EXCHANGE_POSITIONS_UNAVAILABLE err=%s", exc)
            return []
        return [self._position_from_exchange(row) for row in raw_positions or [] if self._position_from_exchange(row).qty > 0.0]

    def load_exchange_open_orders(self) -> List[Dict[str, Any]]:
        client = getattr(self.broker, "client", None)
        if client is None or getattr(self.broker, "dry_run", False):
            return []
        try:
            if hasattr(client, "get_open_orders"):
                rows = client.get_open_orders()
            else:
                return []
            return [dict(r) if isinstance(r, dict) else {"raw": repr(r)} for r in rows or []]
        except Exception as exc:
            if self._is_live_mode():
                raise
            log.warning("RECONCILE_EXCHANGE_OPEN_ORDERS_UNAVAILABLE err=%s", exc)
            return []

    def adopt_exchange_positions_into_broker(self, positions: List[LivePositionState], ctx=None) -> List[str]:
        adopted = []
        for p in positions:
            self._adopt_one(p, ctx)
            adopted.append(p.symbol)
        return adopted

    def persist_reconciliation_result(self, result: ReconciliationResult) -> None:
        if hasattr(self.store, "insert_reconciliation_run"):
            self.store.insert_reconciliation_run(result)

    def _adopt_local_positions(self, positions: List[LivePositionState], ctx=None) -> None:
        for p in positions:
            self._adopt_one(p, ctx)

    def _adopt_one(self, p: LivePositionState, ctx=None) -> None:
        try:
            from tradarbot.execution.live_broker import LivePosition
        except Exception:
            LivePosition = None
        if hasattr(self.broker, "positions") and LivePosition is not None:
            self.broker.positions[p.symbol] = LivePosition(
                qty=float(p.qty or 0.0), avg_px=float(p.avg_px or 0.0), entry_ts_ms=p.entry_ts_ms,
                current_price=p.current_price, unrealized_pnl=float(p.unrealized_pnl or 0.0)
            )
        if ctx is not None and hasattr(ctx, "state") and hasattr(ctx.state, "set_live_position"):
            ctx.state.set_live_position(p.symbol, p)

    def _position_from_exchange(self, row: Any) -> LivePositionState:
        r = dict(row or {}) if isinstance(row, dict) else {}
        symbol = str(r.get("symbol") or r.get("asset") or r.get("asset_id") or "")
        qty = float(r.get("qty", r.get("quantity", r.get("free", 0.0))) or 0.0)
        avg_px = float(r.get("avg_entry_price", r.get("avg_px", r.get("entry_price", 0.0))) or 0.0)
        cur = r.get("current_price", r.get("market_value"))
        current_price = None
        if cur is not None:
            try:
                current_price = float(cur) / qty if "market_value" in r and qty > 0 else float(cur)
            except Exception:
                current_price = None
        return LivePositionState(symbol=symbol, venue_symbol=symbol, qty=qty, avg_px=avg_px, current_price=current_price, metadata={"exchange_raw": r})

    def _is_live_mode(self) -> bool:
        mode = str(getattr(self.broker, "broker_mode", "") or "").lower()
        return mode.startswith("live_") and "paper" not in mode and "dry" not in mode
