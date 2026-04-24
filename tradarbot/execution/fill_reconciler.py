from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


FINAL_ORDER_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}


@dataclass
class NormalizedFill:
    symbol: str
    side: str
    qty: float
    px: float
    fee: float = 0.0
    fee_asset: Optional[str] = None
    ts_ms: int = 0
    trade_id: Optional[str] = None
    is_maker: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedOrderState:
    symbol: str
    client_order_id: Optional[str]
    exchange_order_id: Optional[str]
    side: str
    status: str
    executed_qty: float
    orig_qty: float
    avg_px: Optional[float]
    update_ts_ms: int
    fills: List[NormalizedFill] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return self.status in FINAL_ORDER_STATUSES


class FillReconciler:
    def normalize_order(self, payload: Dict[str, Any]) -> NormalizedOrderState:
        payload = dict(payload or {})
        symbol = str(payload.get("symbol", ""))
        side = str(payload.get("side", "")).upper()
        status = str(payload.get("status", "NEW")).upper()
        executed_qty = float(payload.get("executedQty", payload.get("executed_qty", 0.0)) or 0.0)
        orig_qty = float(payload.get("origQty", payload.get("orig_qty", 0.0)) or 0.0)
        avg_px = self._derive_avg_px(payload)
        update_ts_ms = int(payload.get("updateTime", payload.get("transactTime", payload.get("ts_ms", 0))) or 0)

        fills = [self.normalize_fill(symbol=symbol, side=side, payload=fill) for fill in payload.get("fills", []) or []]

        return NormalizedOrderState(
            symbol=symbol,
            client_order_id=payload.get("clientOrderId") or payload.get("client_order_id"),
            exchange_order_id=str(payload.get("orderId")) if payload.get("orderId") is not None else payload.get("exchange_order_id"),
            side=side,
            status=status,
            executed_qty=executed_qty,
            orig_qty=orig_qty,
            avg_px=avg_px,
            update_ts_ms=update_ts_ms,
            fills=fills,
            raw=payload,
        )

    def normalize_fill(self, *, symbol: str, side: str, payload: Dict[str, Any]) -> NormalizedFill:
        payload = dict(payload or {})
        qty = float(payload.get("qty", payload.get("executedQty", 0.0)) or 0.0)
        px = float(payload.get("price", payload.get("px", 0.0)) or 0.0)
        fee = float(payload.get("commission", payload.get("fee", 0.0)) or 0.0)
        ts_ms = int(payload.get("time", payload.get("ts_ms", payload.get("transactTime", 0))) or 0)
        return NormalizedFill(
            symbol=symbol,
            side=side,
            qty=qty,
            px=px,
            fee=fee,
            fee_asset=payload.get("commissionAsset") or payload.get("fee_asset"),
            ts_ms=ts_ms,
            trade_id=str(payload.get("tradeId")) if payload.get("tradeId") is not None else None,
            is_maker=payload.get("isMaker"),
            raw=payload,
        )

    @staticmethod
    def _derive_avg_px(payload: Dict[str, Any]) -> Optional[float]:
        if payload.get("avgPrice") not in (None, ""):
            return float(payload["avgPrice"])

        executed_qty = float(payload.get("executedQty", 0.0) or 0.0)
        cummulative_quote_qty = float(payload.get("cummulativeQuoteQty", 0.0) or 0.0)
        if executed_qty > 0.0 and cummulative_quote_qty > 0.0:
            return cummulative_quote_qty / executed_qty

        fills = payload.get("fills", []) or []
        if fills:
            total_qty = 0.0
            total_quote = 0.0
            for fill in fills:
                qty = float(fill.get("qty", 0.0) or 0.0)
                px = float(fill.get("price", 0.0) or 0.0)
                total_qty += qty
                total_quote += qty * px
            if total_qty > 0.0:
                return total_quote / total_qty
        return None
