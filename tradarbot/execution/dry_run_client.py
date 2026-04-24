from __future__ import annotations

import time
from typing import Any, Dict, Optional


class DryRunExchangeClient:
    """ExchangeClient implementation that never sends network orders.

    It returns exchange-shaped payloads so LiveBroker, OrderRouter,
    FillReconciler and SQLite persistence can be exercised safely.
    """

    provider_name = "dry_run"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.exec_cfg = dict(self.cfg.get("execution_live", {}) or {})
        self.base_url = str(self.exec_cfg.get("base_url", "dry-run://local"))
        self.fill_policy = str(self.exec_cfg.get("dry_run_fill_policy", "fill_immediately") or "fill_immediately").lower()
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._next_order_id = 1

    def ping(self) -> Dict[str, Any]:
        return {"ok": True, "provider": self.provider_name, "dry_run": True}

    def get_account(self) -> Dict[str, Any]:
        return {"dry_run": True, "balances": [{"asset": "USDT", "free": str(self.exec_cfg.get("starting_cash", 10000))}]}

    def get_exchange_info(self) -> Dict[str, Any]:
        return {"dry_run": True, "symbols": []}

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        orders = [o for o in self._orders.values() if o.get("status") not in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}]
        if symbol:
            orders = [o for o in orders if o.get("symbol") == symbol]
        return orders

    def get_order(self, *, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        if client_order_id and client_order_id in self._orders:
            return dict(self._orders[client_order_id])
        for order in self._orders.values():
            if order_id is not None and str(order.get("orderId")) == str(order_id):
                return dict(order)
        return self._make_order_payload(symbol=symbol, side="BUY", order_type="LIMIT", quantity=0.0, price=0.0, tif="IOC", client_order_id=client_order_id or f"dryrun-missing-{int(time.time()*1000)}", status="REJECTED", executed_qty=0.0, fills=[])

    def place_limit_order(self, *, symbol: str, side: str, quantity: float, price: float, tif: str = "IOC", client_order_id: Optional[str] = None) -> Dict[str, Any]:
        fill = self.fill_policy == "fill_immediately"
        payload = self._make_order_payload(symbol=symbol, side=side, order_type="LIMIT", quantity=quantity, price=price, tif=tif, client_order_id=client_order_id, status="FILLED" if fill else "NEW", executed_qty=quantity if fill else 0.0, fills=[self._fill(symbol, side, quantity, price)] if fill else [])
        self._orders[payload["clientOrderId"]] = payload
        return dict(payload)

    def place_market_order(self, *, symbol: str, side: str, quantity: float, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        fill = self.fill_policy == "fill_immediately"
        payload = self._make_order_payload(symbol=symbol, side=side, order_type="MARKET", quantity=quantity, price=0.0, tif=None, client_order_id=client_order_id, status="FILLED" if fill else "NEW", executed_qty=quantity if fill else 0.0, fills=[self._fill(symbol, side, quantity, 0.0)] if fill else [])
        self._orders[payload["clientOrderId"]] = payload
        return dict(payload)

    def cancel_order(self, *, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        payload = self.get_order(symbol=symbol, order_id=order_id, client_order_id=client_order_id)
        payload["status"] = "CANCELED"
        if payload.get("clientOrderId"):
            self._orders[payload["clientOrderId"]] = payload
        return payload

    def _make_order_payload(self, *, symbol: str, side: str, order_type: str, quantity: float, price: float, tif: Optional[str], client_order_id: Optional[str], status: str, executed_qty: float, fills: list) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        cid = client_order_id or f"dryrun-{symbol.lower()}-{side.lower()}-{now_ms}"
        order_id = self._next_order_id
        self._next_order_id += 1
        return {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": cid,
            "transactTime": now_ms,
            "updateTime": now_ms,
            "price": str(price or 0.0),
            "origQty": str(quantity),
            "executedQty": str(executed_qty),
            "cummulativeQuoteQty": str(float(executed_qty) * float(price or 0.0)),
            "status": status,
            "timeInForce": tif,
            "type": order_type,
            "side": side,
            "fills": fills,
            "dry_run": True,
        }

    @staticmethod
    def _fill(symbol: str, side: str, qty: float, px: float) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "price": str(px),
            "qty": str(qty),
            "commission": "0",
            "commissionAsset": "USDT",
            "tradeId": int(time.time() * 1000),
            "time": int(time.time() * 1000),
            "isMaker": False,
            "isBuyer": side.upper() == "BUY",
        }
