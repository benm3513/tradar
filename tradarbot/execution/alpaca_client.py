from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class AlpacaAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str, payload: Any = None):
        super().__init__(f"Alpaca API error status={status_code}: {message}")
        self.status_code = int(status_code)
        self.payload = payload


class AlpacaClient:
    """Alpaca Trading API client for Phase 5.1 LiveBroker.

    The public methods match the ExchangeClient protocol used by LiveBroker.
    For plug compatibility with the existing FillReconciler, order responses are
    returned in a Binance-like normalized shape while preserving the raw Alpaca
    response under `raw_alpaca`.
    """

    provider_name = "alpaca"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.exec_cfg = dict(self.cfg.get("execution_live", {}) or {})
        mode = str(self.exec_cfg.get("mode", "paper") or "paper").lower()
        default_base = "https://paper-api.alpaca.markets" if mode in {"paper", "dry_run", "dry-run"} else "https://api.alpaca.markets"
        self.base_url = str(self.exec_cfg.get("base_url", default_base) or default_base).rstrip("/")
        self.api_key_env = str(self.exec_cfg.get("api_key_env", "ALPACA_API_KEY") or "ALPACA_API_KEY")
        self.api_secret_env = str(self.exec_cfg.get("api_secret_env", "ALPACA_API_SECRET") or "ALPACA_API_SECRET")
        self.api_key = os.environ.get(self.api_key_env, "")
        self.api_secret = os.environ.get(self.api_secret_env, "")
        self.timeout_s = float(self.exec_cfg.get("request_timeout_s", 10.0) or 10.0)

    def ping(self) -> Dict[str, Any]:
        account = self.get_account()
        return {"ok": True, "provider": self.provider_name, "account_status": account.get("status")}

    def get_account(self) -> Dict[str, Any]:
        return self._request("GET", "/v2/account")

    def get_exchange_info(self) -> Dict[str, Any]:
        # Alpaca has no Binance-style exchangeInfo. /v2/assets is the closest
        # master list and works for equities and crypto.
        assets = self._request("GET", "/v2/assets", params={"status": "active"})
        return {"provider": self.provider_name, "symbols": assets, "raw_assets": assets}

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"status": "open", "limit": 500}
        if symbol:
            params["symbols"] = symbol
        rows = self._request("GET", "/v2/orders", params=params)
        return [self._to_exchange_order(row) for row in rows]

    def get_order(
        self,
        *,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if client_order_id:
            row = self._request(
                "GET",
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
            )
        elif order_id:
            row = self._request("GET", f"/v2/orders/{urllib.parse.quote(str(order_id), safe='')}")
        else:
            raise ValueError("get_order requires order_id or client_order_id")
        return self._to_exchange_order(row)

    def place_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        tif: str = "IOC",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = {
            "symbol": symbol,
            "qty": self._fmt(quantity),
            "side": str(side).lower(),
            "type": "limit",
            "time_in_force": str(tif or "ioc").lower(),
            "limit_price": self._fmt(price),
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        row = self._request("POST", "/v2/orders", body=body)
        return self._to_exchange_order(row, fallback_price=price, fallback_qty=quantity)

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = {
            "symbol": symbol,
            "qty": self._fmt(quantity),
            "side": str(side).lower(),
            "type": "market",
            # Alpaca crypto supports gtc/ioc; market with ioc is appropriate for immediate execution tests.
            "time_in_force": str(self.exec_cfg.get("default_tif", "ioc") or "ioc").lower(),
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        row = self._request("POST", "/v2/orders", body=body)
        return self._to_exchange_order(row, fallback_qty=quantity)

    def cancel_order(
        self,
        *,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not order_id and client_order_id:
            found = self.get_order(symbol=symbol, client_order_id=client_order_id)
            order_id = str(found.get("orderId") or found.get("order_id") or "")
        if not order_id:
            raise ValueError("cancel_order requires order_id or resolvable client_order_id")
        path = f"/v2/orders/{urllib.parse.quote(str(order_id), safe='')}"
        try:
            self._request("DELETE", path)
            return {
                "symbol": symbol,
                "orderId": order_id,
                "clientOrderId": client_order_id,
                "status": "CANCELED",
                "executedQty": "0",
                "origQty": "0",
                "fills": [],
                "raw_alpaca": {"deleted": True},
            }
        except AlpacaAPIError as exc:
            # If DELETE returns a body, preserve it for logs/reconciliation.
            raise exc

    def _headers(self) -> Dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise AlpacaAPIError(
                0,
                f"Missing Alpaca credentials. Set {self.api_key_env} and {self.api_secret_env} in your shell.",
            )
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}{path}{query}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            payload: Any = raw
            message = raw
            try:
                payload = json.loads(raw) if raw else {}
                message = payload.get("message") or payload.get("error") or raw
            except Exception:
                pass
            raise AlpacaAPIError(exc.code, message, payload=payload) from exc
        except urllib.error.URLError as exc:
            raise AlpacaAPIError(0, str(exc), payload=None) from exc

    def _to_exchange_order(
        self,
        row: Dict[str, Any],
        *,
        fallback_price: Optional[float] = None,
        fallback_qty: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not isinstance(row, dict):
            raise AlpacaAPIError(0, f"Unexpected Alpaca order payload: {row!r}", payload=row)

        symbol = str(row.get("symbol") or "")
        side = str(row.get("side") or "").upper()
        status = self._map_status(str(row.get("status") or ""))
        order_type = str(row.get("type") or "limit").upper()
        tif = str(row.get("time_in_force") or "").upper() or None
        order_id = row.get("id") or row.get("order_id")
        client_order_id = row.get("client_order_id")

        qty = self._float(row.get("qty"), fallback_qty or 0.0)
        filled_qty = self._float(row.get("filled_qty"), 0.0)
        limit_price = self._float(row.get("limit_price"), fallback_price or 0.0)
        filled_avg_price = self._float(row.get("filled_avg_price"), 0.0)
        avg_px = filled_avg_price or limit_price or self._float(row.get("price"), 0.0)

        submitted_ms = self._ts_to_ms(row.get("submitted_at"))
        updated_ms = self._ts_to_ms(row.get("updated_at") or row.get("filled_at") or row.get("canceled_at"))
        now_ms = int(time.time() * 1000)
        ts_ms = updated_ms or submitted_ms or now_ms

        fills = []
        if filled_qty > 0.0:
            fills.append(
                {
                    "symbol": symbol,
                    "price": str(avg_px),
                    "qty": str(filled_qty),
                    "commission": "0",
                    "commissionAsset": "USD",
                    "tradeId": str(order_id or client_order_id or ts_ms),
                    "time": ts_ms,
                    "isMaker": False,
                    "isBuyer": side == "BUY",
                    "raw_alpaca_fill_synthetic": True,
                }
            )

        return {
            "symbol": symbol,
            "orderId": str(order_id) if order_id is not None else None,
            "clientOrderId": client_order_id,
            "transactTime": submitted_ms or ts_ms,
            "updateTime": ts_ms,
            "price": str(limit_price or avg_px or 0.0),
            "origQty": str(qty),
            "executedQty": str(filled_qty),
            "cummulativeQuoteQty": str(filled_qty * (avg_px or limit_price or 0.0)),
            "status": status,
            "timeInForce": tif,
            "type": order_type,
            "side": side,
            "fills": fills,
            "provider": self.provider_name,
            "raw_alpaca": row,
        }

    @staticmethod
    def _map_status(status: str) -> str:
        s = status.lower()
        if s in {"accepted", "pending_new", "accepted_for_bidding", "new"}:
            return "NEW"
        if s in {"partially_filled"}:
            return "PARTIALLY_FILLED"
        if s in {"filled"}:
            return "FILLED"
        if s in {"canceled", "pending_cancel", "stopped"}:
            return "CANCELED"
        if s in {"expired"}:
            return "EXPIRED"
        if s in {"rejected", "suspended", "calculated"}:
            return "REJECTED"
        return status.upper() if status else "UNKNOWN"

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _fmt(value: float) -> str:
        text = f"{float(value):.12f}".rstrip("0").rstrip(".")
        return text if text else "0"

    @staticmethod
    def _ts_to_ms(value: Any) -> Optional[int]:
        if not value:
            return None
        if isinstance(value, (int, float)):
            # Alpaca timestamps are normally ISO strings, but support numeric just in case.
            return int(value if value > 10_000_000_000 else value * 1000)
        try:
            from datetime import datetime

            text = str(value).replace("Z", "+00:00")
            return int(datetime.fromisoformat(text).timestamp() * 1000)
        except Exception:
            return None
