from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx


class BinanceAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class BinanceClient:
    """Thin signed REST wrapper for Binance Spot testnet/live order operations."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        binance_cfg = dict(self.cfg.get("binance", {}))
        exec_cfg = dict(self.cfg.get("execution_live", {}))

        self.mode = str(exec_cfg.get("mode", "testnet") or "testnet").lower()
        self.enabled = bool(exec_cfg.get("enabled", False))
        self.timeout_s = float(exec_cfg.get("request_timeout_s", 10.0) or 10.0)
        self.recv_window_ms = int(exec_cfg.get("recv_window_ms", 5000) or 5000)

        default_base = binance_cfg.get("exec_rest_base_url", "https://testnet.binance.vision/api")
        self.base_url = str(exec_cfg.get("base_url") or default_base).rstrip("/")
        self.exchange_info_url = str(
            exec_cfg.get("exchange_info_url")
            or binance_cfg.get("exchange_info_url")
            or f"{self.base_url}/v3/exchangeInfo"
        )

        api_key_env = str(exec_cfg.get("api_key_env", "BINANCE_API_KEY"))
        api_secret_env = str(exec_cfg.get("api_secret_env", "BINANCE_API_SECRET"))
        self.api_key = os.getenv(api_key_env, "")
        self.api_secret = os.getenv(api_secret_env, "")

        self.client = httpx.Client(
            timeout=self.timeout_s,
            headers={
                "User-Agent": "Tradar/0.1 (+httpx)",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def ping(self) -> Dict[str, Any]:
        self._request("GET", "/v3/ping", signed=False)
        return {"ok": True, "mode": self.mode, "base_url": self.base_url}

    def get_exchange_info(self) -> Dict[str, Any]:
        response = self.client.get(self.exchange_info_url)
        self._raise_for_status(response)
        return self._normalize_json(response)

    def get_account(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/account", signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/v3/openOrders", params=params, signed=True)

    def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return self._request("GET", "/v3/order", params=params, signed=True)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        tif: str = "IOC",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": str(side).upper(),
            "type": "LIMIT",
            "timeInForce": tif,
            "quantity": self._num(quantity),
            "price": self._num(price),
            "newOrderRespType": "FULL",
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self._request("POST", "/v3/order", params=params, signed=True)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": str(side).upper(),
            "type": "MARKET",
            "quantity": self._num(quantity),
            "newOrderRespType": "FULL",
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self._request("POST", "/v3/order", params=params, signed=True)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return self._request("DELETE", "/v3/order", params=params, signed=True)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        params = dict(params or {})
        headers: Dict[str, str] = {}
        if signed:
            self._ensure_credentials()
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window_ms
            query_string = urlencode(params, doseq=True)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.api_key

        response = self.client.request(method.upper(), f"{self.base_url}{path}", params=params, headers=headers)
        self._raise_for_status(response)
        return self._normalize_json(response)

    def _ensure_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise BinanceAPIError(
                "Missing Binance API credentials in environment.",
                status_code=None,
                payload={
                    "required_env": [
                        self.cfg.get("execution_live", {}).get("api_key_env", "BINANCE_API_KEY"),
                        self.cfg.get("execution_live", {}).get("api_secret_env", "BINANCE_API_SECRET"),
                    ]
                },
            )

    @staticmethod
    def _normalize_json(response: httpx.Response) -> Any:
        payload = response.json()
        if isinstance(payload, dict):
            payload.setdefault("http_status", response.status_code)
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            payload: Any
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            raise BinanceAPIError(
                f"Binance API request failed: status={response.status_code}",
                status_code=response.status_code,
                payload=payload,
            ) from exc

    @staticmethod
    def _num(value: float) -> str:
        return format(float(value), ".12f").rstrip("0").rstrip(".")
