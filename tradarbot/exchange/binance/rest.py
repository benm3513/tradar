from typing import Any, Dict, Optional
import httpx

class BinanceRestClient:
    """
    One client with separate bases:
      - data_rest_base_url: for market data endpoints (reachable, e.g. Binance.US)
      - exec_rest_base_url: for trading endpoints (testnet)
      - exchange_info_url: optional full URL override
    """
    def __init__(
        self,
        data_rest_base_url: str,
        exec_rest_base_url: str,
        exchange_info_url: Optional[str] = None,
        timeout_s: float = 10.0,
    ):
        self.data_rest_base_url = data_rest_base_url.rstrip("/")
        self.exec_rest_base_url = exec_rest_base_url.rstrip("/")
        self.exchange_info_url = exchange_info_url

        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": "Tradar/0.1 (+httpx)", "Accept": "application/json"},
        )

    async def exchange_info(self) -> Dict[str, Any]:
        url = self.exchange_info_url or f"{self.data_rest_base_url}/v3/exchangeInfo"
        r = await self.client.get(url)
        r.raise_for_status()
        return r.json()

    #Market data (use data_rest_base_url) ----
    async def ticker_book(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.data_rest_base_url}/v3/ticker/bookTicker"
        r = await self.client.get(url, params={"symbol": symbol})
        r.raise_for_status()
        return r.json()

    async def ticker_price(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.data_rest_base_url}/v3/ticker/price"
        r = await self.client.get(url, params={"symbol": symbol})
        r.raise_for_status()
        return r.json()

    #Execution placeholder (later: signed endpoints on exec_rest_base_url) ----
    async def ping_exec(self) -> Dict[str, Any]:
        url = f"{self.exec_rest_base_url}/v3/ping"
        r = await self.client.get(url)
        r.raise_for_status()
        return {"ok": True}
