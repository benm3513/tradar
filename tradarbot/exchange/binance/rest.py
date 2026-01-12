from typing import Any, Dict, List, Optional
import httpx

class BinanceRestClient:
    def __init__(
        self,
        rest_base_url: str = "https://testnet.binance.vision/api",
        exchange_info_url: Optional[str] = None,
        timeout_s: float = 10.0,
    ):
        self.rest_base_url = rest_base_url.rstrip("/")
        self.exchange_info_url = exchange_info_url
        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": "Tradar/0.1 (+httpx)", "Accept": "application/json"},
        )

    async def exchange_info(self) -> Dict[str, Any]:
        url = self.exchange_info_url or f"{self.rest_base_url}/v3/exchangeInfo"
        r = await self.client.get(url)
        r.raise_for_status()
        return r.json()

    async def ticker_book(self, symbol: str) -> Dict[str, Any]:
        # /api/v3/ticker/bookTicker?symbol=BTCUSDT
        url = f"{self.rest_base_url}/v3/ticker/bookTicker"
        r = await self.client.get(url, params={"symbol": symbol})
        r.raise_for_status()
        return r.json()

    async def ticker_price(self, symbol: str) -> Dict[str, Any]:
        # /api/v3/ticker/price?symbol=BTCUSDT
        url = f"{self.rest_base_url}/v3/ticker/price"
        r = await self.client.get(url, params={"symbol": symbol})
        r.raise_for_status()
        return r.json()
