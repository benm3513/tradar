from typing import Any, Dict

import httpx

class BinanceRestClient:
    """
    Binance Spot Testnet REST base:
    https://testnet.binance.vision/api
    """
    def __init__(self, base_url: str = "https://testnet.binance.vision/api"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def exchange_info(self) -> Dict[str, Any]:
        r = await self.client.get(f"{self.base_url}/v3/exchangeInfo")
        r.raise_for_status()
        return r.json()
