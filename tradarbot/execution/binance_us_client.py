from __future__ import annotations

from tradarbot.execution.binance_client import BinanceClient


class BinanceUSClient(BinanceClient):
    """Binance.US execution client.

    Reuses the Binance spot-style signing/order implementation. Configure
    execution_live.base_url to https://api.binance.us/api.
    """

    provider_name = "binance_us"
