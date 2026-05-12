from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Set

from tradarbot.core.events import BookEvent
from tradarbot.exchange.binance.ws_manager import BinanceWSManager


class LiveMarketDataClient:
    """Thin lifecycle/health wrapper around BinanceWSManager.

    REST polling can remain active alongside this client; this class only owns
    WS subscriptions and reports health into app state.
    """

    def __init__(self, cfg: Dict[str, Any], bus: Any, on_book: Callable[[BookEvent], None]):
        self.cfg = cfg
        self.bus = bus
        self.on_book = on_book
        self._last_msg_ts_ms: Optional[int] = None
        self._message_count = 0
        self._reconnect_count = 0
        self._connected = False
        self._symbols: Set[str] = set()
        self._manager = BinanceWSManager(cfg=cfg, on_book=self._handle_book)

    async def set_symbols(self, symbols: Set[str]) -> None:
        self._symbols = set(symbols or set())
        await self._manager.set_symbols(self._symbols)

    async def run_forever(self) -> None:
        self._connected = True
        try:
            await self._manager.run_forever()
        finally:
            self._connected = False

    def _handle_book(self, ev: BookEvent) -> None:
        self._last_msg_ts_ms = int(ev.ts_ms)
        self._message_count += 1
        self._connected = True
        self.on_book(ev)

    def mark_reconnect(self) -> None:
        self._reconnect_count += 1

    def health_snapshot(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        age_s = None if self._last_msg_ts_ms is None else max(0.0, (now_ms - self._last_msg_ts_ms) / 1000.0)
        shards = getattr(self._manager, "_shards", []) or []
        return {
            "mode": "ws",
            "connected": bool(self._connected),
            "symbols": len(self._symbols),
            "last_message_ts_ms": self._last_msg_ts_ms,
            "last_message_age_s": age_s,
            "message_count": self._message_count,
            "reconnect_count": self._reconnect_count,
            "shard_count": len(shards),
        }
