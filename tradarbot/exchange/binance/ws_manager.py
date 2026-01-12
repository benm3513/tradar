import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set

import websockets

from tradarbot.core.events import BookEvent
from tradarbot.core.timeutil import now_ms

log = logging.getLogger("tradar.binance.ws")

def _chunk(items: List[str], n: int) -> List[List[str]]:
    return [items[i:i+n] for i in range(0, len(items), n)]

class _ShardConn:
    def __init__(self, url: str, on_msg: Callable[[Dict[str, Any]], None]):
        self.url = url
        self.on_msg = on_msg
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name=f"ws_shard")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        msg = json.loads(raw)
                        self.on_msg(msg)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("ws shard error url=%s", self.url)
                await asyncio.sleep(1)

class BinanceWSManager:
    """
    Sharded combined-stream WS manager for Binance Spot Testnet.
    Rebuilds all shard connections when symbol set changes.
    """
    def __init__(self, cfg: Dict[str, Any], on_book: Callable[[BookEvent], None]):
        self.cfg = cfg
        self.on_book = on_book
        self.ws_base = cfg.get("binance", {}).get("ws_base_url", "wss://testnet.binance.vision").rstrip("/")
        self.shard_size = int(cfg.get("ws", {}).get("shard_size", 60))
        self.stream_types = cfg.get("ws", {}).get("stream_types", ["bookTicker"])

        self.current_symbols: Set[str] = set()
        self._desired_symbols: Set[str] = set()
        self._update_event = asyncio.Event()
        self._shards: List[_ShardConn] = []

    async def set_symbols(self, symbols: Set[str]) -> None:
        self._desired_symbols = set(symbols)
        self._update_event.set()

    async def run_forever(self) -> None:
        while True:
            await self._update_event.wait()
            self._update_event.clear()

            if self._desired_symbols == self.current_symbols:
                continue

            await self._rebuild(self._desired_symbols)

    async def _rebuild(self, symbols: Set[str]) -> None:
        # Build streams list
        streams: List[str] = []
        for s in sorted(symbols):
            sl = s.lower()
            for st in self.stream_types:
                streams.append(f"{sl}@{st}")

        shards = _chunk(streams, self.shard_size)
        log.info("ws rebuild: symbols=%d streams=%d shards=%d", len(symbols), len(streams), len(shards))

        # Create and start new shards
        new_shards: List[_ShardConn] = []
        for sh in shards:
            url = f"{self.ws_base}/stream?streams=" + "/".join(sh)
            conn = _ShardConn(url=url, on_msg=self._on_msg)
            await conn.start()
            new_shards.append(conn)

        # Swap + stop old shards
        old = self._shards
        self._shards = new_shards
        self.current_symbols = set(symbols)

        for c in old:
            await c.stop()

    def _on_msg(self, msg: Dict[str, Any]) -> None:
        # Combined stream wrapper: {"stream": "...", "data": {...}}
        data = msg.get("data") if isinstance(msg, dict) else None
        if not isinstance(data, dict):
            return

        # bookTicker: s,b,a are strings
        if "s" in data and "b" in data and "a" in data:
            try:
                sym = data["s"]
                bid = float(data["b"])
                ask = float(data["a"])
                ts = data.get("E")
                ts_ms = int(ts) if ts is not None else now_ms()
                self.on_book(BookEvent(symbol=sym, ts_ms=ts_ms, bid=bid, ask=ask))
            except Exception:
                return
