import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Set

from tradarbot.core.events import ListingEvent
from tradarbot.core.timeutil import now_ms

log = logging.getLogger("tradar.symbol_registry")

class SymbolRegistry:
    """
    Polls Binance exchangeInfo, filters symbols (e.g., TRADING USDT),
    yields symbol set when changed, emits ListingEvent for newly added symbols.
    """
    def __init__(self, rest, cfg: Dict[str, Any], bus):
        self.rest = rest
        self.cfg = cfg
        self.bus = bus
        self.known: Set[str] = set()

    async def run(self) -> AsyncGenerator[Set[str], None]:
        poll_s = int(self.cfg.get("poll_interval_s", 15))
        emit_listing = bool(self.cfg.get("emit_listing_events", True))

        while True:
            try:
                symbols = await self.fetch_symbols()
                added = symbols - self.known
                removed = self.known - symbols

                if added:
                    log.info("new symbols=%d (sample=%s)", len(added), sorted(list(added))[:10])
                    if emit_listing:
                        ts = now_ms()
                        for s in added:
                            self.bus.publish(ListingEvent(symbol=s, ts_ms=ts))

                if removed:
                    log.info("removed symbols=%d (sample=%s)", len(removed), sorted(list(removed))[:10])

                if symbols != self.known:
                    self.known = symbols
                    yield symbols

            except Exception:
                log.exception("symbol registry poll failed")

            await asyncio.sleep(poll_s)

    async def fetch_symbols(self) -> Set[str]:
        info = await self.rest.exchange_info()

        quote = self.cfg.get("quote_asset", "USDT")
        status = self.cfg.get("status", "TRADING")
        deny = set(self.cfg.get("denylist", []) or [])
        allowlist = self.cfg.get("allowlist", None)
        allow = set(allowlist) if allowlist else None

        out: Set[str] = set()
        for x in info.get("symbols", []):
            if x.get("status") != status:
                continue
            if x.get("quoteAsset") != quote:
                continue
            sym = x.get("symbol")
            if not sym:
                continue
            if sym in deny:
                continue
            if allow is not None and sym not in allow:
                continue
            out.add(sym)
        return out
