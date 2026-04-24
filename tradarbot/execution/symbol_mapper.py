from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_QUOTE_MAP: Dict[str, str] = {
    "USDT": "USD",
    "USDC": "USD",
    "USD": "USD",
}


@dataclass(frozen=True)
class SymbolPair:
    source_symbol: str
    venue_symbol: str


class SymbolMapper:
    """Provider-aware, reversible symbol mapping for execution.

    Tradar historically uses compact Binance-style symbols for market data,
    e.g. ETHUSDT. Some venues use different execution symbols, e.g. Alpaca
    uses ETH/USD. This mapper provides:

    1. explicit overrides from execution_live.symbol_map
    2. automatic provider defaults, currently ETHUSDT -> ETH/USD for Alpaca
    3. reversible candidate generation for market lookups and position lookups

    The explicit map remains useful for edge cases, but common crypto USD
    pairs no longer need to be listed manually.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.exec_cfg = dict(self.cfg.get("execution_live", {}) or {})
        self.provider = self.normalize_provider(
            self.exec_cfg.get("provider", self.exec_cfg.get("exchange", "generic"))
        )
        raw_map = dict(self.exec_cfg.get("symbol_map", {}) or {})
        self.symbol_map: Dict[str, str] = {
            self.normalize_symbol(k): self.normalize_symbol(v) for k, v in raw_map.items()
        }
        self.reverse_symbol_map: Dict[str, str] = {
            self.normalize_symbol(v): self.normalize_symbol(k) for k, v in raw_map.items()
        }
        raw_quote_map = dict(self.exec_cfg.get("quote_map", {}) or {})
        self.quote_map: Dict[str, str] = dict(DEFAULT_QUOTE_MAP)
        self.quote_map.update({str(k).upper(): str(v).upper() for k, v in raw_quote_map.items()})

    @staticmethod
    def normalize_provider(value: Any) -> str:
        provider = str(value or "generic").lower().replace("-", "_")
        if provider in {"alpaca_paper", "alpaca_live"}:
            return "alpaca"
        if provider in {"binanceus", "binance_us"}:
            return "binance_us"
        return provider

    @staticmethod
    def normalize_symbol(symbol: Any) -> str:
        return str(symbol or "").strip().upper().replace("-", "/")

    def to_venue_symbol(self, symbol: Any) -> str:
        sym = self.normalize_symbol(symbol)
        if not sym:
            return sym

        # Explicit override always wins.
        if sym in self.symbol_map:
            return self.symbol_map[sym]

        if self.provider == "alpaca":
            return self._to_alpaca_symbol(sym)

        return sym

    def to_source_symbol(self, symbol: Any) -> str:
        """Map a venue symbol back to Tradar's compact market-data style.

        For Alpaca, ETH/USD -> ETHUSDT by default because the current market
        data feed uses Binance-style USDT pairs. Explicit reverse overrides win.
        """
        sym = self.normalize_symbol(symbol)
        if not sym:
            return sym

        if sym in self.reverse_symbol_map:
            return self.reverse_symbol_map[sym]

        if self.provider == "alpaca":
            return self._from_alpaca_symbol(sym)

        return sym

    def market_symbol_candidates(self, symbol: Any) -> List[str]:
        """Return candidate symbols to try against ctx.state.market.

        Includes raw, explicit reverse/forward overrides, automatic source form,
        automatic venue form, and compact/slash variants. Ordered with the most
        likely candidates first and de-duplicated.
        """
        raw = self.normalize_symbol(symbol)
        candidates: List[str] = []

        def add(value: Any) -> None:
            value = self.normalize_symbol(value)
            if value and value not in candidates:
                candidates.append(value)

        add(raw)
        add(self.to_source_symbol(raw))
        add(self.to_venue_symbol(raw))

        if raw in self.symbol_map:
            add(self.symbol_map[raw])
        if raw in self.reverse_symbol_map:
            add(self.reverse_symbol_map[raw])

        # Include all explicit aliases touching this symbol.
        for source, venue in self.symbol_map.items():
            if raw == source or raw == venue:
                add(source)
                add(venue)

        # Generate slash/compact variants.
        for base, quote in self._parse_symbol(raw):
            add(f"{base}{quote}")
            add(f"{base}/{quote}")
            if quote == "USD":
                add(f"{base}USDT")
                add(f"{base}USDC")
            elif quote in {"USDT", "USDC"}:
                add(f"{base}/USD")
                add(f"{base}USD")

        return candidates

    def _to_alpaca_symbol(self, sym: str) -> str:
        # Alpaca crypto execution symbols use slash notation, e.g. ETH/USD.
        if "/" in sym:
            base, quote = sym.split("/", 1)
            mapped_quote = self.quote_map.get(quote, quote)
            return f"{base}/{mapped_quote}"

        for quote in sorted(self.quote_map.keys(), key=len, reverse=True):
            if sym.endswith(quote) and len(sym) > len(quote):
                base = sym[: -len(quote)]
                return f"{base}/{self.quote_map.get(quote, quote)}"
        return sym

    def _from_alpaca_symbol(self, sym: str) -> str:
        # Current Tradar market-data symbols are compact USDT pairs.
        if "/" in sym:
            base, quote = sym.split("/", 1)
            if quote == "USD":
                return f"{base}USDT"
            return f"{base}{quote}"
        return sym

    def _parse_symbol(self, sym: str) -> Iterable[Tuple[str, str]]:
        if not sym:
            return []
        if "/" in sym:
            base, quote = sym.split("/", 1)
            return [(base, quote)]
        pairs = []
        for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
            if sym.endswith(quote) and len(sym) > len(quote):
                pairs.append((sym[: -len(quote)], quote))
        return pairs
