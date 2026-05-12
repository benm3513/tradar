from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from tradarbot.core.events import CandleEvent


@dataclass
class BuiltCandle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class RollingCandleBuilder:
    """Configurable OHLCV builder for live BookEvent/ticker price streams.

    It returns closed CandleEvent objects and optionally invokes emit_fn. This lets
    main.py keep the current event bus path while supporting intervals beyond 1s.
    """

    def __init__(self, interval_s: int = 1, emit_fn: Optional[Callable[[CandleEvent], None]] = None):
        self.interval_s = max(1, int(interval_s or 1))
        self.interval_ms = self.interval_s * 1000
        self.emit_fn = emit_fn
        self.cur: Dict[str, Tuple[int, BuiltCandle]] = {}

    def on_price(self, symbol: str, ts_ms: int, price: float, qty: float = 0.0) -> List[CandleEvent]:
        symbol = str(symbol)
        ts_ms = int(ts_ms)
        price = float(price)
        qty = float(qty or 0.0)
        bucket_ms = (ts_ms // self.interval_ms) * self.interval_ms
        closed: List[CandleEvent] = []

        if symbol not in self.cur:
            self.cur[symbol] = (bucket_ms, BuiltCandle(price, price, price, price, qty))
            return closed

        cur_bucket_ms, candle = self.cur[symbol]
        if bucket_ms != cur_bucket_ms:
            ev = CandleEvent(
                symbol=symbol,
                interval_s=self.interval_s,
                ts_ms=cur_bucket_ms + self.interval_ms,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            closed.append(ev)
            if self.emit_fn is not None:
                self.emit_fn(ev)
            self.cur[symbol] = (bucket_ms, BuiltCandle(price, price, price, price, qty))
            return closed

        candle.high = max(candle.high, price)
        candle.low = min(candle.low, price)
        candle.close = price
        candle.volume += qty
        return closed

    def flush(self) -> List[CandleEvent]:
        out: List[CandleEvent] = []
        for symbol, (bucket_ms, candle) in list(self.cur.items()):
            ev = CandleEvent(
                symbol=symbol,
                interval_s=self.interval_s,
                ts_ms= bucket_ms + self.interval_ms,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            out.append(ev)
            if self.emit_fn is not None:
                self.emit_fn(ev)
        self.cur.clear()
        return out
