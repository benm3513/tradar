from dataclasses import dataclass
from typing import Callable, Dict, Tuple

@dataclass
class _Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float

class CandleBuilder1s:
    """
    Builds 1-second OHLCV candles from a stream of prices.
    Emits the previous candle when the second changes.
    """
    def __init__(self, emit_fn: Callable[[str, int, _Candle], None]):
        self.emit = emit_fn
        self.cur: Dict[str, Tuple[int, _Candle]] = {}

    def on_price(self, symbol: str, ts_ms: int, price: float, qty: float = 0.0) -> None:
        bucket_ms = (ts_ms // 1000) * 1000

        if symbol not in self.cur:
            self.cur[symbol] = (bucket_ms, _Candle(price, price, price, price, qty))
            return

        cur_bucket_ms, c = self.cur[symbol]

        if bucket_ms != cur_bucket_ms:
            self.emit(symbol, cur_bucket_ms + 1000, c)
            self.cur[symbol] = (bucket_ms, _Candle(price, price, price, price, qty))
            return

        if price > c.high:
            c.high = price
        if price < c.low:
            c.low = price
        c.close = price
        c.volume += qty
