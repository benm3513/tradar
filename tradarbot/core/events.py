from dataclasses import dataclass

@dataclass(frozen=True)
class BookEvent:
    symbol: str
    ts_ms: int
    bid: float
    ask: float

@dataclass(frozen=True)
class CandleEvent:
    symbol: str
    interval_s: int
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class ListingEvent:
    symbol: str
    ts_ms: int

@dataclass(frozen=True)
class OrderIntent:
    side: str      # "BUY" / "SELL"
    symbol: str
    qty: float
    limit_px: float
    tif: str = "IOC"
