from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MarketState:
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    last_ts_ms: Optional[int] = None

class State:
    def __init__(self):
        self.market: Dict[str, MarketState] = {}

    @staticmethod
    def market_state_factory() -> MarketState:
        return MarketState()
