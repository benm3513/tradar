import logging
from typing import List

from tradarbot.core.events import CandleEvent, ListingEvent, OrderIntent

log = logging.getLogger("tradar.engine")

class StrategyEngine:
    def __init__(self, strategies, risk, broker, ctx):
        self.strategies = strategies
        self.risk = risk
        self.broker = broker
        self.ctx = ctx

    def on_listing(self, ev: ListingEvent):
        for strat in self.strategies:
            fn = getattr(strat, "on_listing", None)
            if callable(fn):
                intents = fn(ev, self.ctx) or []
                self._handle_intents(intents, strat.name)

    def on_candle(self, ev: CandleEvent):
        self.ctx.store.insert_candle(ev)
        for strat in self.strategies:
            intents = strat.on_candle(ev, self.ctx) or []
            self._handle_intents(intents, strat.name)

    def _handle_intents(self, intents: List[OrderIntent], strat_name: str) -> None:
        for intent in intents:
            decision = self.risk.check(intent, self.ctx, strat_name)
            if not decision["approved"]:
                return
            self.broker.execute_intent(decision["intent"], self.ctx)
