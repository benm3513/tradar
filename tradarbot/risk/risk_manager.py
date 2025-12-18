from typing import Any, Dict
import time

class RiskManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.cooldowns = {}  # symbol -> until_epoch_s

    def in_cooldown(self, symbol: str) -> bool:
        return time.time() < self.cooldowns.get(symbol, 0)

    def set_cooldown(self, symbol: str, seconds: int) -> None:
        self.cooldowns[symbol] = time.time() + seconds

    def check(self, intent, ctx, strat_name: str) -> Dict[str, Any]:
        ms = ctx.state.market.get(intent.symbol)
        if not ms or ms.bid is None or ms.ask is None:
            return {"approved": False, "reason": "no_market"}

        mid = (ms.bid + ms.ask) / 2.0
        spread_pct = (ms.ask - ms.bid) / max(mid, 1e-12)
        if spread_pct > float(ctx.cfg["execution"]["max_spread_pct"]):
            return {"approved": False, "reason": "spread"}

        if self.in_cooldown(intent.symbol):
            return {"approved": False, "reason": "cooldown"}

        max_pos = int(ctx.cfg["risk"]["max_positions"])
        if intent.side == "BUY" and len(ctx.broker.positions) >= max_pos:
            return {"approved": False, "reason": "max_positions"}

        return {"approved": True, "intent": intent}
