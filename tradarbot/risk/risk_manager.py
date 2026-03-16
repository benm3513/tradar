import time
from collections import deque
from typing import Any, Dict

from tradarbot.core.events import OrderIntent


class RiskManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.cooldowns: Dict[str, float] = {}  # symbol -> unix timestamp until tradable

        # local order limiter
        self.order_times = deque()  # timestamps (time.time()) of approved orders

    def set_cooldown(self, symbol: str, seconds: int) -> None:
        self.cooldowns[symbol] = time.time() + float(seconds)

    def _cooldown_ok(self, symbol: str) -> bool:
        until = self.cooldowns.get(symbol, 0.0)
        return time.time() >= until

    def check(self, intent: OrderIntent, ctx, strat_name: str) -> Dict[str, Any]:
        """
        Returns dict:
          { approved: bool, intent: OrderIntent, reason?: str }
        """
        r_cfg = self.cfg.get("risk", {})
        e_cfg = self.cfg.get("execution", {})

        # cooldown
        if not self._cooldown_ok(intent.symbol):
            return {"approved": False, "reason": "cooldown"}

        # max positions
        max_positions = int(r_cfg.get("max_positions", 2))
        if intent.side == "BUY":
            if len(ctx.broker.positions) >= max_positions and intent.symbol not in ctx.broker.positions:
                return {"approved": False, "reason": "max_positions"}

        # spread gate (requires market state)
        ms = ctx.state.market.get(intent.symbol)
        if ms and ms.bid is not None and ms.ask is not None:
            spread = (ms.ask - ms.bid) / max(ms.ask, 1e-12)
            max_spread = float(e_cfg.get("max_spread_pct", 0.01))
            if spread > max_spread:
                return {"approved": False, "reason": "spread_too_wide"}

        # max orders per minute
        max_opm = int(r_cfg.get("max_orders_per_minute", 60))
        now = time.time()
        while self.order_times and now - self.order_times[0] > 60.0:
            self.order_times.popleft()
        if len(self.order_times) >= max_opm:
            return {"approved": False, "reason": "order_rate_limited"}

        # notional limits
        ms = ctx.state.market.get(intent.symbol)
        px = None
        if ms and ms.bid is not None and ms.ask is not None:
            px = ms.ask if intent.side == "BUY" else ms.bid
        if px is None:
            px = intent.limit_px

        notional = float(intent.qty) * float(px)

        max_notional = r_cfg.get("max_notional_per_trade_usd", None)
        if max_notional is not None and notional > float(max_notional):
            return {"approved": False, "reason": "max_notional_per_trade"}

        max_pos_notional = r_cfg.get("max_position_notional_usd", None)
        if max_pos_notional is not None:
            existing = ctx.broker.positions.get(intent.symbol)
            existing_notional = 0.0
            if existing and ms and ms.bid is not None and ms.ask is not None:
                mid = (ms.bid + ms.ask) / 2.0
                existing_notional = existing.qty * mid
            if intent.side == "BUY" and (existing_notional + notional) > float(max_pos_notional):
                return {"approved": False, "reason": "max_position_notional"}

        # max daily loss (paper) - uses realized pnl only
        max_daily_loss = r_cfg.get("max_daily_loss_usd", None)
        if max_daily_loss is not None:
            if ctx.broker.realized_pnl <= -float(max_daily_loss):
                return {"approved": False, "reason": "daily_loss_limit"}

        # approved
        self.order_times.append(now)
        return {"approved": True, "intent": intent}
