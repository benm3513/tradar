from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from tradarbot.core.events import OrderIntent
from tradarbot.portfolio.positions import LivePositionState, update_peak_and_trailing


@dataclass
class ExitConfig:
    enabled: bool = True
    take_profit_pct: Optional[float] = 0.20
    stop_loss_pct: Optional[float] = 0.06
    max_hold_hours: Optional[float] = 24.0
    trailing_stop_pct: Optional[float] = 0.05
    trailing_stop_activation_pct: Optional[float] = 0.08
    partial_take_profit_pct: Optional[float] = 0.10
    partial_take_profit_fraction: float = 0.50
    time_stop_hours: Optional[float] = None
    time_stop_min_return_pct: float = 0.01
    default_tif: str = "IOC"
    exit_slippage_pct: float = 0.01


@dataclass
class ExitDecision:
    should_exit: bool
    symbol: str
    side: str = "SELL"
    qty: float = 0.0
    limit_px: float = 0.0
    reason: str = ""
    partial: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExitManager:
    def __init__(self, config: ExitConfig):
        self.config = config

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ExitManager":
        p = dict((cfg or {}).get("portfolio", {}) or {})
        e = dict(p.get("exits", {}) or {})
        return cls(ExitConfig(
            enabled=bool(e.get("enabled", p.get("enabled", True))),
            take_profit_pct=e.get("take_profit_pct", 0.20),
            stop_loss_pct=e.get("stop_loss_pct", 0.06),
            max_hold_hours=e.get("max_hold_hours", 24),
            trailing_stop_pct=e.get("trailing_stop_pct", 0.05),
            trailing_stop_activation_pct=e.get("trailing_stop_activation_pct", 0.08),
            partial_take_profit_pct=e.get("partial_take_profit_pct", 0.10),
            partial_take_profit_fraction=float(e.get("partial_take_profit_fraction", 0.50) or 0.50),
            time_stop_hours=e.get("time_stop_hours"),
            time_stop_min_return_pct=float(e.get("time_stop_min_return_pct", 0.01) or 0.0),
            default_tif=str(e.get("exit_tif", e.get("default_tif", "IOC")) or "IOC"),
            exit_slippage_pct=float(e.get("exit_slippage_pct", 0.01) or 0.0),
        ))

    def evaluate_all(self, positions: Dict[str, LivePositionState], state, now_ts_ms: int) -> List[ExitDecision]:
        if not self.config.enabled:
            return []
        decisions: List[ExitDecision] = []
        for symbol, pos in dict(positions or {}).items():
            market_state = None
            for candidate in self._symbol_candidates(pos, symbol):
                market_state = getattr(state, "market", {}).get(candidate)
                if market_state is not None:
                    break
            decisions.extend(self.evaluate_position(pos, market_state, now_ts_ms))
        return decisions

    def evaluate_position(self, position: LivePositionState, market_state, now_ts_ms: int) -> List[ExitDecision]:
        if not self.config.enabled or position is None or float(position.qty or 0.0) <= 0.0:
            return []
        bid = getattr(market_state, "bid", None) if market_state is not None else None
        ask = getattr(market_state, "ask", None) if market_state is not None else None
        mark = self._mark_price(bid, ask, getattr(position, "current_price", None))
        if bid is None or float(bid or 0.0) <= 0.0 or mark is None or float(mark) <= 0.0:
            return []

        update_peak_and_trailing(
            position,
            float(mark),
            self.config.trailing_stop_pct,
            self.config.trailing_stop_activation_pct,
        )

        avg_px = float(position.avg_px or 0.0)
        if avg_px <= 0.0:
            return []
        qty = float(position.qty or 0.0)
        ret = (float(mark) - avg_px) / avg_px
        hold_hours = None
        if position.entry_ts_ms is not None and now_ts_ms:
            hold_hours = max(0.0, (int(now_ts_ms) - int(position.entry_ts_ms)) / 3_600_000.0)

        base_meta = {
            "return_pct": ret,
            "mark_price": mark,
            "bid": bid,
            "ask": ask,
            "avg_px": avg_px,
            "hold_hours": hold_hours,
        }

        # Highest-priority defensive exits first.
        if self.config.stop_loss_pct is not None and ret <= -abs(float(self.config.stop_loss_pct)):
            return [self._decision(position, qty, bid, "stop_loss", False, base_meta)]

        if position.trailing_stop_price is not None and float(mark) <= float(position.trailing_stop_price):
            meta = {**base_meta, "peak_price": position.peak_price, "trailing_stop_price": position.trailing_stop_price}
            return [self._decision(position, qty, bid, "trailing_stop", False, meta)]

        if self.config.time_stop_hours is not None and hold_hours is not None:
            if hold_hours >= float(self.config.time_stop_hours) and ret < float(self.config.time_stop_min_return_pct):
                return [self._decision(position, qty, bid, "time_stop", False, base_meta)]

        if self.config.max_hold_hours is not None and hold_hours is not None and hold_hours >= float(self.config.max_hold_hours):
            return [self._decision(position, qty, bid, "max_hold", False, base_meta)]

        # Partial TP happens before full TP when enabled and not yet taken.
        if (
            self.config.partial_take_profit_pct is not None
            and not bool(position.partial_exit_taken)
            and ret >= float(self.config.partial_take_profit_pct)
        ):
            frac = min(1.0, max(0.0, float(self.config.partial_take_profit_fraction or 0.0)))
            partial_qty = qty * frac
            if partial_qty > 0.0 and partial_qty < qty:
                return [self._decision(position, partial_qty, bid, "partial_take_profit", True, base_meta)]

        if self.config.take_profit_pct is not None and ret >= float(self.config.take_profit_pct):
            return [self._decision(position, qty, bid, "take_profit", False, base_meta)]

        return []

    def to_order_intent(self, decision: ExitDecision) -> Optional[OrderIntent]:
        if not decision or not decision.should_exit or decision.qty <= 0.0 or decision.limit_px <= 0.0:
            return None
        return OrderIntent(side=decision.side, symbol=decision.symbol, qty=decision.qty, limit_px=decision.limit_px, tif=self.config.default_tif)

    def _decision(self, position: LivePositionState, qty: float, bid: float, reason: str, partial: bool, metadata: Dict[str, Any]) -> ExitDecision:
        slip = max(0.0, min(float(self.config.exit_slippage_pct or 0.0), 0.25))
        limit_px = float(bid) * (1.0 - slip)
        return ExitDecision(True, symbol=position.symbol, qty=float(qty), limit_px=limit_px, reason=reason, partial=partial, metadata=dict(metadata or {}))

    @staticmethod
    def _mark_price(bid: Optional[float], ask: Optional[float], fallback: Optional[float]) -> Optional[float]:
        if bid is not None and ask is not None and float(bid) > 0.0 and float(ask) > 0.0:
            return (float(bid) + float(ask)) / 2.0
        if bid is not None and float(bid) > 0.0:
            return float(bid)
        if ask is not None and float(ask) > 0.0:
            return float(ask)
        if fallback is not None and float(fallback) > 0.0:
            return float(fallback)
        return None

    @staticmethod
    def _symbol_candidates(position: LivePositionState, fallback: str) -> Iterable[str]:
        seen = []
        for s in [position.symbol, position.venue_symbol, fallback]:
            if s and s not in seen:
                seen.append(s)
        for s in list(seen):
            if "/" in s:
                seen.append(s.replace("/", "") + ("T" if s.endswith("USD") and not s.endswith("USDT") else ""))
            elif s.endswith("USDT"):
                seen.append(f"{s[:-4]}/USD")
        return seen
