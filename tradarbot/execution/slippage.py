from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SlippageCheck:
    ok: bool
    slippage_pct: float
    spread_pct: float
    reason: Optional[str] = None


def spread_pct(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
        return 0.0
    mid = (bid + ask) / 2.0
    if mid <= 0.0:
        return 0.0
    return max(0.0, (ask - bid) / mid)


def reference_fill_price(side: str, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    side = str(side).upper()
    if side == "BUY":
        return ask
    if side == "SELL":
        return bid
    return None


def slippage_vs_reference(side: str, intended_px: float, actual_px: float) -> float:
    intended_px = float(intended_px or 0.0)
    actual_px = float(actual_px or 0.0)
    if intended_px <= 0.0 or actual_px <= 0.0:
        return 0.0

    side = str(side).upper()
    if side == "BUY":
        return max(0.0, (actual_px - intended_px) / intended_px)
    return max(0.0, (intended_px - actual_px) / intended_px)


def validate_execution_bounds(
    *,
    side: str,
    intended_px: float,
    actual_px: float,
    bid: Optional[float],
    ask: Optional[float],
    max_slippage_pct: Optional[float],
    max_spread_pct: Optional[float],
) -> SlippageCheck:
    current_spread_pct = spread_pct(bid, ask)
    if max_spread_pct is not None and current_spread_pct > float(max_spread_pct):
        return SlippageCheck(False, 0.0, current_spread_pct, "max_spread_pct")

    slippage_pct = slippage_vs_reference(side=side, intended_px=intended_px, actual_px=actual_px)
    if max_slippage_pct is not None and slippage_pct > float(max_slippage_pct):
        return SlippageCheck(False, slippage_pct, current_spread_pct, "max_slippage_pct")

    return SlippageCheck(True, slippage_pct, current_spread_pct, None)
