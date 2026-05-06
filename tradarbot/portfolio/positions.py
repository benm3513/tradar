from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PositionOwner:
    strategy_name: Optional[str] = None
    signal_id: Optional[str] = None
    model_name: Optional[str] = None
    prediction_source: Optional[str] = None
    entry_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["PositionOwner"]:
        if not payload:
            return None
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: payload.get(k) for k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LivePositionState:
    symbol: str
    venue_symbol: Optional[str] = None
    qty: float = 0.0
    avg_px: float = 0.0
    entry_ts_ms: Optional[int] = None
    last_update_ts_ms: Optional[int] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    partial_exit_taken: bool = False
    owner: Optional[PositionOwner] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LivePositionState":
        data = dict(payload or {})
        owner = data.get("owner")
        if isinstance(owner, dict):
            data["owner"] = PositionOwner.from_dict(owner)
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: data.get(k) for k in fields})

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        return out


@dataclass
class PortfolioSnapshot:
    ts_ms: int
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_exposure: float
    positions: Dict[str, LivePositionState] = field(default_factory=dict)
    broker_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PortfolioSnapshot":
        data = dict(payload or {})
        positions = data.get("positions") or {}
        data["positions"] = {
            str(sym): (pos if isinstance(pos, LivePositionState) else LivePositionState.from_dict(pos))
            for sym, pos in dict(positions).items()
        }
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: data.get(k) for k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "cash": self.cash,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_exposure": self.total_exposure,
            "broker_mode": self.broker_mode,
            "metadata": dict(self.metadata or {}),
            "positions": {sym: pos.to_dict() for sym, pos in dict(self.positions or {}).items()},
        }


def position_unrealized_pnl(position: LivePositionState, mark_price: Optional[float]) -> float:
    if mark_price is None or position is None:
        return 0.0
    qty = float(getattr(position, "qty", 0.0) or 0.0)
    avg_px = float(getattr(position, "avg_px", 0.0) or 0.0)
    if qty <= 0.0 or avg_px <= 0.0:
        return 0.0
    return (float(mark_price) - avg_px) * qty


def position_notional(position: LivePositionState, mark_price: Optional[float] = None) -> float:
    if position is None:
        return 0.0
    qty = float(getattr(position, "qty", 0.0) or 0.0)
    price = mark_price
    if price is None:
        price = getattr(position, "current_price", None)
    if price is None:
        price = getattr(position, "avg_px", 0.0)
    return max(0.0, qty * float(price or 0.0))


def update_peak_and_trailing(
    position: LivePositionState,
    mark_price: Optional[float],
    trailing_stop_pct: Optional[float],
    activation_pct: Optional[float],
) -> LivePositionState:
    if position is None or mark_price is None:
        return position
    price = float(mark_price or 0.0)
    if price <= 0.0 or float(position.avg_px or 0.0) <= 0.0:
        return position

    position.current_price = price
    position.unrealized_pnl = position_unrealized_pnl(position, price)
    ret = (price - float(position.avg_px)) / float(position.avg_px)
    activation = float(activation_pct or 0.0)
    trail = float(trailing_stop_pct or 0.0)
    if trail <= 0.0 or ret < activation:
        return position

    if position.peak_price is None or price > float(position.peak_price):
        position.peak_price = price
    if position.peak_price is not None:
        position.trailing_stop_price = float(position.peak_price) * (1.0 - trail)
    return position
