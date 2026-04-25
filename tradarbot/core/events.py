from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Existing core market/execution events
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Phase 5.0 ML strategy events
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MLFeatureEvent:
    """Snapshot of live features for one symbol at one timestamp.

    Produced by the Phase 5.0 live feature layer and consumable by
    ML strategy components or downstream logging/inspection.
    """
    symbol: str
    ts_ms: int
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MLPredictionEvent:
    """Normalized predictor output for one symbol.

    Carries the per-symbol probability and ranking score used by the
    ML strategy before candidate filtering and top-N selection.
    """
    symbol: str
    ts_ms: int
    prob: float
    score: float
    entry_score: Optional[float] = None
    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MLCandidateEvent:
    """Candidate row after feature/prediction merge but before final entry intent."""
    symbol: str
    ts_ms: int
    prob: float
    score: float
    entry_score: Optional[float] = None
    prob_percentile_rank: Optional[float] = None
    rolling_volatility_24h: Optional[float] = None
    predicted_time_to_peak_hours: Optional[float] = None
    market_risk_off_score: Optional[float] = None
    accepted: bool = False
    reject_reason: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MLRankingEvent:
    """Cross-sectional ranking output for a batch of ML candidates.

    This is the event-type equivalent of the ranked candidate table used in
    replay, adapted for live strategy inspection and optional event-bus use.
    """
    ts_ms: int
    symbols_ranked: List[str] = field(default_factory=list)
    top_n: int = 0
    ranking_mode: str = "composite"
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MLEntryRequestEvent:
    """Pre-execution ML entry request.

    This is intentionally richer than OrderIntent so the strategy can keep
    sizing/risk metadata attached while still converting to OrderIntent for
    the existing broker path.
    """
    symbol: str
    ts_ms: int
    side: str
    qty: float
    limit_px: float
    tif: str = "IOC"

    prob: Optional[float] = None
    score: Optional[float] = None
    entry_score: Optional[float] = None
    notional_usd: Optional[float] = None

    prob_size_multiplier: float = 1.0
    vol_size_multiplier: float = 1.0
    kelly_fraction: float = 0.0
    kelly_multiplier: float = 1.0
    regime_size_multiplier: float = 1.0
    total_size_multiplier: float = 1.0

    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_order_intent(self) -> OrderIntent:
        return OrderIntent(
            side=self.side,
            symbol=self.symbol,
            qty=self.qty,
            limit_px=self.limit_px,
            tif=self.tif,
        )


@dataclass(frozen=True)
class MLExitRequestEvent:
    """Deterministic ML exit request before conversion to OrderIntent."""
    symbol: str
    ts_ms: int
    side: str
    qty: float
    limit_px: float
    reason: str
    tif: str = "IOC"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_order_intent(self) -> OrderIntent:
        return OrderIntent(
            side=self.side,
            symbol=self.symbol,
            qty=self.qty,
            limit_px=self.limit_px,
            tif=self.tif,
        )


@dataclass(frozen=True)
class MLSignalEvent:
    """Final ML strategy signal event.

    This is a compact event for downstream logging / UI / persistence when
    the strategy has already decided to act.
    """
    symbol: str
    ts_ms: int
    action: str
    prob: Optional[float] = None
    score: Optional[float] = None
    entry_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BookEvent",
    "CandleEvent",
    "ListingEvent",
    "OrderIntent",
    "MLFeatureEvent",
    "MLPredictionEvent",
    "MLCandidateEvent",
    "MLRankingEvent",
    "MLEntryRequestEvent",
    "MLExitRequestEvent",
    "MLSignalEvent",
    "FeatureStateUpdatedEvent",
    "RegimeContextEvent",
    "LiveContextSnapshotEvent",
    "MarketDataHealthEvent",
]

# ---------------------------------------------------------------------
# Phase 5.2 live market-data/context events
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureStateUpdatedEvent:
    ts_ms: int
    ready_symbols: List[str] = field(default_factory=list)
    health: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeContextEvent:
    ts_ms: int
    regime: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveContextSnapshotEvent:
    ts_ms: int
    ready_symbols: List[str] = field(default_factory=list)
    feature_rows: int = 0
    regime: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketDataHealthEvent:
    ts_ms: int
    health: Dict[str, Any] = field(default_factory=dict)
