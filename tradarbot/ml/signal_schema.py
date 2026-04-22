from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from tradarbot.core.events import OrderIntent


@dataclass(frozen=True)
class MLSymbolFeatures:
    """Live feature row used by the Phase 5.0 ML path."""
    symbol: str
    timestamp: Optional[Any] = None
    price_close: Optional[float] = None

    prob_proxy: Optional[float] = None
    prob_percentile_rank: Optional[float] = None
    volatility_percentile_rank: Optional[float] = None

    return_1h: Optional[float] = None
    return_6h: Optional[float] = None
    return_24h: Optional[float] = None

    rolling_volatility_24h: Optional[float] = None
    range_pct_24h: Optional[float] = None
    drawup_from_recent_low_24h: Optional[float] = None
    price_zscore_24h: Optional[float] = None
    volume_zscore_24h: Optional[float] = None
    volume_spike_ratio_7d: Optional[float] = None
    momentum_accel_6h_vs_24h: Optional[float] = None
    trend_strength_local_24h: Optional[float] = None
    candle_efficiency_24h: Optional[float] = None

    target_time_to_peak_seconds_24h: Optional[float] = None
    predicted_time_to_peak_hours: Optional[float] = None

    market_risk_off_score: Optional[float] = None
    market_dispersion_1h: Optional[float] = None
    market_dispersion_24h: Optional[float] = None
    market_breadth_up_1h: Optional[float] = None
    market_breadth_up_24h: Optional[float] = None
    market_trend_strength_24h: Optional[float] = None
    market_volume_regime_24h: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MLSymbolFeatures":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: payload.get(k) for k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLPrediction:
    """Normalized predictor output for a single symbol."""
    symbol: str
    prob: float
    score: float

    entry_score: Optional[float] = None
    pred_prob: Optional[float] = None
    prob_percentile_rank: Optional[float] = None

    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    timestamp: Optional[Any] = None

    rolling_volatility_24h: Optional[float] = None
    predicted_time_to_peak_hours: Optional[float] = None
    target_time_to_peak_seconds_24h: Optional[float] = None

    market_risk_off_score: Optional[float] = None
    market_dispersion_24h: Optional[float] = None
    market_trend_strength_24h: Optional[float] = None
    market_volume_regime_24h: Optional[float] = None

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MLPrediction":
        fields = cls.__dataclass_fields__.keys()
        base = {k: payload.get(k) for k in fields if k != "raw"}
        base["raw"] = dict(payload)
        if base.get("entry_score") is None:
            base["entry_score"] = payload.get("score")
        if base.get("pred_prob") is None:
            base["pred_prob"] = payload.get("prob")
        return cls(**base)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if not out.get("raw"):
            out["raw"] = {
                "symbol": self.symbol,
                "prob": self.prob,
                "score": self.score,
                "entry_score": self.entry_score,
                "pred_prob": self.pred_prob,
                "prob_percentile_rank": self.prob_percentile_rank,
                "prediction_source": self.prediction_source,
                "model_name": self.model_name,
                "timestamp": self.timestamp,
            }
        return out


@dataclass(frozen=True)
class MLCandidate:
    """Cross-sectional candidate row after live feature + predictor merge."""
    symbol: str
    prob: float
    score: float

    entry_score: Optional[float] = None
    prob_percentile_rank: Optional[float] = None
    rolling_volatility_24h: Optional[float] = None
    predicted_time_to_peak_hours: Optional[float] = None

    market_risk_off_score: Optional[float] = None
    market_dispersion_24h: Optional[float] = None
    market_trend_strength_24h: Optional[float] = None
    market_volume_regime_24h: Optional[float] = None

    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    accepted: bool = False
    reject_reason: Optional[str] = None

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MLCandidate":
        fields = cls.__dataclass_fields__.keys()
        base = {k: payload.get(k) for k in fields if k != "raw"}
        base["raw"] = dict(payload)
        if base.get("entry_score") is None:
            base["entry_score"] = payload.get("score")
        return cls(**base)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLSizingDecision:
    """Sizing output after dynamic sizing / Kelly / regime scaling."""
    symbol: str
    approved: bool
    qty: float
    limit_px: float

    notional_usd: float
    base_notional_usd: float

    prob_size_multiplier: float = 1.0
    vol_size_multiplier: float = 1.0
    kelly_fraction: float = 0.0
    kelly_multiplier: float = 1.0
    regime_size_multiplier: float = 1.0
    total_size_multiplier: float = 1.0

    score: Optional[float] = None
    prob: Optional[float] = None
    reject_reason: Optional[str] = None

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MLSizingDecision":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: payload.get(k) for k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLEntrySignal:
    """Final strategy-level signal before conversion to OrderIntent."""
    symbol: str
    side: str
    qty: float
    limit_px: float

    score: Optional[float] = None
    prob: Optional[float] = None
    entry_score: Optional[float] = None
    notional_usd: Optional[float] = None

    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    tif: str = "IOC"

    prob_size_multiplier: float = 1.0
    vol_size_multiplier: float = 1.0
    kelly_fraction: float = 0.0
    kelly_multiplier: float = 1.0
    regime_size_multiplier: float = 1.0
    total_size_multiplier: float = 1.0

    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate_and_sizing(
        cls,
        candidate: MLCandidate,
        sizing: MLSizingDecision,
        side: str = "BUY",
        tif: str = "IOC",
    ) -> "MLEntrySignal":
        return cls(
            symbol=candidate.symbol,
            side=side,
            qty=sizing.qty,
            limit_px=sizing.limit_px,
            score=candidate.score,
            prob=candidate.prob,
            entry_score=candidate.entry_score,
            notional_usd=sizing.notional_usd,
            prediction_source=candidate.prediction_source,
            model_name=candidate.model_name,
            tif=tif,
            prob_size_multiplier=sizing.prob_size_multiplier,
            vol_size_multiplier=sizing.vol_size_multiplier,
            kelly_fraction=sizing.kelly_fraction,
            kelly_multiplier=sizing.kelly_multiplier,
            regime_size_multiplier=sizing.regime_size_multiplier,
            total_size_multiplier=sizing.total_size_multiplier,
            meta={
                "candidate": candidate.to_dict(),
                "sizing": sizing.to_dict(),
            },
        )

    def to_order_intent(self) -> OrderIntent:
        return OrderIntent(
            side=self.side,
            symbol=self.symbol,
            qty=self.qty,
            limit_px=self.limit_px,
            tif=self.tif,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLExitSignal:
    """Deterministic exit instruction for ML-managed positions."""
    symbol: str
    side: str
    qty: float
    limit_px: float
    reason: str
    tif: str = "IOC"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_order_intent(self) -> OrderIntent:
        return OrderIntent(
            side=self.side,
            symbol=self.symbol,
            qty=self.qty,
            limit_px=self.limit_px,
            tif=self.tif,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def merge_feature_and_prediction(
    features: MLSymbolFeatures,
    prediction: MLPrediction,
) -> MLCandidate:
    """Build a candidate row from typed feature + prediction payloads."""
    merged = features.to_dict()
    merged.update(prediction.to_dict())

    return MLCandidate.from_dict(
        {
            "symbol": merged["symbol"],
            "prob": merged.get("prob", merged.get("pred_prob", 0.0)),
            "score": merged.get("score", merged.get("entry_score", 0.0)),
            "entry_score": merged.get("entry_score", merged.get("score", 0.0)),
            "prob_percentile_rank": merged.get("prob_percentile_rank"),
            "rolling_volatility_24h": merged.get("rolling_volatility_24h"),
            "predicted_time_to_peak_hours": merged.get(
                "predicted_time_to_peak_hours",
                merged.get("time_to_peak_hours"),
            ),
            "market_risk_off_score": merged.get("market_risk_off_score"),
            "market_dispersion_24h": merged.get("market_dispersion_24h"),
            "market_trend_strength_24h": merged.get("market_trend_strength_24h"),
            "market_volume_regime_24h": merged.get("market_volume_regime_24h"),
            "prediction_source": merged.get("prediction_source"),
            "model_name": merged.get("model_name"),
            "raw": merged,
        }
    )
