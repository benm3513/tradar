from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("tradarbot.shadow_eval")


def now_ms() -> int:
    return int(time.time() * 1000)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


@dataclass
class ShadowPredictionRecord:
    ts_ms: int
    symbol: str
    mode: str = "shadow"
    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    prob: Optional[float] = None
    pred_prob: Optional[float] = None
    score: Optional[float] = None
    entry_score: Optional[float] = None
    prob_percentile_rank: Optional[float] = None
    rolling_volatility_24h: Optional[float] = None
    predicted_time_to_peak_hours: Optional[float] = None
    market_risk_off_score: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, ts_ms: int, symbol: str, payload: Dict[str, Any], mode: str = "shadow") -> "ShadowPredictionRecord":
        return cls(
            ts_ms=int(ts_ms),
            symbol=str(symbol),
            mode=str(mode),
            prediction_source=payload.get("prediction_source"),
            model_name=payload.get("model_name"),
            prob=_safe_float(payload.get("prob")),
            pred_prob=_safe_float(payload.get("pred_prob", payload.get("prob"))),
            score=_safe_float(payload.get("score")),
            entry_score=_safe_float(payload.get("entry_score", payload.get("score"))),
            prob_percentile_rank=_safe_float(payload.get("prob_percentile_rank")),
            rolling_volatility_24h=_safe_float(payload.get("rolling_volatility_24h")),
            predicted_time_to_peak_hours=_safe_float(payload.get("predicted_time_to_peak_hours")),
            market_risk_off_score=_safe_float(payload.get("market_risk_off_score")),
            payload=dict(payload or {}),
        )


@dataclass
class ShadowDecisionRecord:
    ts_ms: int
    symbol: str
    mode: str = "shadow"
    accepted: bool = False
    reject_reason: Optional[str] = None
    would_trade: bool = False
    would_side: Optional[str] = None
    would_qty: Optional[float] = None
    would_limit_px: Optional[float] = None
    would_notional_usd: Optional[float] = None
    prob: Optional[float] = None
    pred_prob: Optional[float] = None
    score: Optional[float] = None
    entry_score: Optional[float] = None
    prob_percentile_rank: Optional[float] = None
    rolling_volatility_24h: Optional[float] = None
    predicted_time_to_peak_hours: Optional[float] = None
    market_risk_off_score: Optional[float] = None
    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    prob_size_multiplier: Optional[float] = None
    vol_size_multiplier: Optional[float] = None
    kelly_fraction: Optional[float] = None
    kelly_multiplier: Optional[float] = None
    regime_size_multiplier: Optional[float] = None
    total_size_multiplier: Optional[float] = None
    top_n_rank: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowSignalRecord:
    ts_ms: int
    symbol: str
    mode: str = "shadow"
    action: str = "would_buy"
    side: str = "BUY"
    qty: Optional[float] = None
    limit_px: Optional[float] = None
    notional_usd: Optional[float] = None
    blocked_execution: bool = True
    prediction_source: Optional[str] = None
    model_name: Optional[str] = None
    prob: Optional[float] = None
    score: Optional[float] = None
    entry_score: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionComparison:
    ts_ms: int
    symbol: str
    comparison_type: str
    status: str
    shadow_signal_id: Optional[int] = None
    fill_id: Optional[int] = None
    lag_seconds: Optional[float] = None
    price_diff_pct: Optional[float] = None
    size_diff_pct: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowEvaluationSummary:
    since_ts_ms: Optional[int]
    prediction_count: int = 0
    signal_count: int = 0
    decision_count: int = 0
    would_trade_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    symbols: List[str] = field(default_factory=list)


def _call_store(store: Any, method: str, **kwargs) -> None:
    fn = getattr(store, method, None)
    if not callable(fn):
        return
    try:
        fn(**kwargs)
    except Exception:
        log.exception("shadow persistence failed method=%s", method)


def record_shadow_prediction(store: Any, record: ShadowPredictionRecord | Dict[str, Any], **kwargs) -> None:
    data = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record or {})
    data.update(kwargs)
    _call_store(store, "insert_ml_shadow_prediction", **data)


def record_shadow_decision(store: Any, record: ShadowDecisionRecord | Dict[str, Any], **kwargs) -> None:
    data = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record or {})
    data.update(kwargs)
    _call_store(store, "insert_ml_shadow_decision", **data)


def record_shadow_signal(store: Any, record: ShadowSignalRecord | Dict[str, Any], **kwargs) -> None:
    data = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record or {})
    data.update(kwargs)
    _call_store(store, "insert_ml_shadow_signal", **data)


def build_shadow_summary(store: Any, since_ts_ms: Optional[int] = None) -> Dict[str, Any]:
    try:
        if hasattr(store, "shadow_summary"):
            return store.shadow_summary(since_ts_ms=since_ts_ms)
    except Exception:
        log.exception("failed to build shadow summary from store helper")
    preds = getattr(store, "recent_ml_shadow_predictions", lambda **_: [])(limit=100000, since_ts_ms=since_ts_ms)
    sigs = getattr(store, "recent_ml_shadow_signals", lambda **_: [])(limit=100000, since_ts_ms=since_ts_ms)
    symbols = sorted({str(r.get("symbol")) for r in list(preds or []) + list(sigs or []) if r.get("symbol")})
    return asdict(ShadowEvaluationSummary(
        since_ts_ms=since_ts_ms,
        prediction_count=len(preds or []),
        signal_count=len(sigs or []),
        decision_count=0,
        would_trade_count=len([s for s in sigs or [] if s.get("blocked_execution") or s.get("action")]),
        symbols=symbols,
    ))


def compare_shadow_to_fills(store: Any, since_ts_ms: Optional[int] = None, tolerance_seconds: float = 300.0) -> Dict[str, Any]:
    try:
        signals = getattr(store, "recent_ml_shadow_signals", lambda **_: [])(limit=100000, since_ts_ms=since_ts_ms)
        fills = getattr(store, "recent_execution_fills", lambda **_: [])(limit=100000, since_ts_ms=since_ts_ms)
    except Exception as exc:
        return {"status": "FAIL", "error": str(exc), "comparisons": []}

    comparisons: List[Dict[str, Any]] = []
    fill_pool = list(fills or [])
    matched_fill_ids = set()
    tol_ms = int(float(tolerance_seconds) * 1000)

    for sig in signals or []:
        sig_ts = int(sig.get("ts_ms") or 0)
        sig_symbol = str(sig.get("symbol") or "")
        sig_side = str(sig.get("side") or sig.get("would_side") or "BUY").upper()
        candidates = [
            f for f in fill_pool
            if str(f.get("symbol") or "") == sig_symbol
            and str(f.get("side") or "").upper() == sig_side
            and abs(int(f.get("ts_ms") or 0) - sig_ts) <= tol_ms
        ]
        if candidates:
            fill = sorted(candidates, key=lambda f: abs(int(f.get("ts_ms") or 0) - sig_ts))[0]
            matched_fill_ids.add(fill.get("id"))
            lag = (int(fill.get("ts_ms") or 0) - sig_ts) / 1000.0
            sig_px = _safe_float(sig.get("limit_px")) or _safe_float(sig.get("would_limit_px"))
            fill_px = _safe_float(fill.get("px"))
            price_diff = None
            if sig_px and fill_px:
                price_diff = (fill_px - sig_px) / max(sig_px, 1e-12)
            sig_qty = _safe_float(sig.get("qty")) or _safe_float(sig.get("would_qty"))
            fill_qty = _safe_float(fill.get("qty"))
            size_diff = None
            if sig_qty and fill_qty:
                size_diff = (fill_qty - sig_qty) / max(sig_qty, 1e-12)
            comparisons.append(asdict(ExecutionComparison(
                ts_ms=now_ms(), symbol=sig_symbol, comparison_type="shadow_to_fill", status="MATCHED",
                shadow_signal_id=sig.get("id"), fill_id=fill.get("id"), lag_seconds=lag,
                price_diff_pct=price_diff, size_diff_pct=size_diff,
                details={"signal": sig, "fill": fill},
            )))
        else:
            comparisons.append(asdict(ExecutionComparison(
                ts_ms=now_ms(), symbol=sig_symbol, comparison_type="shadow_to_fill", status="MISSED_SHADOW_ENTRY",
                shadow_signal_id=sig.get("id"), details={"signal": sig},
            )))

    for fill in fill_pool:
        if fill.get("id") not in matched_fill_ids:
            comparisons.append(asdict(ExecutionComparison(
                ts_ms=now_ms(), symbol=str(fill.get("symbol") or ""), comparison_type="fill_without_shadow", status="PAPER_WITHOUT_SHADOW_AGREEMENT",
                fill_id=fill.get("id"), details={"fill": fill},
            )))

    for comp in comparisons:
        _call_store(store, "insert_ml_shadow_execution_comparison", **comp)
    status = "OK" if not comparisons or all(c.get("status") == "MATCHED" for c in comparisons) else "WARN"
    return {"status": status, "count": len(comparisons), "comparisons": comparisons}


def compare_paper_live_execution(store: Any, since_ts_ms: Optional[int] = None) -> Dict[str, Any]:
    try:
        rows = getattr(store, "recent_execution_fills", lambda **_: [])(limit=100000, since_ts_ms=since_ts_ms)
    except Exception as exc:
        return {"status": "FAIL", "error": str(exc), "comparisons": []}
    paper = [r for r in rows if str(r.get("broker_mode", "paper")).lower() == "paper"]
    live = [r for r in rows if str(r.get("broker_mode", "")).lower() in {"live", "dry_run_live", "dry-run-live", "dryrun"}]
    if not live:
        return {"status": "WARN", "message": "not enough live data", "paper_fills": len(paper), "live_fills": 0, "comparisons": []}
    return {"status": "OK", "paper_fills": len(paper), "live_fills": len(live), "comparisons": []}
