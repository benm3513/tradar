from __future__ import annotations

import json
import logging
import math
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

log = logging.getLogger("tradarbot.parity_checks")

EXPECTED_FEATURE_COLUMNS = [
    "symbol", "timestamp", "price_close", "prob_proxy", "return_1h", "return_6h", "return_24h",
    "rolling_volatility_24h", "range_pct_24h", "drawup_from_recent_low_24h", "price_zscore_24h",
    "volume_zscore_24h", "volume_spike_ratio_7d", "momentum_accel_6h_vs_24h",
    "trend_strength_local_24h", "candle_efficiency_24h", "target_time_to_peak_seconds_24h",
    "predicted_time_to_peak_hours", "market_risk_off_score", "market_dispersion_24h",
    "market_trend_strength_24h", "market_volume_regime_24h",
]
EXPECTED_PREDICTION_FIELDS = ["symbol", "prob", "pred_prob", "score", "entry_score", "prediction_source", "model_name"]


def _status(failures: List[str], warnings: List[str]) -> str:
    return "FAIL" if failures else ("WARN" if warnings else "OK")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def check_live_feature_columns(feature_df: pd.DataFrame, expected_columns: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    expected = list(expected_columns or EXPECTED_FEATURE_COLUMNS)
    failures: List[str] = []
    warnings: List[str] = []
    checks: List[Dict[str, Any]] = []
    if feature_df is None or feature_df.empty:
        failures.append("feature_df is empty")
        return {"status": "FAIL", "checks": checks, "failures": failures, "warnings": warnings, "metrics": {"rows": 0}}
    missing = [c for c in expected if c not in feature_df.columns]
    if missing:
        warnings.append(f"missing expected live feature columns: {missing}")
    checks.append({"name": "feature_columns_present", "missing": missing, "passed": not missing})
    numeric_cols = [c for c in feature_df.columns if c != "symbol"]
    nan_cols = []
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(feature_df[col]) and feature_df[col].isna().all():
            nan_cols.append(col)
    if nan_cols:
        warnings.append(f"all-null numeric feature columns: {nan_cols}")
    return {
        "status": _status(failures, warnings),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "metrics": {"rows": len(feature_df), "columns": len(feature_df.columns), "missing_columns": len(missing), "all_null_numeric_columns": len(nan_cols)},
    }


def check_prediction_payload(prediction_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    checks: List[Dict[str, Any]] = []
    if not prediction_map:
        failures.append("prediction_map is empty")
        return {"status": "FAIL", "checks": checks, "failures": failures, "warnings": warnings, "metrics": {"predictions": 0}}
    for symbol, payload in prediction_map.items():
        missing = [f for f in EXPECTED_PREDICTION_FIELDS if payload.get(f) is None]
        if missing:
            warnings.append(f"{symbol} missing prediction fields: {missing}")
        prob = _safe_float(payload.get("pred_prob", payload.get("prob")), -1.0)
        if prob < 0.0 or prob > 1.0:
            failures.append(f"{symbol} pred_prob outside [0,1]: {prob}")
        checks.append({"name": "prediction_payload", "symbol": symbol, "missing": missing, "prob": prob, "passed": not missing and 0.0 <= prob <= 1.0})
    return {"status": _status(failures, warnings), "checks": checks, "failures": failures, "warnings": warnings, "metrics": {"predictions": len(prediction_map)}}


def check_ranked_candidate_parity(candidate_df: pd.DataFrame, runtime_args: Any) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    checks: List[Dict[str, Any]] = []
    if candidate_df is None or candidate_df.empty:
        warnings.append("candidate_df is empty")
        return {"status": "WARN", "checks": checks, "failures": failures, "warnings": warnings, "metrics": {"candidates": 0}}
    try:
        from scripts.replay_ml_strategy import filter_ranked_candidates
        diagnostics = SimpleNamespace(
            candidate_rows_seen=0, candidate_rows_after_prob_threshold=0, candidate_rows_after_percentile=0,
            candidate_rows_after_volatility=0, candidate_rows_after_time_to_peak=0, candidate_rows_after_rank_score=0,
            candidate_rows_after_regime_gate=0, regime_gate_blocks=0, regime_scale_events=0, regime_score_raise_events=0,
        )
        ranked = filter_ranked_candidates(candidate_df.copy(), runtime_args, diagnostics)
        ranked_symbols = ranked["symbol"].astype(str).tolist() if isinstance(ranked, pd.DataFrame) and "symbol" in ranked.columns else []
        checks.append({"name": "replay_filter_path", "passed": True, "ranked_symbols": ranked_symbols, "diagnostics": vars(diagnostics)})
        return {
            "status": _status(failures, warnings),
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
            "metrics": {"candidates": len(candidate_df), "ranked": len(ranked_symbols)},
        }
    except Exception as exc:
        failures.append(f"replay-compatible filter path failed: {exc}")
        return {"status": "FAIL", "checks": checks, "failures": failures, "warnings": warnings, "metrics": {"candidates": len(candidate_df)}}


def build_recent_window_parity_report(store: Any, symbols: Optional[List[str]] = None, since_ts_ms: Optional[int] = None, limit: int = 500, write: bool = False) -> Dict[str, Any]:
    warnings: List[str] = []
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    try:
        preds = getattr(store, "recent_ml_shadow_predictions", lambda **_: [])(limit=limit, since_ts_ms=since_ts_ms)
        sigs = getattr(store, "recent_ml_shadow_signals", lambda **_: [])(limit=limit, since_ts_ms=since_ts_ms)
    except Exception as exc:
        failures.append(f"failed reading shadow tables: {exc}")
        preds, sigs = [], []
    if symbols:
        allowed = {str(s) for s in symbols}
        preds = [p for p in preds if str(p.get("symbol")) in allowed]
        sigs = [s for s in sigs if str(s.get("symbol")) in allowed]
    metrics.update({"shadow_predictions": len(preds), "shadow_signals": len(sigs)})
    if not preds:
        warnings.append("no recent shadow predictions available for parity report")
    else:
        bad_prob = [p for p in preds if not (0.0 <= _safe_float(p.get("pred_prob", p.get("prob")), -1.0) <= 1.0)]
        if bad_prob:
            failures.append(f"{len(bad_prob)} shadow predictions have invalid probabilities")
        missing_source = [p for p in preds if not p.get("prediction_source")]
        if missing_source:
            warnings.append(f"{len(missing_source)} shadow predictions missing prediction_source")
        checks.append({"name": "shadow_prediction_probability_bounds", "passed": not bad_prob})
    report = {"status": _status(failures, warnings), "checks": checks, "failures": failures, "warnings": warnings, "metrics": metrics, "since_ts_ms": since_ts_ms}
    if write and hasattr(store, "insert_ml_replay_parity_check"):
        try:
            store.insert_ml_replay_parity_check(ts_ms=int(time.time() * 1000), status=report["status"], check_name="recent_window", symbols=symbols or [], metrics=metrics, failures=failures, warnings=warnings, payload=report)
        except Exception:
            log.exception("failed to persist parity report")
    return report
