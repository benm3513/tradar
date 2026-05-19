from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Deque, Dict, Optional

LOGGER = logging.getLogger("tradarbot.calibration")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def clip_probability(prob: Any, lo: float = 0.0, hi: float = 0.95) -> float:
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, _safe_float(prob, 0.0)))


def dampen_probability(prob: Any, factor: float = 0.75, baseline: float = 0.5) -> float:
    """Compress probability confidence toward a neutral baseline.

    factor=1.0 leaves the probability unchanged. factor=0.75 turns 0.98 into
    0.86, preserving ordering while preventing every heuristic score from
    looking like a near-certain spike.
    """
    p = _safe_float(prob, baseline)
    factor = max(0.0, min(1.0, float(factor)))
    return baseline + (p - baseline) * factor


def normalize_probability(prob: Any, history: Optional[Deque[float]] = None, window: int = 100) -> float:
    p = _safe_float(prob, 0.0)
    if history is None or len(history) < max(10, min(window, 20)):
        return p
    vals = sorted(float(v) for v in history if v is not None)
    if not vals:
        return p
    below = sum(1 for v in vals if v <= p)
    pct = below / max(len(vals), 1)
    # Blend raw probability and rolling percentile so ranking is preserved but
    # local calibration can adapt as the live heuristic drifts.
    return 0.65 * p + 0.35 * pct


class ProbabilityCalibrator:
    """Online heuristic probability calibration helper."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        self.clip_min = float(cfg.get("probability_clip_min", 0.0) or 0.0)
        self.clip_max = float(cfg.get("probability_clip_max", 0.95) or 0.95)
        self.dampening_factor = float(cfg.get("confidence_dampening_factor", 0.75) or 0.75)
        self.percentile_smoothing_window = int(cfg.get("percentile_smoothing_window", 100) or 100)
        self.enable_rolling_normalization = bool(cfg.get("enable_probability_rolling_normalization", False))
        self._history: Dict[str, Deque[float]] = {}

    def calibrate_payload(self, symbol: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload or {})
        source = str(out.get("prediction_source") or "").lower()
        predictor_mode = str(out.get("model_name") or "").lower()
        is_heuristic = "heuristic" in source or "heuristic" in predictor_mode
        if not is_heuristic:
            return out

        raw_prob = _safe_float(out.get("pred_prob", out.get("prob", 0.0)), 0.0)
        history = self._history.setdefault(str(symbol), deque(maxlen=max(10, self.percentile_smoothing_window)))
        normalized = normalize_probability(raw_prob, history, self.percentile_smoothing_window) if self.enable_rolling_normalization else raw_prob
        dampened = dampen_probability(normalized, self.dampening_factor)
        calibrated = clip_probability(dampened, self.clip_min, self.clip_max)
        history.append(raw_prob)

        out["raw_prob"] = raw_prob
        out["raw_pred_prob"] = _safe_float(out.get("pred_prob", raw_prob), raw_prob)
        out["uncalibrated_score"] = _safe_float(out.get("score", raw_prob), raw_prob)
        out["prob"] = calibrated
        out["pred_prob"] = calibrated
        out["score"] = calibrated
        out["entry_score"] = calibrated
        out["calibrated"] = True
        out["calibration_method"] = "clip+dampen"
        out["calibration_factor"] = self.dampening_factor
        out["probability_clip_max"] = self.clip_max

        LOGGER.info(
            "ML_PROBABILITY_CALIBRATED symbol=%s raw=%.6f calibrated=%.6f factor=%.3f clip=[%.3f,%.3f]",
            symbol,
            raw_prob,
            calibrated,
            self.dampening_factor,
            self.clip_min,
            self.clip_max,
        )
        if calibrated != dampened:
            LOGGER.info("ML_PROBABILITY_CLIPPED symbol=%s before=%.6f after=%.6f", symbol, dampened, calibrated)
        if dampened != normalized:
            LOGGER.info("ML_PROBABILITY_DAMPENED symbol=%s before=%.6f after=%.6f", symbol, normalized, dampened)
        return out


def calibrate_heuristic_prediction(symbol: str, payload: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return ProbabilityCalibrator(cfg).calibrate_payload(symbol, payload)
