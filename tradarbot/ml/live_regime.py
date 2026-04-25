from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    roll_mean = series.rolling(window=window, min_periods=max(3, window // 4)).mean()
    roll_std = series.rolling(window=window, min_periods=max(3, window // 4)).std(ddof=0)
    denom = roll_std.replace(0.0, pd.NA)
    return ((series - roll_mean) / denom).infer_objects(copy=False).fillna(0.0)


def _empty_regime() -> Dict[str, float]:
    return {
        "market_dispersion_1h": 0.0,
        "market_dispersion_24h": 0.0,
        "market_breadth_up_1h": 0.0,
        "market_breadth_up_24h": 0.0,
        "market_trend_strength_24h": 0.0,
        "market_volume_regime_24h": 0.0,
        "market_risk_off_score": 0.5,
    }


def compute_live_regime(symbol_frames: Mapping[str, pd.DataFrame]) -> Dict[str, float]:
    """Compute cross-sectional live regime features from recent symbol candles.

    This keeps the same feature family used by the Phase 4 replay tables:
    breadth, dispersion, trend strength, volume regime, and market_risk_off_score.
    """
    one_bar_returns = []
    day_returns = []
    normalized_day_returns = []
    volume_zscores = []
    breadth_up_1h = 0
    breadth_up_24h = 0
    considered = 0

    for _, frame in dict(symbol_frames or {}).items():
        if frame is None or frame.empty or len(frame) < 3 or "close" not in frame.columns:
            continue

        close = pd.to_numeric(frame["close"], errors="coerce").dropna().astype(float)
        if len(close) < 3:
            continue

        ret_1 = _safe_float(close.iloc[-1] / max(close.iloc[-2], 1e-12) - 1.0)
        lookback_24_idx = max(0, len(close) - min(24, len(close)))
        ret_24 = _safe_float(close.iloc[-1] / max(close.iloc[lookback_24_idx], 1e-12) - 1.0)

        rets = close.pct_change().dropna()
        vol_24 = _safe_float(rets.tail(24).std(ddof=0), default=0.0)
        norm_day = ret_24 / max(vol_24, 1e-6)

        if "volume" in frame.columns:
            vol_series = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).astype(float)
        else:
            vol_series = pd.Series([0.0] * len(frame))
        vol_z = _safe_float(_rolling_zscore(vol_series, 24).iloc[-1] if len(vol_series) else 0.0)

        one_bar_returns.append(ret_1)
        day_returns.append(ret_24)
        normalized_day_returns.append(norm_day)
        volume_zscores.append(vol_z)
        breadth_up_1h += 1 if ret_1 > 0 else 0
        breadth_up_24h += 1 if ret_24 > 0 else 0
        considered += 1

    if considered == 0:
        return _empty_regime()

    dispersion_1h = _safe_float(pd.Series(one_bar_returns).std(ddof=0), 0.0)
    dispersion_24h = _safe_float(pd.Series(day_returns).std(ddof=0), 0.0)
    breadth_1h = breadth_up_1h / considered
    breadth_24h = breadth_up_24h / considered
    trend_strength = abs(_safe_float(pd.Series(normalized_day_returns).mean(), 0.0))
    volume_regime = _safe_float(pd.Series(volume_zscores).mean(), 0.0)

    breadth_penalty = 1.0 - breadth_24h
    trend_penalty = 1.0 / (1.0 + trend_strength)
    dispersion_term = min(1.0, dispersion_24h / 0.08)
    volume_term = 1.0 / (1.0 + max(volume_regime, -0.95))

    risk_off_score = _bounded(
        0.40 * dispersion_term
        + 0.25 * breadth_penalty
        + 0.20 * trend_penalty
        + 0.15 * volume_term
    )

    return {
        "market_dispersion_1h": dispersion_1h,
        "market_dispersion_24h": dispersion_24h,
        "market_breadth_up_1h": breadth_1h,
        "market_breadth_up_24h": breadth_24h,
        "market_trend_strength_24h": trend_strength,
        "market_volume_regime_24h": volume_regime,
        "market_risk_off_score": risk_off_score,
    }
