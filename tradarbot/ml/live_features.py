from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd


DEFAULT_HISTORY_BARS = 24 * 7  # 7 days of hourly bars by default


@dataclass
class LiveFeatureConfig:
    lookback_bars: int = DEFAULT_HISTORY_BARS
    price_column: str = "close"
    volume_column: str = "volume"
    candles_table: str = "candles"
    interval_s: int = 3600


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _choose_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    roll_mean = series.rolling(window=window, min_periods=max(3, window // 4)).mean()
    roll_std = series.rolling(window=window, min_periods=max(3, window // 4)).std(ddof=0)
    denom = roll_std.replace(0.0, pd.NA)
    z = (series - roll_mean) / denom
    return z.infer_objects(copy=False).fillna(0.0)


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _get_store_connection(ctx: Any) -> Optional[sqlite3.Connection]:
    store = getattr(ctx, "store", None)
    if store is None:
        return None

    for attr in ("conn", "_conn", "connection"):
        conn = getattr(store, attr, None)
        if isinstance(conn, sqlite3.Connection):
            return conn

    db_path = getattr(store, "db_path", None)
    if db_path:
        try:
            return sqlite3.connect(db_path)
        except Exception:
            return None
    return None


def load_symbol_candles(
    symbol: str,
    ctx: Any,
    lookback_bars: int = DEFAULT_HISTORY_BARS,
    interval_s: Optional[int] = None,
    candles_table: str = "candles",
) -> pd.DataFrame:
    """Load recent candles for a symbol from the runtime store.

    This helper is defensive because the exact SQLiteStore interface may vary.
    It prefers a live sqlite connection on ctx.store and falls back gracefully.
    """
    conn = _get_store_connection(ctx)
    if conn is None:
        return pd.DataFrame()

    interval_s = int(
        interval_s
        or getattr(getattr(ctx, "cfg", {}), "get", lambda *_: {})("runtime", {}).get("candle_interval_s", 3600)
    )

    queries = [
        f'''
        SELECT symbol, ts_ms, open, high, low, close, volume
        FROM "{candles_table}"
        WHERE symbol = ?
          AND interval_s = ?
        ORDER BY ts_ms DESC
        LIMIT ?
        ''',
        f'''
        SELECT symbol, ts_ms, open, high, low, close, volume
        FROM "{candles_table}"
        WHERE symbol = ?
        ORDER BY ts_ms DESC
        LIMIT ?
        ''',
    ]

    for idx, query in enumerate(queries):
        try:
            params = (symbol, interval_s, int(lookback_bars)) if idx == 0 else (symbol, int(lookback_bars))
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                break
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        return df

    if "ts_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = _coerce_timestamp(df["timestamp"])
    else:
        return pd.DataFrame()

    numeric_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)


def _compute_market_regime_features(
    current_symbol: str,
    symbol_frames: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Cross-sectional market context from all currently active symbols.

    These proxies are designed to preserve the regime feature family used in replay:
    breadth / dispersion / trend / volume regime / risk-off score.
    """
    one_bar_returns = []
    day_returns = []
    normalized_day_returns = []
    volume_zscores = []
    breadth_up_1h = 0
    breadth_up_24h = 0
    considered = 0

    for sym, frame in symbol_frames.items():
        if frame is None or frame.empty or len(frame) < 3:
            continue

        close = frame["close"].astype(float)
        ret_1 = _safe_float(close.iloc[-1] / close.iloc[-2] - 1.0)
        ret_24 = _safe_float(close.iloc[-1] / close.iloc[max(0, len(close) - min(24, len(close)))] - 1.0)

        rets = close.pct_change().dropna()
        vol_24 = _safe_float(rets.tail(24).std(ddof=0), default=0.0)
        norm_day = ret_24 / max(vol_24, 1e-6)

        vol_series = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series([0.0] * len(frame))
        vol_z = _safe_float(_rolling_zscore(vol_series, 24).iloc[-1] if len(vol_series) else 0.0)

        one_bar_returns.append(ret_1)
        day_returns.append(ret_24)
        normalized_day_returns.append(norm_day)
        volume_zscores.append(vol_z)
        breadth_up_1h += 1 if ret_1 > 0 else 0
        breadth_up_24h += 1 if ret_24 > 0 else 0
        considered += 1

    if considered == 0:
        return {
            "market_dispersion_1h": 0.0,
            "market_dispersion_24h": 0.0,
            "market_breadth_up_1h": 0.0,
            "market_breadth_up_24h": 0.0,
            "market_trend_strength_24h": 0.0,
            "market_volume_regime_24h": 0.0,
            "market_risk_off_score": 0.5,
        }

    dispersion_1h = _safe_float(pd.Series(one_bar_returns).std(ddof=0), 0.0)
    dispersion_24h = _safe_float(pd.Series(day_returns).std(ddof=0), 0.0)
    breadth_1h = breadth_up_1h / considered
    breadth_24h = breadth_up_24h / considered
    trend_strength = abs(_safe_float(pd.Series(normalized_day_returns).mean(), 0.0))
    volume_regime = _safe_float(pd.Series(volume_zscores).mean(), 0.0)

    # Heuristic risk-off proxy:
    # more dispersion + weaker breadth + weaker trend => more risk off
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


def compute_symbol_live_features(
    symbol: str,
    candles_df: pd.DataFrame,
    market_regime: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute a replay-compatible live feature row from candles only.

    Phase 5.0 requirement alignment:
    - probability proxy inputs
    - volatility
    - time-to-peak estimate
    - percentile ranking context
    - regime family columns that replay uses for gating
    """
    if candles_df is None or candles_df.empty:
        return {
            "symbol": symbol,
            "prob_proxy": 0.0,
            "prob": 0.0,
            "volatility": 0.0,
            "rolling_volatility_24h": 0.0,
            "target_time_to_peak_seconds_24h": 24.0 * 3600.0,
            "time_to_peak": 24.0,
            "time_to_peak_hours": 24.0,
        }

    df = candles_df.copy()
    if "timestamp" not in df.columns and "ts_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = _coerce_timestamp(df["timestamp"])

    close_col = _choose_column(df, ("close", "price_close"))
    volume_col = _choose_column(df, ("volume", "volume_base"))
    high_col = _choose_column(df, ("high", "price_high"))
    low_col = _choose_column(df, ("low", "price_low"))
    open_col = _choose_column(df, ("open", "price_open"))

    if close_col is None:
        raise KeyError("live_features.py requires a close/price_close column")

    if volume_col is None:
        df["volume"] = 0.0
        volume_col = "volume"

    close = pd.to_numeric(df[close_col], errors="coerce").astype(float)
    volume = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0).astype(float)

    high = pd.to_numeric(df[high_col], errors="coerce").astype(float) if high_col else close
    low = pd.to_numeric(df[low_col], errors="coerce").astype(float) if low_col else close
    open_ = pd.to_numeric(df[open_col], errors="coerce").astype(float) if open_col else close

    returns = close.pct_change().fillna(0.0)
    last_close = _safe_float(close.iloc[-1], 0.0)
    prev_close = _safe_float(close.iloc[-2], last_close) if len(close) >= 2 else last_close

    ret_1h = _safe_float(last_close / max(prev_close, 1e-12) - 1.0, 0.0)
    ret_6h = _safe_float(last_close / max(_safe_float(close.iloc[max(0, len(close) - min(6, len(close)))]), 1e-12) - 1.0, 0.0)
    ret_24h = _safe_float(last_close / max(_safe_float(close.iloc[max(0, len(close) - min(24, len(close)))]), 1e-12) - 1.0, 0.0)

    rolling_volatility_24h = _safe_float(returns.tail(24).std(ddof=0), 0.0)
    range_pct_24h = _safe_float((high.tail(24).max() - low.tail(24).min()) / max(last_close, 1e-12), 0.0)
    drawup_from_recent_low_24h = _safe_float((last_close - low.tail(24).min()) / max(low.tail(24).min(), 1e-12), 0.0)

    price_zscore_24h = _safe_float(_rolling_zscore(close, 24).iloc[-1], 0.0)
    clean_volume = volume.replace(0.0, pd.NA).infer_objects(copy=False).fillna(0.0)

    volume_zscore_24h = _safe_float(
        _rolling_zscore(clean_volume, 24).iloc[-1],
        0.0
    )

    vol_ma_24h = _safe_float(volume.tail(24).mean(), 0.0)
    vol_ma_7d = _safe_float(volume.tail(24 * 7).mean(), vol_ma_24h)
    volume_spike_ratio_7d = _safe_float(vol_ma_24h / max(vol_ma_7d, 1e-12), 0.0)

    momentum_accel_6h_vs_24h = _safe_float(ret_6h - (ret_24h / 4.0), 0.0)

    candle_body = (close - open_).abs().tail(24).mean()
    wick_range = (high - low).tail(24).mean()
    efficiency = _safe_float(candle_body / max(wick_range, 1e-12), 0.0)

    trend_strength_local = _safe_float(abs(ret_24h) / max(rolling_volatility_24h, 1e-6), 0.0)

    # Heuristic time-to-peak estimate:
    # stronger acceleration + stronger volume + stronger trend -> shorter expected path
    speed_score = max(
        0.0,
        1.5 * momentum_accel_6h_vs_24h
        + 0.8 * ret_6h
        + 0.5 * volume_zscore_24h
        + 0.25 * trend_strength_local,
    )
    time_to_peak_hours = _bounded(24.0 / (1.0 + speed_score), lo=2.0, hi=24.0)
    target_time_to_peak_seconds_24h = time_to_peak_hours * 3600.0

    # Heuristic probability proxy for Phase 5.0 predictor inputs.
    prob_proxy_raw = (
        1.40 * momentum_accel_6h_vs_24h
        + 0.90 * ret_6h
        + 0.50 * ret_24h
        + 0.35 * volume_zscore_24h
        + 0.30 * volume_spike_ratio_7d
        + 0.20 * drawup_from_recent_low_24h
        + 0.15 * range_pct_24h
        + 0.10 * efficiency
        - 0.45 * rolling_volatility_24h
    )
    prob_proxy = 1.0 / (1.0 + math.exp(-4.0 * prob_proxy_raw))

    feature_row: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
        "price_close": last_close,
        "return_1h": ret_1h,
        "return_6h": ret_6h,
        "return_24h": ret_24h,
        "rolling_volatility_24h": rolling_volatility_24h,
        "volatility": rolling_volatility_24h,
        "range_pct_24h": range_pct_24h,
        "drawup_from_recent_low_24h": drawup_from_recent_low_24h,
        "price_zscore_24h": price_zscore_24h,
        "volume_zscore_24h": volume_zscore_24h,
        "volume_spike_ratio_7d": volume_spike_ratio_7d,
        "momentum_accel_6h_vs_24h": momentum_accel_6h_vs_24h,
        "trend_strength_local_24h": trend_strength_local,
        "candle_efficiency_24h": efficiency,
        "prob_proxy": prob_proxy,
        "prob": prob_proxy,
        "target_time_to_peak_seconds_24h": target_time_to_peak_seconds_24h,
        "time_to_peak": time_to_peak_hours,
        "time_to_peak_hours": time_to_peak_hours,
        "predicted_time_to_peak_hours": time_to_peak_hours,
    }

    if market_regime:
        feature_row.update(market_regime)

    # Keep aliases the live predictor / strategy may look for.
    feature_row["market_trend_strength_24h"] = _safe_float(
        feature_row.get("market_trend_strength_24h", trend_strength_local), trend_strength_local
    )
    feature_row["market_dispersion_24h"] = _safe_float(feature_row.get("market_dispersion_24h", 0.0), 0.0)
    feature_row["market_risk_off_score"] = _safe_float(feature_row.get("market_risk_off_score", 0.5), 0.5)
    feature_row["market_volume_regime_24h"] = _safe_float(feature_row.get("market_volume_regime_24h", volume_zscore_24h), volume_zscore_24h)
    feature_row["market_dispersion_1h"] = _safe_float(feature_row.get("market_dispersion_1h", abs(ret_1h)), abs(ret_1h))
    feature_row["market_breadth_up_1h"] = _safe_float(feature_row.get("market_breadth_up_1h", 0.5), 0.5)
    feature_row["market_breadth_up_24h"] = _safe_float(feature_row.get("market_breadth_up_24h", 0.5), 0.5)

    return feature_row


def build_live_feature_frame(
    symbols: Iterable[str],
    ctx: Any,
    lookback_bars: int = DEFAULT_HISTORY_BARS,
    interval_s: Optional[int] = None,
    candles_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Build a cross-sectional live feature frame for MLStrategy.

    This is the Phase 5.0 bridge from live candles -> replay-style candidate rows.
    It intentionally does not require replay prediction tables.
    """
    symbols = list(symbols)
    symbol_frames: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        if candles_by_symbol and symbol in candles_by_symbol:
            frame = candles_by_symbol[symbol]
        else:
            frame = load_symbol_candles(
                symbol=symbol,
                ctx=ctx,
                lookback_bars=lookback_bars,
                interval_s=interval_s,
            )
        if frame is not None and not frame.empty:
            symbol_frames[symbol] = frame

    market_regime = _compute_market_regime_features(
        current_symbol=symbols[0] if symbols else "",
        symbol_frames=symbol_frames,
    )

    rows = []
    for symbol in symbols:
        frame = symbol_frames.get(symbol)
        if frame is None or frame.empty:
            continue
        rows.append(compute_symbol_live_features(symbol, frame, market_regime=market_regime))

    if not rows:
        return pd.DataFrame(columns=["symbol", "prob_proxy", "rolling_volatility_24h"])

    out = pd.DataFrame(rows).sort_values(["prob_proxy", "symbol"], ascending=[False, True]).reset_index(drop=True)

    if "prob_proxy" in out.columns:
        out["prob_percentile_rank"] = out["prob_proxy"].rank(method="average", pct=True)
        out["prob_percentile_context"] = out["prob_percentile_rank"]

    if "rolling_volatility_24h" in out.columns:
        out["volatility_percentile_rank"] = out["rolling_volatility_24h"].rank(method="average", pct=True)

    return out


def compute_features(
    symbol: str,
    candles_df: pd.DataFrame,
    market_regime: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Backward-compatible single-symbol entry point."""
    return compute_symbol_live_features(symbol, candles_df, market_regime=market_regime)
