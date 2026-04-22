
"""Replay an ML-driven long-only spike strategy from saved model predictions.

Research-only replay script for Phase 4/5 Tradar development.

Phase 4.8 upgrades
------------------
- Supports multi-horizon prediction sources:
    - ensemble
    - 6h
    - 24h
    - 72h
    - direct (legacy/manual table mode)
- Uses ensemble_score for ranking when prediction_source=ensemble
- Preserves backward compatibility with single-model prediction tables
- Allows explicit score-column override for experiments
- Adds configurable regime gating using context-table regime features:
    - market_risk_off_score
    - market_dispersion_24h
    - market_trend_strength_24h
    - market_volume_regime_24h

Inputs
------
1. predictions table:
   direct/horizon tables:
       symbol, timestamp, pred_prob[, model_name]
   ensemble tables:
       symbol, timestamp, ensemble_score[, prob_ensemble, prob_6h, prob_24h, prob_72h]
2. price table: symbol, timestamp, price_close
3. optional context table: symbol, timestamp, rolling_volatility_24h,
   target_time_to_peak_seconds_24h, and optional regime columns
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from tradarbot.risk.risk_manager import RiskManager

LOGGER = logging.getLogger("replay_ml_strategy")


def _nested_get(mapping: dict, path: tuple[str, ...], default=None):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_replay_defaults(config_path: str | None) -> dict:
    """Load defaults exclusively from the top-level `ml_replay` config section."""
    if not config_path:
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    section = _nested_get(cfg, ("ml_replay",), {})
    return section if isinstance(section, dict) else {}


REGIME_COL_RISK_OFF = "market_risk_off_score"
REGIME_COL_DISPERSION_24H = "market_dispersion_24h"
REGIME_COL_TREND_STRENGTH_24H = "market_trend_strength_24h"
REGIME_COL_VOLUME_REGIME_24H = "market_volume_regime_24h"


@dataclass
class Position:
    symbol: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_prob: float
    quantity: float
    notional_usd: float
    rolling_volatility_24h: Optional[float]
    predicted_time_to_peak_hours: Optional[float]
    prob_percentile_rank: Optional[float]
    entry_score: Optional[float]
    prob_size_multiplier: float
    vol_size_multiplier: float
    kelly_fraction: float
    kelly_multiplier: float
    total_size_multiplier: float
    peak_price: float
    trailing_stop_price: Optional[float]
    partial_exit_taken: bool = False


@dataclass
class TradeRecord:
    symbol: str
    entry_timestamp: str
    exit_timestamp: str
    entry_price: float
    exit_price: float
    entry_prob: float
    entry_prob_percentile_rank: Optional[float]
    entry_rolling_volatility_24h: Optional[float]
    entry_predicted_time_to_peak_hours: Optional[float]
    entry_score: Optional[float]
    quantity: float
    notional_usd: float
    hold_hours: float
    gross_return_pct: float
    pnl_usd: float
    exit_reason: str
    model_name: Optional[str]
    prob_size_multiplier: float
    vol_size_multiplier: float
    kelly_fraction: float
    kelly_multiplier: float
    total_size_multiplier: float


@dataclass
class ReplayDiagnostics:
    timestamps_considered: int = 0
    candidate_rows_seen: int = 0
    candidate_rows_after_prob_threshold: int = 0
    candidate_rows_after_percentile: int = 0
    candidate_rows_after_volatility: int = 0
    candidate_rows_after_time_to_peak: int = 0
    candidate_rows_after_rank_score: int = 0
    candidate_rows_after_regime_gate: int = 0
    entries_submitted: int = 0
    entries_opened: int = 0
    skipped_already_open: int = 0
    skipped_missing_price: int = 0
    skipped_cash: int = 0
    skipped_position_cap: int = 0
    sized_below_min_notional: int = 0
    partial_take_profit_events: int = 0
    trailing_stop_exits: int = 0
    time_stop_exits: int = 0
    daily_loss_triggered: int = 0
    exposure_violations: int = 0
    trades_blocked_by_risk: int = 0
    regime_gate_blocks: int = 0
    regime_scale_events: int = 0
    regime_score_raise_events: int = 0
    forced_exits: int = 0


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/tradar.yaml")
    known, _ = pre.parse_known_args()
    defaults = _load_replay_defaults(known.config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--db-path", default=defaults.get("db_path", "tradarbot.db"))
    parser.add_argument("--predictions-table", default=defaults.get("predictions_table", "spike_model_predictions_hgb"))
    parser.add_argument("--price-table", default=defaults.get("price_table", "spike_base_rows"))
    parser.add_argument("--context-table", default=defaults.get("context_table", "spike_training_rows"))

    parser.add_argument(
        "--prediction-source",
        choices=["ensemble", "6h", "24h", "72h", "direct"],
        default=defaults.get("prediction_source", "direct"),
        help=(
            "Prediction source mode. "
            "'ensemble' expects ensemble_score/prob_ensemble style tables. "
            "'6h'/'24h'/'72h' are horizon-specific modes. "
            "'direct' preserves legacy behavior."
        ),
    )
    parser.add_argument("--score-column", default=defaults.get("score_column"))

    parser.add_argument("--symbol-column", default=defaults.get("symbol_column", "symbol"))
    parser.add_argument("--timestamp-column", default=defaults.get("timestamp_column", "timestamp"))
    parser.add_argument("--price-column", default=defaults.get("price_column", "price_close"))
    parser.add_argument("--prob-column", default=defaults.get("prob_column", "pred_prob"))
    parser.add_argument("--model-name-column", default=defaults.get("model_name_column", "model_name"))
    parser.add_argument("--rolling-volatility-column", default=defaults.get("rolling_volatility_column", "rolling_volatility_24h"))
    parser.add_argument("--time-to-peak-seconds-column", default=defaults.get("time_to_peak_seconds_column", "target_time_to_peak_seconds_24h"))

    parser.add_argument("--prob-threshold", type=float, default=defaults.get("prob_threshold", 0.18))
    parser.add_argument("--min-prob-percentile", type=float, default=defaults.get("min_prob_percentile", 0.0))
    parser.add_argument("--min-rolling-volatility-24h", type=float, default=defaults.get("min_rolling_volatility_24h"))
    parser.add_argument("--max-predicted-time-to-peak-hours", type=float, default=defaults.get("max_predicted_time_to_peak_hours"))

    parser.add_argument("--ranking-mode", choices=["probability", "composite"], default=defaults.get("ranking_mode", "composite"))
    parser.add_argument("--prob-zscore-weight", type=float, default=defaults.get("prob_zscore_weight", 1.00))
    parser.add_argument("--percentile-weight", type=float, default=defaults.get("percentile_weight", 0.20))
    parser.add_argument("--volatility-weight", type=float, default=defaults.get("volatility_weight", 0.35))
    parser.add_argument("--time-to-peak-weight", type=float, default=defaults.get("time_to_peak_weight", 0.25))
    parser.add_argument("--rank-score-min", type=float, default=defaults.get("rank_score_min"))

    parser.add_argument("--top-n", type=int, default=defaults.get("top_n", 3))
    parser.add_argument("--max-positions", type=int, default=defaults.get("max_positions", 3))
    parser.add_argument("--enable-dynamic-max-positions", action="store_true", default=defaults.get("enable_dynamic_max_positions", False))
    parser.add_argument("--min-dynamic-max-positions", type=int, default=defaults.get("min_dynamic_max_positions", 1))
    parser.add_argument("--dynamic-position-score-threshold", type=float, default=defaults.get("dynamic_position_score_threshold", 0.15))
    parser.add_argument("--notional-per-trade", type=float, default=defaults.get("notional_per_trade", 1000.0))
    parser.add_argument("--min-notional-per-trade", type=float, default=defaults.get("min_notional_per_trade", 0.0))
    parser.add_argument("--initial-cash", "--starting-equity-usd", dest="initial_cash", type=float, default=defaults.get("initial_cash", 100000.0))

    parser.add_argument("--enable-dynamic-sizing", action="store_true", default=defaults.get("enable_dynamic_sizing", False))
    parser.add_argument("--prob-size-cap", type=float, default=defaults.get("prob_size_cap", 2.0))
    parser.add_argument("--vol-reference", type=float, default=defaults.get("vol_reference", 0.006))
    parser.add_argument("--vol-size-floor", type=float, default=defaults.get("vol_size_floor", 0.75))
    parser.add_argument("--vol-size-cap", type=float, default=defaults.get("vol_size_cap", 1.25))
    parser.add_argument("--combined-size-cap", type=float, default=defaults.get("combined_size_cap", 2.0))

    parser.add_argument("--enable-kelly-sizing", action="store_true", default=defaults.get("enable_kelly_sizing", False))
    parser.add_argument("--kelly-fraction-scale", type=float, default=defaults.get("kelly_fraction_scale", 0.25))
    parser.add_argument("--kelly-probability-mode", choices=["raw", "threshold_relative"], default=defaults.get("kelly_probability_mode", "threshold_relative"))
    parser.add_argument("--kelly-size-cap", type=float, default=defaults.get("kelly_size_cap", 1.5))

    parser.add_argument("--take-profit-pct", type=float, default=defaults.get("take_profit_pct", 0.08))
    parser.add_argument("--stop-loss-pct", type=float, default=defaults.get("stop_loss_pct", 0.04))
    parser.add_argument("--max-hold-hours", type=float, default=defaults.get("max_hold_hours", 24.0))
    parser.add_argument("--trailing-stop-pct", type=float, default=defaults.get("trailing_stop_pct", 0.05))
    parser.add_argument("--trailing-stop-activation-pct", type=float, default=defaults.get("trailing_stop_activation_pct", 0.08))
    parser.add_argument("--partial-take-profit-pct", type=float, default=defaults.get("partial_take_profit_pct", 0.10))
    parser.add_argument("--partial-take-profit-fraction", type=float, default=defaults.get("partial_take_profit_fraction", 0.50))
    parser.add_argument("--time-stop-hours", type=float, default=defaults.get("time_stop_hours"))
    parser.add_argument("--time-stop-min-return-pct", type=float, default=defaults.get("time_stop_min_return_pct", 0.01))

    parser.add_argument("--enable-risk-manager", action="store_true", default=defaults.get("enable_risk_manager", False))
    parser.add_argument("--max-daily-loss-usd", type=float, default=defaults.get("max_daily_loss_usd"))
    parser.add_argument("--max-total-exposure-usd", type=float, default=defaults.get("max_total_exposure_usd"))
    parser.add_argument("--max-total-exposure-pct", type=float, default=defaults.get("max_total_exposure_pct"))
    parser.add_argument("--max-exposure-per-symbol-usd", type=float, default=defaults.get("max_exposure_per_symbol_usd"))
    parser.add_argument("--max-drawdown-pct", type=float, default=defaults.get("max_drawdown_pct"))
    parser.add_argument("--cooldown-minutes-per-symbol", type=float, default=defaults.get("cooldown_minutes_per_symbol", 0.0))

    parser.add_argument("--enable-drawdown-scaling", action="store_true", default=defaults.get("enable_drawdown_scaling", False))
    parser.add_argument("--drawdown-full-size-pct", type=float, default=defaults.get("drawdown_full_size_pct", 0.04))
    parser.add_argument("--drawdown-half-size-pct", type=float, default=defaults.get("drawdown_half_size_pct", 0.06))
    parser.add_argument("--drawdown-quarter-size-pct", type=float, default=defaults.get("drawdown_quarter_size_pct", 0.08))
    parser.add_argument("--drawdown-half-size-multiplier", type=float, default=defaults.get("drawdown_half_size_multiplier", 0.50))
    parser.add_argument("--drawdown-quarter-size-multiplier", type=float, default=defaults.get("drawdown_quarter_size_multiplier", 0.25))

    parser.add_argument("--enable-regime-gating", action="store_true", default=defaults.get("enable_regime_gating", False))
    parser.add_argument("--regime-gating-mode", choices=["block", "scale", "score_raise"], default=defaults.get("regime_gating_mode", "block"))
    parser.add_argument("--max-market-risk-off-score", type=float, default=defaults.get("max_market_risk_off_score"))
    parser.add_argument("--max-market-dispersion-24h", type=float, default=defaults.get("max_market_dispersion_24h"))
    parser.add_argument("--min-market-trend-strength-24h", type=float, default=defaults.get("min_market_trend_strength_24h"))
    parser.add_argument("--risk-off-size-multiplier", type=float, default=defaults.get("risk_off_size_multiplier", 0.50))
    parser.add_argument("--risk-off-score-raise", type=float, default=defaults.get("risk_off_score_raise", 0.0))

    parser.add_argument("--trades-table", default=defaults.get("trades_table", "ml_replay_trades"))
    parser.add_argument("--equity-table", default=defaults.get("equity_table", "ml_replay_equity"))
    parser.add_argument("--summary-table", default=defaults.get("summary_table", "ml_replay_summary"))
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default=defaults.get("if_exists", "replace"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=defaults.get("log_level", "INFO"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def resolve_prediction_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str]:
    """
    Returns:
        prob_col: column used for probability gating/sizing
        score_col: column used for ranking
    """
    source = args.prediction_source
    explicit_score = args.score_column

    if source == "ensemble":
        prob_candidates = ["prob_ensemble", "ensemble_score", args.prob_column]
        score_candidates = [explicit_score] if explicit_score else ["ensemble_score", "prob_ensemble", args.prob_column]
    elif source in {"6h", "24h", "72h"}:
        horizon_prob = f"prob_{source}"
        prob_candidates = [horizon_prob, args.prob_column]
        score_candidates = [explicit_score] if explicit_score else [horizon_prob, args.prob_column]
    else:
        prob_candidates = [args.prob_column]
        score_candidates = [explicit_score] if explicit_score else [args.prob_column]

    prob_col = next((c for c in prob_candidates if c and c in df.columns), None)
    score_col = next((c for c in score_candidates if c and c in df.columns), None)

    if prob_col is None:
        raise KeyError(
            f"Could not resolve probability column for prediction_source={source}. "
            f"Tried: {prob_candidates}"
        )
    if score_col is None:
        raise KeyError(
            f"Could not resolve score column for prediction_source={source}. "
            f"Tried: {score_candidates}"
        )
    return prob_col, score_col


def load_predictions(conn: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    query = f'SELECT * FROM "{args.predictions_table}"'
    df = pd.read_sql_query(query, conn)

    required = [args.symbol_column, args.timestamp_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Prediction table missing required columns: {missing}")

    prob_col, score_col = resolve_prediction_columns(df, args)

    df = df.copy()
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column], utc=True, errors="raise")
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[args.symbol_column, args.timestamp_column, prob_col, score_col])

    if prob_col != args.prob_column:
        df[args.prob_column] = df[prob_col]
    df["_resolved_prob_col"] = prob_col
    df["_resolved_score_col"] = score_col

    sort_cols = [args.timestamp_column, score_col, args.symbol_column]
    return df.sort_values(sort_cols, ascending=[True, False, True]).reset_index(drop=True)


def load_prices(conn: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    query = f'SELECT * FROM "{args.price_table}"'
    df = pd.read_sql_query(query, conn)

    required = [args.symbol_column, args.timestamp_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Price table missing required columns: {missing}")

    price_col = args.price_column
    if price_col not in df.columns:
        for fallback in ["price_close", "close", "close_price"]:
            if fallback in df.columns:
                price_col = fallback
                break
        else:
            raise KeyError(f"Price table missing configured price column '{args.price_column}'")

    df = df.copy()
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column], utc=True, errors="raise")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[args.symbol_column, args.timestamp_column, price_col])
    df = df[df[price_col] > 0].copy()
    if price_col != args.price_column:
        df[args.price_column] = df[price_col]
    return df.sort_values([args.timestamp_column, args.symbol_column]).reset_index(drop=True)


def load_context(conn: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    query = f'SELECT * FROM "{args.context_table}"'
    df = pd.read_sql_query(query, conn)

    required = [args.symbol_column, args.timestamp_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Context table missing required columns: {missing}")

    df = df.copy()
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column], utc=True, errors="raise")

    optional_numeric_columns = [
        args.rolling_volatility_column,
        args.time_to_peak_seconds_column,
        REGIME_COL_RISK_OFF,
        REGIME_COL_DISPERSION_24H,
        REGIME_COL_TREND_STRENGTH_24H,
        REGIME_COL_VOLUME_REGIME_24H,
    ]
    for col in optional_numeric_columns:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["predicted_time_to_peak_hours"] = df[args.time_to_peak_seconds_column] / 3600.0

    keep = [
        args.symbol_column,
        args.timestamp_column,
        args.rolling_volatility_column,
        "predicted_time_to_peak_hours",
        REGIME_COL_RISK_OFF,
        REGIME_COL_DISPERSION_24H,
        REGIME_COL_TREND_STRENGTH_24H,
        REGIME_COL_VOLUME_REGIME_24H,
    ]
    return (
        df[keep]
        .drop_duplicates([args.symbol_column, args.timestamp_column])
        .sort_values([args.timestamp_column, args.symbol_column])
        .reset_index(drop=True)
    )


def _safe_group_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((numeric - numeric.mean()) / std).fillna(0.0)


def enrich_predictions(predictions: pd.DataFrame, context: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ts_col = args.timestamp_column
    sym_col = args.symbol_column
    prob_col = "_resolved_prob_col"
    score_col = "_resolved_score_col"
    vol_col = args.rolling_volatility_column

    if prob_col not in predictions.columns or score_col not in predictions.columns:
        raise KeyError("Predictions frame is missing resolved probability/score metadata columns")

    actual_prob_col = str(predictions[prob_col].iloc[0])
    actual_score_col = str(predictions[score_col].iloc[0])

    merged = predictions.merge(
        context,
        on=[sym_col, ts_col],
        how="left",
        validate="many_to_one",
    )

    merged["prob_percentile_rank"] = merged.groupby(ts_col)[actual_prob_col].rank(method="min", pct=True, ascending=True)
    merged["prob_rank_desc"] = merged.groupby(ts_col)[actual_prob_col].rank(method="first", ascending=False)
    merged["prob_zscore"] = merged.groupby(ts_col)[actual_prob_col].transform(_safe_group_zscore)
    merged["score_zscore"] = merged.groupby(ts_col)[actual_score_col].transform(_safe_group_zscore)

    if vol_col in merged.columns:
        merged[vol_col] = pd.to_numeric(merged[vol_col], errors="coerce")
        merged["volatility_zscore"] = merged.groupby(ts_col)[vol_col].transform(_safe_group_zscore)
    else:
        merged["volatility_zscore"] = 0.0

    merged["predicted_time_to_peak_hours"] = pd.to_numeric(
        merged.get("predicted_time_to_peak_hours"),
        errors="coerce",
    )
    merged["time_to_peak_zscore"] = merged.groupby(ts_col)["predicted_time_to_peak_hours"].transform(_safe_group_zscore)

    for regime_col in [
        REGIME_COL_RISK_OFF,
        REGIME_COL_DISPERSION_24H,
        REGIME_COL_TREND_STRENGTH_24H,
        REGIME_COL_VOLUME_REGIME_24H,
    ]:
        if regime_col not in merged.columns:
            merged[regime_col] = pd.NA
        merged[regime_col] = pd.to_numeric(merged[regime_col], errors="coerce")

    if args.ranking_mode == "probability":
        merged["entry_score"] = merged[actual_score_col].astype(float)
    else:
        merged["entry_score"] = (
            float(args.prob_zscore_weight) * merged["score_zscore"].fillna(0.0)
            + float(args.percentile_weight) * (merged["prob_percentile_rank"].fillna(0.5) - 0.5)
            + float(args.volatility_weight) * merged["volatility_zscore"].fillna(0.0)
            - float(args.time_to_peak_weight) * merged["time_to_peak_zscore"].fillna(0.0)
        )

    merged["_resolved_prob_col_name"] = actual_prob_col
    merged["_resolved_score_col_name"] = actual_score_col

    return merged.sort_values(
        [ts_col, "entry_score", actual_score_col, sym_col],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def compute_max_drawdown(equity: pd.Series) -> float:
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak.replace(0, pd.NA)
    return abs(float(drawdown.min())) if not drawdown.empty else 0.0


def compute_equity_state(
    cash: float,
    open_positions: Dict[str, Position],
    current_prices: Dict[str, float],
) -> tuple[float, float, float]:
    unrealized = 0.0
    open_notional = 0.0
    for symbol, pos in open_positions.items():
        px = current_prices.get(symbol, pos.entry_price)
        unrealized += pos.quantity * (px - pos.entry_price)
        open_notional += pos.quantity * px
    equity = float(cash + open_notional)
    return float(unrealized), float(open_notional), float(equity)


def compute_kelly_terms(pred_prob: float, args: argparse.Namespace) -> Tuple[float, float]:
    if not args.enable_kelly_sizing:
        return 0.0, 1.0

    reward = max(float(args.take_profit_pct), 1e-9)
    risk = max(float(args.stop_loss_pct), 1e-9)
    b = reward / risk

    if args.kelly_probability_mode == "raw":
        p = max(0.01, min(0.99, float(pred_prob)))
    else:
        denom = max(float(args.prob_threshold) * 3.0, 1e-9)
        p = max(0.01, min(0.99, float(pred_prob) / denom))

    q = 1.0 - p
    kelly_fraction = max(0.0, (b * p - q) / max(b, 1e-9))
    kelly_multiplier = 1.0 + (float(args.kelly_fraction_scale) * kelly_fraction * b)
    kelly_multiplier = min(float(args.kelly_size_cap), max(0.0, kelly_multiplier))
    return float(kelly_fraction), float(kelly_multiplier)


def compute_size_multipliers(row: pd.Series, args: argparse.Namespace) -> tuple[float, float, float, float, float]:
    if not args.enable_dynamic_sizing:
        kf, km = compute_kelly_terms(float(row[args.prob_column]), args)
        return 1.0, 1.0, kf, km, km

    prob = float(row[args.prob_column])
    prob_threshold = max(float(args.prob_threshold), 1e-9)

    prob_ratio = max(prob / prob_threshold, 0.0)
    prob_multiplier = min(float(args.prob_size_cap), prob_ratio ** 1.3)

    vol_multiplier = 1.0
    vol_value = row.get(args.rolling_volatility_column)
    if pd.notna(vol_value):
        vol_reference = max(float(args.vol_reference), 1e-9)
        vol_multiplier = float(vol_value) / vol_reference
        vol_multiplier = max(float(args.vol_size_floor), min(float(args.vol_size_cap), vol_multiplier))

    kelly_fraction, kelly_multiplier = compute_kelly_terms(prob, args)

    total_multiplier = prob_multiplier * vol_multiplier * kelly_multiplier
    total_multiplier = max(0.0, min(float(args.combined_size_cap), total_multiplier))
    return (
        float(prob_multiplier),
        float(vol_multiplier),
        float(kelly_fraction),
        float(kelly_multiplier),
        float(total_multiplier),
    )


def _lookup_model_name(
    predictions_by_ts: Dict[pd.Timestamp, pd.DataFrame],
    ts: pd.Timestamp,
    symbol: str,
    model_col: str,
    sym_col: str,
) -> Optional[str]:
    pred_rows = predictions_by_ts.get(ts)
    if pred_rows is None or model_col not in pred_rows.columns:
        return None
    hit = pred_rows[pred_rows[sym_col] == symbol]
    if hit.empty:
        return None
    value = hit.iloc[0][model_col]
    return None if pd.isna(value) else str(value)


def finalize_trade(
    position: Position,
    exit_timestamp: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    model_name: Optional[str],
    quantity: Optional[float] = None,
    notional_usd: Optional[float] = None,
) -> TradeRecord:
    actual_quantity = float(position.quantity if quantity is None else quantity)
    actual_notional = float(position.notional_usd if notional_usd is None else notional_usd)
    gross_return = (exit_price - position.entry_price) / position.entry_price
    pnl_usd = actual_quantity * (exit_price - position.entry_price)
    hold_hours = (exit_timestamp - position.entry_timestamp).total_seconds() / 3600.0
    return TradeRecord(
        symbol=position.symbol,
        entry_timestamp=position.entry_timestamp.isoformat(),
        exit_timestamp=exit_timestamp.isoformat(),
        entry_price=float(position.entry_price),
        exit_price=float(exit_price),
        entry_prob=float(position.entry_prob),
        entry_prob_percentile_rank=(
            float(position.prob_percentile_rank)
            if position.prob_percentile_rank is not None and not math.isnan(position.prob_percentile_rank)
            else None
        ),
        entry_rolling_volatility_24h=(
            float(position.rolling_volatility_24h)
            if position.rolling_volatility_24h is not None and not math.isnan(position.rolling_volatility_24h)
            else None
        ),
        entry_predicted_time_to_peak_hours=(
            float(position.predicted_time_to_peak_hours)
            if position.predicted_time_to_peak_hours is not None and not math.isnan(position.predicted_time_to_peak_hours)
            else None
        ),
        entry_score=(
            float(position.entry_score)
            if position.entry_score is not None and not math.isnan(position.entry_score)
            else None
        ),
        quantity=actual_quantity,
        notional_usd=actual_notional,
        hold_hours=float(hold_hours),
        gross_return_pct=float(gross_return),
        pnl_usd=float(pnl_usd),
        exit_reason=exit_reason,
        model_name=model_name,
        prob_size_multiplier=float(position.prob_size_multiplier),
        vol_size_multiplier=float(position.vol_size_multiplier),
        kelly_fraction=float(position.kelly_fraction),
        kelly_multiplier=float(position.kelly_multiplier),
        total_size_multiplier=float(position.total_size_multiplier),
    )


def _compute_dynamic_position_limit(candidates: pd.DataFrame, args: argparse.Namespace) -> int:
    hard_cap = max(int(args.max_positions), 0)
    if hard_cap <= 0:
        return 0
    if not args.enable_dynamic_max_positions:
        return hard_cap
    if candidates.empty:
        return 0
    min_cap = max(1, min(int(args.min_dynamic_max_positions), hard_cap))
    strong_count = int((candidates["entry_score"] >= float(args.dynamic_position_score_threshold)).sum())
    if strong_count <= 0:
        strong_count = 1
    return min(hard_cap, max(min_cap, strong_count))


def _regime_condition_triggered(value: object, cmp_name: str, threshold: Optional[float]) -> bool:
    if threshold is None or pd.isna(value):
        return False
    numeric_value = float(value)
    numeric_threshold = float(threshold)
    if cmp_name == "gt":
        return numeric_value > numeric_threshold
    if cmp_name == "lt":
        return numeric_value < numeric_threshold
    raise ValueError(f"Unsupported regime comparison: {cmp_name}")


def _row_in_adverse_regime(row: pd.Series, args: argparse.Namespace) -> bool:
    if not args.enable_regime_gating:
        return False

    triggered = False
    triggered = triggered or _regime_condition_triggered(row.get(REGIME_COL_RISK_OFF), "gt", args.max_market_risk_off_score)
    triggered = triggered or _regime_condition_triggered(row.get(REGIME_COL_DISPERSION_24H), "gt", args.max_market_dispersion_24h)
    triggered = triggered or _regime_condition_triggered(row.get(REGIME_COL_TREND_STRENGTH_24H), "lt", args.min_market_trend_strength_24h)
    return triggered


def apply_regime_gating(
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    diagnostics: ReplayDiagnostics,
) -> pd.DataFrame:
    if candidates.empty:
        diagnostics.candidate_rows_after_regime_gate += 0
        return candidates
    if not args.enable_regime_gating:
        diagnostics.candidate_rows_after_regime_gate += int(len(candidates))
        return candidates

    adverse_mask = candidates.apply(lambda row: _row_in_adverse_regime(row, args), axis=1)

    if args.regime_gating_mode == "block":
        blocked = int(adverse_mask.sum())
        diagnostics.regime_gate_blocks += blocked
        gated = candidates.loc[~adverse_mask].copy()
        diagnostics.candidate_rows_after_regime_gate += int(len(gated))
        return gated

    if args.regime_gating_mode == "score_raise":
        # Require stronger candidate quality only when the current row is in an adverse regime.
        # If rank_score_min is unset, treat the raise itself as the temporary minimum score floor.
        base_floor = float(args.rank_score_min) if args.rank_score_min is not None else 0.0
        raised_floor = base_floor + float(args.risk_off_score_raise)
        if adverse_mask.any():
            diagnostics.regime_score_raise_events += int(adverse_mask.sum())
            entry_scores = pd.to_numeric(candidates["entry_score"], errors="coerce")
            adverse_keep_mask = entry_scores >= raised_floor
            keep_mask = (~adverse_mask) | adverse_keep_mask
            gated = candidates.loc[keep_mask].copy()
            gated["_regime_effective_score_floor"] = base_floor
            gated.loc[gated.index.intersection(candidates.index[adverse_mask]), "_regime_effective_score_floor"] = raised_floor
            diagnostics.regime_gate_blocks += int((~keep_mask).sum())
        else:
            gated = candidates.copy()
            gated["_regime_effective_score_floor"] = base_floor
        diagnostics.candidate_rows_after_regime_gate += int(len(gated))
        return gated

    if args.regime_gating_mode == "scale":
        scaled = candidates.copy()
        if adverse_mask.any():
            diagnostics.regime_scale_events += int(adverse_mask.sum())
            scaled.loc[adverse_mask, "_regime_size_multiplier"] = float(args.risk_off_size_multiplier)
        diagnostics.candidate_rows_after_regime_gate += int(len(scaled))
        return scaled

    raise ValueError(f"Unsupported regime_gating_mode: {args.regime_gating_mode}")



def build_runtime_args(overrides: Optional[dict] = None, *, config_path: Optional[str] = None, section: str = "ml_replay") -> argparse.Namespace:
    """Build an argparse-style namespace for replay or live imports.

    Phase 5.0 uses this so live strategy code can import replay math without
    depending on CLI parsing. By default it reads the same top-level `ml_replay`
    section the script already uses, but callers may point it at `ml_live`.
    """
    if config_path is None:
        config_path = "config/tradar.yaml"

    defaults = {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        section_data = _nested_get(cfg, (section,), {})
        if isinstance(section_data, dict):
            defaults.update(section_data)
    except FileNotFoundError:
        pass

    if overrides:
        defaults.update(dict(overrides))

    return argparse.Namespace(
        db_path=defaults.get("db_path", "tradarbot.db"),
        predictions_table=defaults.get("predictions_table", "spike_model_predictions_hgb"),
        price_table=defaults.get("price_table", "spike_base_rows"),
        context_table=defaults.get("context_table", "spike_training_rows"),
        prediction_source=defaults.get("prediction_source", "direct"),
        score_column=defaults.get("score_column"),
        symbol_column=defaults.get("symbol_column", "symbol"),
        timestamp_column=defaults.get("timestamp_column", "timestamp"),
        price_column=defaults.get("price_column", "price_close"),
        prob_column=defaults.get("prob_column", "pred_prob"),
        model_name_column=defaults.get("model_name_column", "model_name"),
        rolling_volatility_column=defaults.get("rolling_volatility_column", "rolling_volatility_24h"),
        time_to_peak_seconds_column=defaults.get("time_to_peak_seconds_column", "target_time_to_peak_seconds_24h"),
        prob_threshold=defaults.get("prob_threshold", 0.18),
        min_prob_percentile=defaults.get("min_prob_percentile", 0.0),
        min_rolling_volatility_24h=defaults.get("min_rolling_volatility_24h"),
        max_predicted_time_to_peak_hours=defaults.get("max_predicted_time_to_peak_hours"),
        ranking_mode=defaults.get("ranking_mode", "composite"),
        prob_zscore_weight=defaults.get("prob_zscore_weight", 1.00),
        percentile_weight=defaults.get("percentile_weight", 0.20),
        volatility_weight=defaults.get("volatility_weight", 0.35),
        time_to_peak_weight=defaults.get("time_to_peak_weight", 0.25),
        rank_score_min=defaults.get("rank_score_min"),
        top_n=defaults.get("top_n", 3),
        max_positions=defaults.get("max_positions", 3),
        enable_dynamic_max_positions=defaults.get("enable_dynamic_max_positions", False),
        min_dynamic_max_positions=defaults.get("min_dynamic_max_positions", 1),
        dynamic_position_score_threshold=defaults.get("dynamic_position_score_threshold", 0.15),
        notional_per_trade=defaults.get("notional_per_trade", 1000.0),
        min_notional_per_trade=defaults.get("min_notional_per_trade", 0.0),
        initial_cash=defaults.get("initial_cash", 100000.0),
        enable_dynamic_sizing=defaults.get("enable_dynamic_sizing", False),
        prob_size_cap=defaults.get("prob_size_cap", 2.0),
        vol_reference=defaults.get("vol_reference", 0.006),
        vol_size_floor=defaults.get("vol_size_floor", 0.75),
        vol_size_cap=defaults.get("vol_size_cap", 1.25),
        combined_size_cap=defaults.get("combined_size_cap", 2.0),
        enable_kelly_sizing=defaults.get("enable_kelly_sizing", False),
        kelly_fraction_scale=defaults.get("kelly_fraction_scale", 0.25),
        kelly_probability_mode=defaults.get("kelly_probability_mode", "threshold_relative"),
        kelly_size_cap=defaults.get("kelly_size_cap", 1.5),
        take_profit_pct=defaults.get("take_profit_pct", 0.08),
        stop_loss_pct=defaults.get("stop_loss_pct", 0.04),
        max_hold_hours=defaults.get("max_hold_hours", 24.0),
        trailing_stop_pct=defaults.get("trailing_stop_pct", 0.05),
        trailing_stop_activation_pct=defaults.get("trailing_stop_activation_pct", 0.08),
        partial_take_profit_pct=defaults.get("partial_take_profit_pct", 0.10),
        partial_take_profit_fraction=defaults.get("partial_take_profit_fraction", 0.50),
        time_stop_hours=defaults.get("time_stop_hours"),
        time_stop_min_return_pct=defaults.get("time_stop_min_return_pct", 0.01),
        enable_risk_manager=defaults.get("enable_risk_manager", False),
        max_daily_loss_usd=defaults.get("max_daily_loss_usd"),
        max_total_exposure_usd=defaults.get("max_total_exposure_usd"),
        max_total_exposure_pct=defaults.get("max_total_exposure_pct"),
        max_exposure_per_symbol_usd=defaults.get("max_exposure_per_symbol_usd"),
        max_drawdown_pct=defaults.get("max_drawdown_pct"),
        cooldown_minutes_per_symbol=defaults.get("cooldown_minutes_per_symbol", 0.0),
        enable_drawdown_scaling=defaults.get("enable_drawdown_scaling", False),
        drawdown_full_size_pct=defaults.get("drawdown_full_size_pct", 0.04),
        drawdown_half_size_pct=defaults.get("drawdown_half_size_pct", 0.06),
        drawdown_quarter_size_pct=defaults.get("drawdown_quarter_size_pct", 0.08),
        drawdown_half_size_multiplier=defaults.get("drawdown_half_size_multiplier", 0.50),
        drawdown_quarter_size_multiplier=defaults.get("drawdown_quarter_size_multiplier", 0.25),
        enable_regime_gating=defaults.get("enable_regime_gating", False),
        regime_gating_mode=defaults.get("regime_gating_mode", "block"),
        max_market_risk_off_score=defaults.get("max_market_risk_off_score"),
        max_market_dispersion_24h=defaults.get("max_market_dispersion_24h"),
        min_market_trend_strength_24h=defaults.get("min_market_trend_strength_24h"),
        risk_off_size_multiplier=defaults.get("risk_off_size_multiplier", 0.50),
        risk_off_score_raise=defaults.get("risk_off_score_raise", 0.0),
        trades_table=defaults.get("trades_table", "ml_replay_trades"),
        equity_table=defaults.get("equity_table", "ml_replay_equity"),
        summary_table=defaults.get("summary_table", "ml_replay_summary"),
        if_exists=defaults.get("if_exists", "replace"),
        log_level=defaults.get("log_level", "INFO"),
    )


def instantiate_risk_manager(args: argparse.Namespace) -> RiskManager:
    """Shared risk-manager constructor for replay and live strategy imports."""
    return RiskManager(
        {
            "enabled": bool(args.enable_risk_manager),
            "max_daily_loss_usd": args.max_daily_loss_usd,
            "max_total_exposure_usd": args.max_total_exposure_usd,
            "max_total_exposure_pct": args.max_total_exposure_pct,
            "max_exposure_per_symbol_usd": args.max_exposure_per_symbol_usd,
            "max_drawdown_pct": args.max_drawdown_pct,
            "cooldown_minutes_per_symbol": args.cooldown_minutes_per_symbol,
            "enable_drawdown_scaling": bool(args.enable_drawdown_scaling),
            "drawdown_full_size_pct": args.drawdown_full_size_pct,
            "drawdown_half_size_pct": args.drawdown_half_size_pct,
            "drawdown_quarter_size_pct": args.drawdown_quarter_size_pct,
            "drawdown_half_size_multiplier": args.drawdown_half_size_multiplier,
            "drawdown_quarter_size_multiplier": args.drawdown_quarter_size_multiplier,
            "close_positions_on_kill_switch": False,
            "close_positions_on_daily_loss": False,
            "close_positions_on_drawdown": False,
        }
    )


def filter_ranked_candidates(
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    diagnostics: Optional[ReplayDiagnostics] = None,
) -> pd.DataFrame:
    """Shared candidate filtering/ranking path for replay and live strategy use.

    Expected input:
    - already enriched with `enrich_predictions(...)` or an equivalent live frame
    - includes args.prob_column, entry_score, percentile/risk/regime columns
    """
    if candidates is None or candidates.empty:
        return candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()

    out = candidates.copy()
    prob_col = args.prob_column
    vol_col = args.rolling_volatility_column
    resolved_score_col = (
        out["_resolved_score_col_name"].iloc[0]
        if "_resolved_score_col_name" in out.columns and not out.empty
        else prob_col
    )
    sym_col = args.symbol_column

    if diagnostics is not None:
        diagnostics.candidate_rows_seen += int(len(out))

    out = out[out[prob_col] >= float(args.prob_threshold)].copy()
    if diagnostics is not None:
        diagnostics.candidate_rows_after_prob_threshold += int(len(out))

    if not out.empty and float(args.min_prob_percentile) > 0.0:
        out = out[
            out["prob_percentile_rank"].notna()
            & (out["prob_percentile_rank"] >= float(args.min_prob_percentile))
        ].copy()
    if diagnostics is not None:
        diagnostics.candidate_rows_after_percentile += int(len(out))

    if not out.empty and args.min_rolling_volatility_24h is not None:
        out = out[
            out[vol_col].notna() & (out[vol_col] >= float(args.min_rolling_volatility_24h))
        ].copy()
    if diagnostics is not None:
        diagnostics.candidate_rows_after_volatility += int(len(out))

    if not out.empty and args.max_predicted_time_to_peak_hours is not None:
        out = out[
            out["predicted_time_to_peak_hours"].notna()
            & (out["predicted_time_to_peak_hours"] <= float(args.max_predicted_time_to_peak_hours))
        ].copy()
    if diagnostics is not None:
        diagnostics.candidate_rows_after_time_to_peak += int(len(out))

    if not out.empty and args.rank_score_min is not None:
        out = out[out["entry_score"] >= float(args.rank_score_min)].copy()
    if diagnostics is not None:
        diagnostics.candidate_rows_after_rank_score += int(len(out))

    out = apply_regime_gating(out, args, diagnostics)

    if not out.empty:
        out = out.sort_values(
            ["entry_score", resolved_score_col, sym_col],
            ascending=[False, False, True],
        ).head(max(int(args.top_n), 0)).reset_index(drop=True)
    return out


def compute_entry_decision(
    row: pd.Series,
    *,
    args: argparse.Namespace,
    symbol: str,
    entry_price: float,
    cash: float,
    risk_manager: Optional[RiskManager] = None,
) -> Optional[dict]:
    """Shared entry-sizing decision for replay and live strategy imports."""
    prob_mult, vol_mult, kelly_fraction, kelly_mult, total_mult = compute_size_multipliers(row, args)

    regime_size_multiplier = 1.0
    if args.enable_regime_gating and args.regime_gating_mode == "scale":
        regime_size_multiplier = float(row.get("_regime_size_multiplier", 1.0) or 1.0)

    target_notional = float(args.notional_per_trade) * total_mult * regime_size_multiplier

    drawdown_size_multiplier = 1.0
    if risk_manager is not None:
        drawdown_size_multiplier = float(risk_manager.get_position_size_multiplier())
        if drawdown_size_multiplier <= 0.0:
            return None

    target_notional *= drawdown_size_multiplier
    if target_notional < float(args.min_notional_per_trade):
        return None
    if cash < target_notional:
        return None
    if entry_price <= 0.0:
        return None

    if risk_manager is not None:
        allowed, _ = risk_manager.can_enter_trade(symbol, target_notional)
        if not allowed:
            return None

    quantity = float(target_notional / entry_price)
    return {
        "symbol": symbol,
        "entry_price": float(entry_price),
        "quantity": quantity,
        "target_notional": float(target_notional),
        "prob_size_multiplier": float(prob_mult),
        "vol_size_multiplier": float(vol_mult),
        "kelly_fraction": float(kelly_fraction),
        "kelly_multiplier": float(kelly_mult),
        "regime_size_multiplier": float(regime_size_multiplier),
        "drawdown_size_multiplier": float(drawdown_size_multiplier),
        "total_size_multiplier": float(total_mult),
    }



def replay_strategy(predictions: pd.DataFrame, prices: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts_col = args.timestamp_column
    sym_col = args.symbol_column
    price_col = args.price_column
    prob_col = args.prob_column
    vol_col = args.rolling_volatility_column
    model_col = args.model_name_column

    resolved_score_col = (
        predictions["_resolved_score_col_name"].iloc[0]
        if "_resolved_score_col_name" in predictions.columns and not predictions.empty
        else prob_col
    )

    timestamps = sorted(set(predictions[ts_col]).intersection(set(prices[ts_col])))
    if not timestamps:
        raise ValueError("No overlapping timestamps between predictions and prices")

    cash = float(args.initial_cash)
    open_positions: Dict[str, Position] = {}
    trades: List[TradeRecord] = []
    equity_rows: List[dict] = []
    diagnostics = ReplayDiagnostics(timestamps_considered=len(timestamps))
    prob_percentile_cutoff_value = None
    if float(args.min_prob_percentile) > 0.0:
        prob_percentile_cutoff_value = float(predictions[prob_col].quantile(float(args.min_prob_percentile)))

    risk_manager = RiskManager(
        {
            "enabled": bool(args.enable_risk_manager),
            "max_daily_loss_usd": args.max_daily_loss_usd,
            "max_total_exposure_usd": args.max_total_exposure_usd,
            "max_total_exposure_pct": args.max_total_exposure_pct,
            "max_exposure_per_symbol_usd": args.max_exposure_per_symbol_usd,
            "max_drawdown_pct": args.max_drawdown_pct,
            "cooldown_minutes_per_symbol": args.cooldown_minutes_per_symbol,
            "enable_drawdown_scaling": bool(args.enable_drawdown_scaling),
            "drawdown_full_size_pct": args.drawdown_full_size_pct,
            "drawdown_half_size_pct": args.drawdown_half_size_pct,
            "drawdown_quarter_size_pct": args.drawdown_quarter_size_pct,
            "drawdown_half_size_multiplier": args.drawdown_half_size_multiplier,
            "drawdown_quarter_size_multiplier": args.drawdown_quarter_size_multiplier,
            "close_positions_on_kill_switch": False,
            "close_positions_on_daily_loss": False,
            "close_positions_on_drawdown": False,
        }
    )

    predictions_by_ts = {
        ts: grp.sort_values(["entry_score", resolved_score_col, sym_col], ascending=[False, False, True]).reset_index(drop=True)
        for ts, grp in predictions.groupby(ts_col)
    }
    prices_by_ts = {ts: grp.reset_index(drop=True) for ts, grp in prices.groupby(ts_col)}

    for ts in timestamps:
        price_group = prices_by_ts.get(ts)
        if price_group is None or price_group.empty:
            continue

        current_prices = {row[sym_col]: float(row[price_col]) for _, row in price_group.iterrows()}

        risk_manager.set_timestamp(ts)
        risk_manager.update_positions(open_positions, current_prices)
        _, _, equity_before = compute_equity_state(cash=cash, open_positions=open_positions, current_prices=current_prices)
        risk_manager.update_equity(equity_before)

        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            if symbol not in current_prices:
                continue

            px = current_prices[symbol]
            pos.peak_price = max(float(pos.peak_price), float(px))
            ret = (px - pos.entry_price) / pos.entry_price
            hold_hours = (ts - pos.entry_timestamp).total_seconds() / 3600.0

            if (
                not pos.partial_exit_taken
                and float(args.partial_take_profit_fraction) > 0.0
                and ret >= float(args.partial_take_profit_pct)
                and pos.quantity > 0.0
            ):
                partial_fraction = min(max(float(args.partial_take_profit_fraction), 0.0), 1.0)
                partial_quantity = pos.quantity * partial_fraction
                partial_notional = pos.notional_usd * partial_fraction
                model_name = _lookup_model_name(predictions_by_ts, pos.entry_timestamp, symbol, model_col, sym_col)
                trade = finalize_trade(
                    pos,
                    ts,
                    px,
                    "partial_take_profit",
                    model_name,
                    quantity=partial_quantity,
                    notional_usd=partial_notional,
                )
                trades.append(trade)
                cash += partial_notional + trade.pnl_usd
                risk_manager.update_realized_pnl(trade.pnl_usd, ts)
                pos.quantity -= partial_quantity
                pos.notional_usd -= partial_notional
                pos.partial_exit_taken = True
                diagnostics.partial_take_profit_events += 1
                if pos.quantity <= 1e-12 or pos.notional_usd <= 1e-9:
                    risk_manager.register_exit(symbol, ts, forced=False)
                    del open_positions[symbol]
                    continue

            if ret >= float(args.trailing_stop_activation_pct):
                candidate_trailing_stop = pos.peak_price * (1.0 - float(args.trailing_stop_pct))
                pos.trailing_stop_price = max(
                    candidate_trailing_stop,
                    pos.trailing_stop_price if pos.trailing_stop_price is not None else 0.0,
                )

            exit_reason: Optional[str] = None
            if ret >= float(args.take_profit_pct):
                exit_reason = "take_profit"
            elif ret <= -float(args.stop_loss_pct):
                exit_reason = "stop_loss"
            elif pos.trailing_stop_price is not None and px <= pos.trailing_stop_price:
                exit_reason = "trailing_stop"
                diagnostics.trailing_stop_exits += 1
            elif (
                args.time_stop_hours is not None
                and hold_hours >= float(args.time_stop_hours)
                and ret < float(args.time_stop_min_return_pct)
            ):
                exit_reason = "time_stop_underperform"
                diagnostics.time_stop_exits += 1
            elif hold_hours >= float(args.max_hold_hours):
                exit_reason = "max_hold"

            if exit_reason is not None:
                model_name = _lookup_model_name(predictions_by_ts, pos.entry_timestamp, symbol, model_col, sym_col)
                trade = finalize_trade(pos, ts, px, exit_reason, model_name)
                cash += trade.pnl_usd + pos.notional_usd
                trades.append(trade)
                risk_manager.update_realized_pnl(trade.pnl_usd, ts)
                risk_manager.register_exit(symbol, ts, forced=False)
                del open_positions[symbol]

        risk_manager.update_positions(open_positions, current_prices)
        _, _, equity_mid = compute_equity_state(cash=cash, open_positions=open_positions, current_prices=current_prices)
        risk_manager.update_equity(equity_mid)

        if risk_manager.should_force_exit() and open_positions:
            for symbol in list(open_positions.keys()):
                pos = open_positions[symbol]
                px = current_prices.get(symbol, pos.entry_price)
                model_name = _lookup_model_name(predictions_by_ts, pos.entry_timestamp, symbol, model_col, sym_col)
                trade = finalize_trade(pos, ts, px, "risk_forced_exit", model_name)
                cash += trade.pnl_usd + pos.notional_usd
                trades.append(trade)
                risk_manager.update_realized_pnl(trade.pnl_usd, ts)
                risk_manager.register_exit(symbol, ts, forced=True)
                del open_positions[symbol]

        candidates = predictions_by_ts.get(ts)
        if candidates is not None and not candidates.empty:
            diagnostics.candidate_rows_seen += int(len(candidates))

            candidates = candidates[candidates[prob_col] >= float(args.prob_threshold)].copy()
            diagnostics.candidate_rows_after_prob_threshold += int(len(candidates))

            if not candidates.empty and prob_percentile_cutoff_value is not None:
                candidates = candidates[candidates[prob_col] >= prob_percentile_cutoff_value].copy()
            diagnostics.candidate_rows_after_percentile += int(len(candidates))

            if not candidates.empty and args.min_rolling_volatility_24h is not None:
                candidates = candidates[
                    candidates[vol_col].notna() & (candidates[vol_col] >= float(args.min_rolling_volatility_24h))
                ].copy()
            diagnostics.candidate_rows_after_volatility += int(len(candidates))

            if not candidates.empty and args.max_predicted_time_to_peak_hours is not None:
                candidates = candidates[
                    candidates["predicted_time_to_peak_hours"].notna()
                    & (candidates["predicted_time_to_peak_hours"] <= float(args.max_predicted_time_to_peak_hours))
                ].copy()
            diagnostics.candidate_rows_after_time_to_peak += int(len(candidates))

            if not candidates.empty and args.rank_score_min is not None:
                candidates = candidates[candidates["entry_score"] >= float(args.rank_score_min)].copy()
            diagnostics.candidate_rows_after_rank_score += int(len(candidates))

            candidates = apply_regime_gating(candidates, args, diagnostics)

            if not candidates.empty:
                candidates = candidates.sort_values(
                    ["entry_score", resolved_score_col, sym_col],
                    ascending=[False, False, True],
                ).head(max(int(args.top_n), 0))
            diagnostics.entries_submitted += int(len(candidates))

            allowed_position_total = _compute_dynamic_position_limit(candidates, args)
            available_slots = max(0, allowed_position_total - len(open_positions))
            if available_slots <= 0 and not candidates.empty:
                diagnostics.skipped_position_cap += int(len(candidates))
            else:
                submitted_count = 0
                for _, row in candidates.iterrows():
                    if submitted_count >= available_slots:
                        diagnostics.skipped_position_cap += 1
                        continue

                    symbol = row[sym_col]
                    if symbol in open_positions:
                        diagnostics.skipped_already_open += 1
                        continue
                    if symbol not in current_prices:
                        diagnostics.skipped_missing_price += 1
                        continue

                    entry_price = current_prices[symbol]
                    decision = compute_entry_decision(
                        row,
                        args=args,
                        symbol=symbol,
                        entry_price=entry_price,
                        cash=cash,
                        risk_manager=risk_manager,
                    )
                    if decision is None:
                        if cash < float(args.notional_per_trade):
                            diagnostics.skipped_cash += 1
                        continue

                    prob_mult = decision["prob_size_multiplier"]
                    vol_mult = decision["vol_size_multiplier"]
                    kelly_fraction = decision["kelly_fraction"]
                    kelly_mult = decision["kelly_multiplier"]
                    total_mult = decision["total_size_multiplier"]
                    regime_size_multiplier = decision["regime_size_multiplier"]
                    target_notional = decision["target_notional"]
                    quantity = decision["quantity"]
                    open_positions[symbol] = Position(
                        symbol=symbol,
                        entry_timestamp=ts,
                        entry_price=entry_price,
                        entry_prob=float(row[prob_col]),
                        quantity=float(quantity),
                        notional_usd=float(target_notional),
                        rolling_volatility_24h=(float(row[vol_col]) if pd.notna(row.get(vol_col)) else None),
                        predicted_time_to_peak_hours=(
                            float(row["predicted_time_to_peak_hours"])
                            if pd.notna(row.get("predicted_time_to_peak_hours")) else None
                        ),
                        prob_percentile_rank=(
                            float(row["prob_percentile_rank"]) if pd.notna(row.get("prob_percentile_rank")) else None
                        ),
                        entry_score=(
                            float(row["entry_score"]) if pd.notna(row.get("entry_score")) else None
                        ),
                        prob_size_multiplier=prob_mult,
                        vol_size_multiplier=vol_mult,
                        kelly_fraction=kelly_fraction,
                        kelly_multiplier=kelly_mult,
                        total_size_multiplier=total_mult * regime_size_multiplier,
                        peak_price=float(entry_price),
                        trailing_stop_price=None,
                    )
                    diagnostics.entries_opened += 1
                    cash -= target_notional
                    submitted_count += 1

                    risk_manager.update_positions(open_positions, current_prices)
                    _, _, equity_after_entry = compute_equity_state(
                        cash=cash,
                        open_positions=open_positions,
                        current_prices=current_prices,
                    )
                    risk_manager.update_equity(equity_after_entry)

        risk_manager.update_positions(open_positions, current_prices)
        unrealized, open_notional, equity = compute_equity_state(
            cash=cash,
            open_positions=open_positions,
            current_prices=current_prices,
        )
        risk_manager.update_equity(equity)

        market_regime_row = predictions_by_ts.get(ts)
        regime_snapshot = market_regime_row.iloc[0] if market_regime_row is not None and not market_regime_row.empty else None

        equity_rows.append(
            {
                "timestamp": ts.isoformat(),
                "cash_usd": float(cash),
                "open_positions": int(len(open_positions)),
                "open_notional_usd": float(open_notional),
                "unrealized_pnl_usd": float(unrealized),
                "equity_usd": float(equity),
                "risk_safe_mode": int(risk_manager.safe_mode),
                "risk_kill_switch": int(risk_manager.kill_switch_triggered),
                "risk_total_exposure_usd": float(risk_manager.total_exposure),
                "risk_daily_pnl_usd": float(risk_manager.current_daily_pnl()),
                "risk_drawdown_pct": float(risk_manager.current_drawdown_pct()),
                REGIME_COL_RISK_OFF: (
                    float(regime_snapshot.get(REGIME_COL_RISK_OFF))
                    if regime_snapshot is not None and pd.notna(regime_snapshot.get(REGIME_COL_RISK_OFF)) else None
                ),
                REGIME_COL_DISPERSION_24H: (
                    float(regime_snapshot.get(REGIME_COL_DISPERSION_24H))
                    if regime_snapshot is not None and pd.notna(regime_snapshot.get(REGIME_COL_DISPERSION_24H)) else None
                ),
                REGIME_COL_TREND_STRENGTH_24H: (
                    float(regime_snapshot.get(REGIME_COL_TREND_STRENGTH_24H))
                    if regime_snapshot is not None and pd.notna(regime_snapshot.get(REGIME_COL_TREND_STRENGTH_24H)) else None
                ),
                REGIME_COL_VOLUME_REGIME_24H: (
                    float(regime_snapshot.get(REGIME_COL_VOLUME_REGIME_24H))
                    if regime_snapshot is not None and pd.notna(regime_snapshot.get(REGIME_COL_VOLUME_REGIME_24H)) else None
                ),
            }
        )

    final_ts = timestamps[-1]
    final_prices_group = prices_by_ts.get(final_ts)
    final_prices = (
        {row[sym_col]: float(row[price_col]) for _, row in final_prices_group.iterrows()}
        if final_prices_group is not None and not final_prices_group.empty
        else {}
    )
    for symbol in list(open_positions.keys()):
        pos = open_positions[symbol]
        px = final_prices.get(symbol, pos.entry_price)
        trade = finalize_trade(pos, final_ts, px, "forced_end", None)
        trades.append(trade)
        cash += pos.notional_usd + (pos.quantity * (px - pos.entry_price))
        risk_manager.update_realized_pnl(trade.pnl_usd, final_ts)
        risk_manager.register_exit(symbol, final_ts, forced=True)
        del open_positions[symbol]

    trade_columns = [
        "symbol",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "entry_prob",
        "entry_prob_percentile_rank",
        "entry_rolling_volatility_24h",
        "entry_predicted_time_to_peak_hours",
        "entry_score",
        "quantity",
        "notional_usd",
        "hold_hours",
        "gross_return_pct",
        "pnl_usd",
        "exit_reason",
        "model_name",
        "prob_size_multiplier",
        "vol_size_multiplier",
        "kelly_fraction",
        "kelly_multiplier",
        "total_size_multiplier",
    ]
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=trade_columns)

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)

    total_trades = int(len(trades_df))
    wins = int((trades_df["pnl_usd"] > 0).sum()) if total_trades else 0
    losses = int((trades_df["pnl_usd"] <= 0).sum()) if total_trades else 0
    total_pnl = float(trades_df["pnl_usd"].sum()) if total_trades else 0.0
    avg_pnl = float(trades_df["pnl_usd"].mean()) if total_trades else 0.0
    avg_return = float(trades_df["gross_return_pct"].mean()) if total_trades else 0.0
    avg_hold = float(trades_df["hold_hours"].mean()) if total_trades else 0.0
    avg_notional = float(trades_df["notional_usd"].mean()) if total_trades else 0.0
    avg_entry_score = float(trades_df["entry_score"].mean()) if total_trades and "entry_score" in trades_df else 0.0
    avg_prob_mult = float(trades_df["prob_size_multiplier"].mean()) if total_trades else 0.0
    avg_vol_mult = float(trades_df["vol_size_multiplier"].mean()) if total_trades else 0.0
    avg_kelly_fraction = float(trades_df["kelly_fraction"].mean()) if total_trades else 0.0
    avg_kelly_mult = float(trades_df["kelly_multiplier"].mean()) if total_trades else 0.0
    avg_total_mult = float(trades_df["total_size_multiplier"].mean()) if total_trades else 0.0
    win_rate = float(wins / total_trades) if total_trades else 0.0
    ending_equity = float(equity_df["equity_usd"].iloc[-1]) if not equity_df.empty else float(args.initial_cash)
    total_return = (ending_equity - args.initial_cash) / args.initial_cash if args.initial_cash else 0.0
    max_drawdown = compute_max_drawdown(equity_df["equity_usd"]) if not equity_df.empty else 0.0

    risk_metrics = risk_manager.summary_metrics()
    diagnostics.daily_loss_triggered = int(risk_metrics["daily_loss_triggered"])
    diagnostics.exposure_violations = int(risk_metrics["exposure_violations"])
    diagnostics.trades_blocked_by_risk = int(risk_metrics["trades_blocked_by_risk"])
    diagnostics.forced_exits = int(risk_metrics["forced_exits"])

    summary_rows = [
        {"metric_name": "input_prediction_rows", "metric_value": int(len(predictions))},
        {"metric_name": "input_price_rows", "metric_value": int(len(prices))},
        {"metric_name": "prediction_source", "metric_value": None},
        {"metric_name": "resolved_prob_column", "metric_value": None},
        {"metric_name": "resolved_score_column", "metric_value": None},
        {"metric_name": "prob_threshold", "metric_value": float(args.prob_threshold)},
        {"metric_name": "min_prob_percentile", "metric_value": float(args.min_prob_percentile)},
        {"metric_name": "prob_percentile_cutoff_value", "metric_value": prob_percentile_cutoff_value},
        {"metric_name": "ranking_mode", "metric_value": None},
        {"metric_name": "rank_score_min", "metric_value": float(args.rank_score_min) if args.rank_score_min is not None else None},
        {"metric_name": "prob_zscore_weight", "metric_value": float(args.prob_zscore_weight)},
        {"metric_name": "percentile_weight", "metric_value": float(args.percentile_weight)},
        {"metric_name": "volatility_weight", "metric_value": float(args.volatility_weight)},
        {"metric_name": "time_to_peak_weight", "metric_value": float(args.time_to_peak_weight)},
        {"metric_name": "enable_dynamic_max_positions", "metric_value": int(args.enable_dynamic_max_positions)},
        {"metric_name": "min_dynamic_max_positions", "metric_value": int(args.min_dynamic_max_positions)},
        {"metric_name": "dynamic_position_score_threshold", "metric_value": float(args.dynamic_position_score_threshold)},
        {
            "metric_name": "min_rolling_volatility_24h",
            "metric_value": float(args.min_rolling_volatility_24h) if args.min_rolling_volatility_24h is not None else None,
        },
        {
            "metric_name": "max_predicted_time_to_peak_hours",
            "metric_value": float(args.max_predicted_time_to_peak_hours) if args.max_predicted_time_to_peak_hours is not None else None,
        },
        {"metric_name": "enable_dynamic_sizing", "metric_value": int(args.enable_dynamic_sizing)},
        {"metric_name": "prob_size_cap", "metric_value": float(args.prob_size_cap)},
        {"metric_name": "vol_reference", "metric_value": float(args.vol_reference)},
        {"metric_name": "vol_size_floor", "metric_value": float(args.vol_size_floor)},
        {"metric_name": "vol_size_cap", "metric_value": float(args.vol_size_cap)},
        {"metric_name": "combined_size_cap", "metric_value": float(args.combined_size_cap)},
        {"metric_name": "enable_kelly_sizing", "metric_value": int(args.enable_kelly_sizing)},
        {"metric_name": "kelly_fraction_scale", "metric_value": float(args.kelly_fraction_scale)},
        {"metric_name": "kelly_size_cap", "metric_value": float(args.kelly_size_cap)},
        {"metric_name": "min_notional_per_trade", "metric_value": float(args.min_notional_per_trade)},
        {"metric_name": "top_n", "metric_value": int(args.top_n)},
        {"metric_name": "max_positions", "metric_value": int(args.max_positions)},
        {"metric_name": "notional_per_trade", "metric_value": float(args.notional_per_trade)},
        {"metric_name": "take_profit_pct", "metric_value": float(args.take_profit_pct)},
        {"metric_name": "stop_loss_pct", "metric_value": float(args.stop_loss_pct)},
        {"metric_name": "trailing_stop_pct", "metric_value": float(args.trailing_stop_pct)},
        {"metric_name": "trailing_stop_activation_pct", "metric_value": float(args.trailing_stop_activation_pct)},
        {"metric_name": "partial_take_profit_pct", "metric_value": float(args.partial_take_profit_pct)},
        {"metric_name": "partial_take_profit_fraction", "metric_value": float(args.partial_take_profit_fraction)},
        {"metric_name": "time_stop_hours", "metric_value": float(args.time_stop_hours) if args.time_stop_hours is not None else None},
        {"metric_name": "time_stop_min_return_pct", "metric_value": float(args.time_stop_min_return_pct)},
        {"metric_name": "max_hold_hours", "metric_value": float(args.max_hold_hours)},
        {"metric_name": "enable_risk_manager", "metric_value": int(args.enable_risk_manager)},
        {"metric_name": "max_daily_loss_usd", "metric_value": float(args.max_daily_loss_usd) if args.max_daily_loss_usd is not None else None},
        {"metric_name": "max_total_exposure_usd", "metric_value": float(args.max_total_exposure_usd) if args.max_total_exposure_usd is not None else None},
        {"metric_name": "max_total_exposure_pct", "metric_value": float(args.max_total_exposure_pct) if args.max_total_exposure_pct is not None else None},
        {"metric_name": "max_exposure_per_symbol_usd", "metric_value": float(args.max_exposure_per_symbol_usd) if args.max_exposure_per_symbol_usd is not None else None},
        {"metric_name": "max_drawdown_pct", "metric_value": float(args.max_drawdown_pct) if args.max_drawdown_pct is not None else None},
        {"metric_name": "cooldown_minutes_per_symbol", "metric_value": float(args.cooldown_minutes_per_symbol)},
        {"metric_name": "enable_drawdown_scaling", "metric_value": int(args.enable_drawdown_scaling)},
        {"metric_name": "drawdown_full_size_pct", "metric_value": float(args.drawdown_full_size_pct)},
        {"metric_name": "drawdown_half_size_pct", "metric_value": float(args.drawdown_half_size_pct)},
        {"metric_name": "drawdown_quarter_size_pct", "metric_value": float(args.drawdown_quarter_size_pct)},
        {"metric_name": "drawdown_half_size_multiplier", "metric_value": float(args.drawdown_half_size_multiplier)},
        {"metric_name": "drawdown_quarter_size_multiplier", "metric_value": float(args.drawdown_quarter_size_multiplier)},
        {"metric_name": "enable_regime_gating", "metric_value": int(args.enable_regime_gating)},
        {"metric_name": "regime_gating_mode", "metric_value": None},
        {"metric_name": "max_market_risk_off_score", "metric_value": float(args.max_market_risk_off_score) if args.max_market_risk_off_score is not None else None},
        {"metric_name": "max_market_dispersion_24h", "metric_value": float(args.max_market_dispersion_24h) if args.max_market_dispersion_24h is not None else None},
        {"metric_name": "min_market_trend_strength_24h", "metric_value": float(args.min_market_trend_strength_24h) if args.min_market_trend_strength_24h is not None else None},
        {"metric_name": "risk_off_size_multiplier", "metric_value": float(args.risk_off_size_multiplier)},
        {"metric_name": "risk_off_score_raise", "metric_value": float(args.risk_off_score_raise)},
        {"metric_name": "rank_score_min_base", "metric_value": float(args.rank_score_min) if args.rank_score_min is not None else None},
        {"metric_name": "timestamps_considered", "metric_value": diagnostics.timestamps_considered},
        {"metric_name": "candidate_rows_seen", "metric_value": diagnostics.candidate_rows_seen},
        {"metric_name": "candidate_rows_after_prob_threshold", "metric_value": diagnostics.candidate_rows_after_prob_threshold},
        {"metric_name": "candidate_rows_after_percentile", "metric_value": diagnostics.candidate_rows_after_percentile},
        {"metric_name": "candidate_rows_after_volatility", "metric_value": diagnostics.candidate_rows_after_volatility},
        {"metric_name": "candidate_rows_after_time_to_peak", "metric_value": diagnostics.candidate_rows_after_time_to_peak},
        {"metric_name": "candidate_rows_after_rank_score", "metric_value": diagnostics.candidate_rows_after_rank_score},
        {"metric_name": "candidate_rows_after_regime_gate", "metric_value": diagnostics.candidate_rows_after_regime_gate},
        {"metric_name": "entries_submitted", "metric_value": diagnostics.entries_submitted},
        {"metric_name": "entries_opened", "metric_value": diagnostics.entries_opened},
        {"metric_name": "skipped_already_open", "metric_value": diagnostics.skipped_already_open},
        {"metric_name": "skipped_missing_price", "metric_value": diagnostics.skipped_missing_price},
        {"metric_name": "skipped_cash", "metric_value": diagnostics.skipped_cash},
        {"metric_name": "skipped_position_cap", "metric_value": diagnostics.skipped_position_cap},
        {"metric_name": "sized_below_min_notional", "metric_value": diagnostics.sized_below_min_notional},
        {"metric_name": "partial_take_profit_events", "metric_value": diagnostics.partial_take_profit_events},
        {"metric_name": "trailing_stop_exits", "metric_value": diagnostics.trailing_stop_exits},
        {"metric_name": "time_stop_exits", "metric_value": diagnostics.time_stop_exits},
        {"metric_name": "daily_loss_triggered", "metric_value": diagnostics.daily_loss_triggered},
        {"metric_name": "exposure_violations", "metric_value": diagnostics.exposure_violations},
        {"metric_name": "trades_blocked_by_risk", "metric_value": diagnostics.trades_blocked_by_risk},
        {"metric_name": "regime_gate_blocks", "metric_value": diagnostics.regime_gate_blocks},
        {"metric_name": "regime_scale_events", "metric_value": diagnostics.regime_scale_events},
        {"metric_name": "regime_score_raise_events", "metric_value": diagnostics.regime_score_raise_events},
        {"metric_name": "forced_exits", "metric_value": diagnostics.forced_exits},
        {"metric_name": "kill_switch_activations", "metric_value": risk_metrics["kill_switch_activations"]},
        {"metric_name": "drawdown_breach_events", "metric_value": risk_metrics["drawdown_breach_events"]},
        {"metric_name": "cooldown_blocks", "metric_value": risk_metrics["cooldown_blocks"]},
        {"metric_name": "daily_loss_blocks", "metric_value": risk_metrics["daily_loss_blocks"]},
        {"metric_name": "risk_safe_mode_active", "metric_value": risk_metrics["safe_mode_active"]},
        {"metric_name": "risk_kill_switch_active", "metric_value": risk_metrics["kill_switch_active"]},
        {"metric_name": "ending_total_exposure_usd", "metric_value": risk_metrics["ending_total_exposure_usd"]},
        {"metric_name": "ending_daily_pnl_usd", "metric_value": risk_metrics["ending_daily_pnl_usd"]},
        {"metric_name": "ending_drawdown_pct", "metric_value": risk_metrics["ending_drawdown_pct"]},
        {"metric_name": "drawdown_scaling_half_count", "metric_value": risk_metrics["drawdown_scaling_half_count"]},
        {"metric_name": "drawdown_scaling_quarter_count", "metric_value": risk_metrics["drawdown_scaling_quarter_count"]},
        {"metric_name": "drawdown_scaling_stop_count", "metric_value": risk_metrics["drawdown_scaling_stop_count"]},
        {"metric_name": "candidate_rate_after_prob_threshold", "metric_value": (diagnostics.candidate_rows_after_prob_threshold / diagnostics.candidate_rows_seen) if diagnostics.candidate_rows_seen else None},
        {"metric_name": "candidate_rate_after_regime_gate", "metric_value": (diagnostics.candidate_rows_after_regime_gate / diagnostics.candidate_rows_after_rank_score) if diagnostics.candidate_rows_after_rank_score else None},
        {"metric_name": "entry_open_rate", "metric_value": (diagnostics.entries_opened / diagnostics.entries_submitted) if diagnostics.entries_submitted else None},
        {"metric_name": "trade_rate_vs_candidates", "metric_value": (total_trades / diagnostics.candidate_rows_after_prob_threshold) if diagnostics.candidate_rows_after_prob_threshold else None},
        {"metric_name": "trades", "metric_value": total_trades},
        {"metric_name": "wins", "metric_value": wins},
        {"metric_name": "losses", "metric_value": losses},
        {"metric_name": "win_rate", "metric_value": win_rate},
        {"metric_name": "total_pnl_usd", "metric_value": total_pnl},
        {"metric_name": "avg_pnl_usd", "metric_value": avg_pnl},
        {"metric_name": "avg_return_pct", "metric_value": avg_return},
        {"metric_name": "avg_hold_hours", "metric_value": avg_hold},
        {"metric_name": "avg_notional_usd", "metric_value": avg_notional},
        {"metric_name": "avg_entry_score", "metric_value": avg_entry_score},
        {"metric_name": "avg_prob_size_multiplier", "metric_value": avg_prob_mult},
        {"metric_name": "avg_vol_size_multiplier", "metric_value": avg_vol_mult},
        {"metric_name": "avg_kelly_fraction", "metric_value": avg_kelly_fraction},
        {"metric_name": "avg_kelly_multiplier", "metric_value": avg_kelly_mult},
        {"metric_name": "avg_total_size_multiplier", "metric_value": avg_total_mult},
        {"metric_name": "starting_equity_usd", "metric_value": float(args.initial_cash)},
        {"metric_name": "ending_equity_usd", "metric_value": ending_equity},
        {"metric_name": "total_return_pct", "metric_value": float(total_return)},
        {"metric_name": "max_drawdown_pct", "metric_value": float(max_drawdown)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["metric_value"] = summary_df["metric_value"].astype(object)
    summary_df.loc[summary_df["metric_name"] == "ranking_mode", "metric_value"] = args.ranking_mode
    summary_df.loc[summary_df["metric_name"] == "prediction_source", "metric_value"] = args.prediction_source
    summary_df.loc[summary_df["metric_name"] == "resolved_prob_column", "metric_value"] = args.prob_column
    summary_df.loc[summary_df["metric_name"] == "resolved_score_column", "metric_value"] = resolved_score_col
    summary_df.loc[summary_df["metric_name"] == "regime_gating_mode", "metric_value"] = args.regime_gating_mode

    return trades_df, equity_df, summary_df


def write_table(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str, if_exists: str) -> None:
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)


def run_replay(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(args.db_path)
    try:
        predictions = load_predictions(conn, args)
        prices = load_prices(conn, args)
        context = load_context(conn, args)
        predictions = enrich_predictions(predictions, context, args)
        trades_df, equity_df, summary_df = replay_strategy(predictions, prices, args)
        write_table(conn, trades_df, args.trades_table, args.if_exists)
        write_table(conn, equity_df, args.equity_table, args.if_exists)
        write_table(conn, summary_df, args.summary_table, args.if_exists)
        return trades_df, equity_df, summary_df
    finally:
        conn.close()



__all__ = [
    "build_runtime_args",
    "instantiate_risk_manager",
    "resolve_prediction_columns",
    "load_predictions",
    "load_prices",
    "load_context",
    "enrich_predictions",
    "compute_kelly_terms",
    "compute_size_multipliers",
    "filter_ranked_candidates",
    "compute_entry_decision",
    "apply_regime_gating",
    "_compute_dynamic_position_limit",
    "replay_strategy",
    "run_replay",
]

def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    LOGGER.info("Opening SQLite database: %s", args.db_path)
    conn = sqlite3.connect(args.db_path)
    try:
        predictions = load_predictions(conn, args)
        prices = load_prices(conn, args)
        context = load_context(conn, args)
        predictions = enrich_predictions(predictions, context, args)

        LOGGER.info(
            "Loaded predictions=%d rows from %s, prices=%d rows from %s, context=%d rows from %s, source=%s",
            len(predictions),
            args.predictions_table,
            len(prices),
            args.price_table,
            len(context),
            args.context_table,
            args.prediction_source,
        )

        trades_df, equity_df, summary_df = replay_strategy(predictions, prices, args)
        write_table(conn, trades_df, args.trades_table, args.if_exists)
        write_table(conn, equity_df, args.equity_table, args.if_exists)
        write_table(conn, summary_df, args.summary_table, args.if_exists)

        LOGGER.info("Wrote trades -> %s", args.trades_table)
        LOGGER.info("Wrote equity -> %s", args.equity_table)
        LOGGER.info("Wrote summary -> %s", args.summary_table)
        LOGGER.info("Replay summary:\n%s", summary_df.to_string(index=False))
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
