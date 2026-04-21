#!/usr/bin/env python3
"""Grid search runner for ML replay strategy.

Compatible with replay_ml_strategy.py versions that support regime gating.
Adds:
- regime gating sweep dimensions
- hard cap on total generated runs (< 1096)
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger("sweep_ml_replay")
MAX_SWEEP_RUNS = 1095


def _nested_get(mapping: dict, path: tuple[str, ...], default=None):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_sweep_defaults(config_path: str | None) -> dict:
    """Load sweep defaults exclusively from the top-level `ml_replay` config section."""
    if not config_path:
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    section = _nested_get(cfg, ("ml_replay",), {})
    return section if isinstance(section, dict) else {}


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import replay_ml_strategy as replay_mod  # noqa: E402

try:
    from tradarbot.ml.ensemble import build_ensemble_predictions as build_ensemble_predictions_mod
except Exception:  # pragma: no cover
    build_ensemble_predictions_mod = None


@dataclass(frozen=True)
class SweepConfig:
    prob_threshold: float
    take_profit_pct: float
    stop_loss_pct: float
    top_n: int
    max_positions: int
    kelly_fraction_scale: float
    combined_size_cap: float
    max_drawdown_pct: float
    drawdown_full_size_pct: float
    drawdown_half_size_pct: float
    drawdown_quarter_size_pct: float
    drawdown_half_size_multiplier: float
    drawdown_quarter_size_multiplier: float
    weight_6h: float
    weight_24h: float
    weight_72h: float
    max_market_risk_off_score: float | None
    max_market_dispersion_24h: float | None
    min_market_trend_strength_24h: float | None
    risk_off_size_multiplier: float
    risk_off_score_raise: float


@dataclass
class SweepRow:
    run_id: int
    prediction_source: str
    predictions_table: str
    regime_gating_mode: str
    enable_regime_gating: int
    weight_6h: float | None
    weight_24h: float | None
    weight_72h: float | None
    prob_threshold: float
    take_profit_pct: float
    stop_loss_pct: float
    top_n: int
    max_positions: int
    enable_dynamic_sizing: int
    enable_kelly_sizing: int
    enable_dynamic_max_positions: int
    enable_risk_manager: int
    enable_drawdown_scaling: int
    kelly_fraction_scale: float
    combined_size_cap: float
    max_drawdown_pct: float
    drawdown_full_size_pct: float
    drawdown_half_size_pct: float
    drawdown_quarter_size_pct: float
    drawdown_half_size_multiplier: float
    drawdown_quarter_size_multiplier: float
    max_market_risk_off_score: float | None
    max_market_dispersion_24h: float | None
    min_market_trend_strength_24h: float | None
    risk_off_size_multiplier: float | None
    risk_off_score_raise: float | None
    ranking_mode: str | None
    resolved_prob_column: str | None
    resolved_score_column: str | None
    total_pnl_usd: float | None
    total_return_pct: float | None
    max_drawdown_pct_realized: float | None
    trades: int | None
    wins: int | None
    losses: int | None
    win_rate: float | None
    avg_pnl_usd: float | None
    avg_return_pct: float | None
    avg_hold_hours: float | None
    avg_notional_usd: float | None
    avg_prob_size_multiplier: float | None
    avg_vol_size_multiplier: float | None
    avg_kelly_fraction: float | None
    avg_kelly_multiplier: float | None
    avg_total_size_multiplier: float | None
    trades_blocked_by_risk: float | None
    regime_gate_blocks: float | None
    regime_scale_events: float | None
    regime_score_raise_events: float | None
    drawdown_scaling_half_count: float | None
    drawdown_scaling_quarter_count: float | None
    drawdown_scaling_stop_count: float | None
    pnl_to_abs_drawdown: float | None


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/tradar.yaml")
    known, _ = pre.parse_known_args()
    defaults = _load_sweep_defaults(known.config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--db-path", default=defaults.get("db_path", "tradarbot.db"))
    parser.add_argument("--predictions-table", default=defaults.get("predictions_table", "spike_model_predictions_hgb"))
    parser.add_argument("--price-table", default=defaults.get("price_table", "spike_base_rows"))
    parser.add_argument("--context-table", default=defaults.get("context_table", "spike_training_rows"))

    parser.add_argument("--prediction-source", choices=["ensemble", "6h", "24h", "72h", "direct"], default=defaults.get("prediction_source", "direct"))
    parser.add_argument("--score-column", default=defaults.get("score_column"))

    parser.add_argument("--prob-thresholds", nargs="+", type=float, default=defaults.get("prob_thresholds", [defaults.get("prob_threshold", 0.10)]), required=False)
    parser.add_argument("--take-profit-pcts", nargs="+", type=float, default=defaults.get("take_profit_pcts", [defaults.get("take_profit_pct", 0.20)]), required=False)
    parser.add_argument("--stop-loss-pcts", nargs="+", type=float, default=defaults.get("stop_loss_pcts", [defaults.get("stop_loss_pct", 0.06)]), required=False)
    parser.add_argument("--top-n-values", nargs="+", type=int, default=defaults.get("top_n_values", [defaults.get("top_n", 1)]))
    parser.add_argument("--max-positions-values", nargs="+", type=int, default=defaults.get("max_positions_values", [defaults.get("max_positions", 3)]))
    parser.add_argument("--kelly-fraction-scales", nargs="+", type=float, default=defaults.get("kelly_fraction_scales", [defaults.get("kelly_fraction_scale", 0.40)]))
    parser.add_argument("--combined-size-caps", nargs="+", type=float, default=defaults.get("combined_size_caps", [defaults.get("combined_size_cap", 2.2)]))

    parser.add_argument("--enable-dynamic-sizing", action="store_true", default=defaults.get("enable_dynamic_sizing", False))
    parser.add_argument("--enable-kelly-sizing", action="store_true", default=defaults.get("enable_kelly_sizing", False))

    parser.add_argument("--symbol-column", default=defaults.get("symbol_column", "symbol"))
    parser.add_argument("--timestamp-column", default=defaults.get("timestamp_column", "timestamp"))
    parser.add_argument("--price-column", default=defaults.get("price_column", "price_close"))
    parser.add_argument("--prob-column", default=defaults.get("prob_column", "pred_prob"))
    parser.add_argument("--model-name-column", default=defaults.get("model_name_column", "model_name"))
    parser.add_argument("--rolling-volatility-column", default=defaults.get("rolling_volatility_column", "rolling_volatility_24h"))
    parser.add_argument("--time-to-peak-seconds-column", default=defaults.get("time_to_peak_seconds_column", "target_time_to_peak_seconds_24h"))

    parser.add_argument("--min-prob-percentile", type=float, default=defaults.get("min_prob_percentile", 0.0))
    parser.add_argument("--min-rolling-volatility-24h", type=float, default=defaults.get("min_rolling_volatility_24h"))
    parser.add_argument("--max-predicted-time-to-peak-hours", type=float, default=defaults.get("max_predicted_time_to_peak_hours"))
    parser.add_argument("--notional-per-trade", type=float, default=defaults.get("notional_per_trade", 1000.0))
    parser.add_argument("--min-notional-per-trade", type=float, default=defaults.get("min_notional_per_trade", 400.0))
    parser.add_argument("--initial-cash", type=float, default=defaults.get("initial_cash", 100000.0))

    parser.add_argument("--ranking-mode", choices=["probability", "composite"], default=defaults.get("ranking_mode", "probability"))
    parser.add_argument("--prob-zscore-weight", type=float, default=defaults.get("prob_zscore_weight", 1.0))
    parser.add_argument("--percentile-weight", type=float, default=defaults.get("percentile_weight", 0.0))
    parser.add_argument("--volatility-weight", type=float, default=defaults.get("volatility_weight", 0.0))
    parser.add_argument("--time-to-peak-weight", type=float, default=defaults.get("time_to_peak_weight", 0.0))
    parser.add_argument("--rank-score-min", type=float, default=defaults.get("rank_score_min"))

    parser.add_argument("--enable-dynamic-max-positions", action="store_true", default=defaults.get("enable_dynamic_max_positions", False))
    parser.add_argument("--min-dynamic-max-positions", type=int, default=defaults.get("min_dynamic_max_positions", 1))
    parser.add_argument("--dynamic-position-score-threshold", type=float, default=defaults.get("dynamic_position_score_threshold", 0.0))

    parser.add_argument("--prob-size-cap", type=float, default=defaults.get("prob_size_cap", 2.0))
    parser.add_argument("--vol-reference", type=float, default=defaults.get("vol_reference", 0.006))
    parser.add_argument("--vol_size_floor", type=float, default=defaults.get("vol_size_floor", 0.75))
    parser.add_argument("--vol-size-floor", dest="vol_size_floor", type=float)
    parser.add_argument("--vol_size_cap", type=float, default=defaults.get("vol_size_cap", 1.25))
    parser.add_argument("--vol-size-cap", dest="vol_size_cap", type=float)
    parser.add_argument("--kelly-probability-mode", choices=["raw", "threshold_relative"], default=defaults.get("kelly_probability_mode", "threshold_relative"))
    parser.add_argument("--kelly-size-cap", type=float, default=defaults.get("kelly_size_cap", 1.5))
    parser.add_argument("--max-hold-hours", type=float, default=defaults.get("max_hold_hours", 24.0))

    parser.add_argument("--trailing-stop-pct", type=float, default=defaults.get("trailing_stop_pct"))
    parser.add_argument("--trailing-stop-activation-pct", type=float, default=defaults.get("trailing_stop_activation_pct"))
    parser.add_argument("--partial-take-profit-pct", type=float, default=defaults.get("partial_take_profit_pct"))
    parser.add_argument("--partial-take-profit-fraction", type=float, default=defaults.get("partial_take_profit_fraction", 0.5))
    parser.add_argument("--time-stop-hours", type=float, default=defaults.get("time_stop_hours"))
    parser.add_argument("--time-stop-min-return-pct", type=float, default=defaults.get("time_stop_min_return_pct", 0.0))

    parser.add_argument("--enable-risk-manager", action="store_true", default=defaults.get("enable_risk_manager", False))
    parser.add_argument("--max-daily-loss-usd", type=float, default=defaults.get("max_daily_loss_usd"))
    parser.add_argument("--max-total-exposure-usd", type=float, default=defaults.get("max_total_exposure_usd"))
    parser.add_argument("--max-total-exposure-pct", type=float, default=defaults.get("max_total_exposure_pct"))
    parser.add_argument("--max-exposure-per-symbol-usd", type=float, default=defaults.get("max_exposure_per_symbol_usd"))
    parser.add_argument("--cooldown-minutes-per-symbol", type=float, default=defaults.get("cooldown_minutes_per_symbol", 0.0))

    parser.add_argument("--enable-drawdown-scaling", action="store_true", default=defaults.get("enable_drawdown_scaling", False))
    parser.add_argument("--max-drawdown-pcts", nargs="+", type=float, default=defaults.get("max_drawdown_pcts", [defaults.get("max_drawdown_pct", 0.08)]))
    parser.add_argument("--drawdown-full-size-pcts", nargs="+", type=float, default=defaults.get("drawdown_full_size_pcts", [defaults.get("drawdown_full_size_pct", 0.04)]))
    parser.add_argument("--drawdown-half-size-pcts", nargs="+", type=float, default=defaults.get("drawdown_half_size_pcts", [defaults.get("drawdown_half_size_pct", 0.06)]))
    parser.add_argument("--drawdown-quarter-size-pcts", nargs="+", type=float, default=defaults.get("drawdown_quarter_size_pcts", [defaults.get("drawdown_quarter_size_pct", 0.08)]))
    parser.add_argument("--drawdown-half-size-multipliers", nargs="+", type=float, default=defaults.get("drawdown_half_size_multipliers", [defaults.get("drawdown_half_size_multiplier", 0.50)]))
    parser.add_argument("--drawdown-quarter-size-multipliers", nargs="+", type=float, default=defaults.get("drawdown_quarter_size_multipliers", [defaults.get("drawdown_quarter_size_multiplier", 0.25)]))

    parser.add_argument("--enable-regime-gating", action="store_true", default=defaults.get("enable_regime_gating", False))
    parser.add_argument("--regime-gating-mode", choices=["block", "scale", "score_raise"], default=defaults.get("regime_gating_mode", "block"))
    parser.add_argument("--max-market-risk-off-scores", nargs="+", type=float, default=defaults.get("max_market_risk_off_scores", [defaults.get("max_market_risk_off_score")]) if defaults.get("max_market_risk_off_score") is not None else None)
    parser.add_argument("--max-market-dispersion-24hs", nargs="+", type=float, default=defaults.get("max_market_dispersion_24hs"))
    parser.add_argument("--min-market-trend-strength-24hs", nargs="+", type=float, default=defaults.get("min_market_trend_strength_24hs"))
    parser.add_argument("--risk-off-size-multipliers", nargs="+", type=float, default=defaults.get("risk_off_size_multipliers", [defaults.get("risk_off_size_multiplier", 0.50)]))
    parser.add_argument("--risk-off-score-raises", nargs="+", type=float, default=defaults.get("risk_off_score_raises", [defaults.get("risk_off_score_raise", 0.0)]))

    parser.add_argument("--weight-6h", nargs="+", type=float, default=defaults.get("weight_6h", [0.3]) if isinstance(defaults.get("weight_6h"), list) else [defaults.get("weight_6h", 0.3)])
    parser.add_argument("--weight-24h", nargs="+", type=float, default=defaults.get("weight_24h", [0.5]) if isinstance(defaults.get("weight_24h"), list) else [defaults.get("weight_24h", 0.5)])
    parser.add_argument("--weight-72h", nargs="+", type=float, default=defaults.get("weight_72h", [0.2]) if isinstance(defaults.get("weight_72h"), list) else [defaults.get("weight_72h", 0.2)])
    parser.add_argument("--rebuild-ensemble-per-run", action="store_true", default=defaults.get("rebuild_ensemble_per_run", False))
    parser.add_argument("--ensemble-output-table", default=defaults.get("ensemble_output_table", "spike_model_predictions_ensemble"))
    parser.add_argument("--ensemble-table-6h", default=defaults.get("ensemble_table_6h", "spike_model_predictions_6h"))
    parser.add_argument("--ensemble-table-24h", default=defaults.get("ensemble_table_24h", "spike_model_predictions_24h"))
    parser.add_argument("--ensemble-table-72h", default=defaults.get("ensemble_table_72h", "spike_model_predictions_72h"))
    parser.add_argument("--ensemble-agreement-threshold", type=float, default=defaults.get("ensemble_agreement_threshold", 0.8))
    parser.add_argument("--ensemble-agreement-boost", type=float, default=defaults.get("ensemble_agreement_boost", 0.05))

    parser.add_argument("--results-table", default=defaults.get("results_table", "ml_replay_sweep_results"))
    parser.add_argument("--summary-table-prefix", default=defaults.get("summary_table_prefix", "ml_replay_sweep_summary"))
    parser.add_argument("--trades-table-prefix", default=defaults.get("trades_table_prefix", "ml_replay_sweep_trades"))
    parser.add_argument("--equity-table-prefix", default=defaults.get("equity_table_prefix", "ml_replay_sweep_equity"))
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default=defaults.get("if_exists", "replace"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=defaults.get("log_level", "INFO"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _weights_sum_to_one(weight_6h: float, weight_24h: float, weight_72h: float, tol: float = 1e-9) -> bool:
    return abs((weight_6h + weight_24h + weight_72h) - 1.0) <= tol


def _normalize_optional_sweep(values: list[float] | None, default_value: float | None) -> list[float | None]:
    if values is None:
        return [default_value]
    return list(values)


def build_grid(args: argparse.Namespace) -> list[SweepConfig]:
    risk_off_scores = _normalize_optional_sweep(args.max_market_risk_off_scores, None)
    dispersion_vals = _normalize_optional_sweep(args.max_market_dispersion_24hs, None)
    trend_vals = _normalize_optional_sweep(args.min_market_trend_strength_24hs, None)
    size_mults = _normalize_optional_sweep(args.risk_off_size_multipliers, 0.50)
    score_raises = _normalize_optional_sweep(args.risk_off_score_raises, 0.0)

    if not args.enable_regime_gating:
        risk_off_scores = [None]
        dispersion_vals = [None]
        trend_vals = [None]
        size_mults = [0.50]
        score_raises = [0.0]
    elif args.regime_gating_mode == "block":
        size_mults = [0.50]
        score_raises = [0.0]
    elif args.regime_gating_mode == "scale":
        score_raises = [0.0]
    elif args.regime_gating_mode == "score_raise":
        size_mults = [0.50]

    configs: list[SweepConfig] = []
    for values in itertools.product(
        args.prob_thresholds,
        args.take_profit_pcts,
        args.stop_loss_pcts,
        args.top_n_values,
        args.max_positions_values,
        args.kelly_fraction_scales,
        args.combined_size_caps,
        args.max_drawdown_pcts,
        args.drawdown_full_size_pcts,
        args.drawdown_half_size_pcts,
        args.drawdown_quarter_size_pcts,
        args.drawdown_half_size_multipliers,
        args.drawdown_quarter_size_multipliers,
        args.weight_6h,
        args.weight_24h,
        args.weight_72h,
        risk_off_scores,
        dispersion_vals,
        trend_vals,
        size_mults,
        score_raises,
    ):
        cfg = SweepConfig(*values)
        if not (cfg.drawdown_full_size_pct <= cfg.drawdown_half_size_pct <= cfg.drawdown_quarter_size_pct <= cfg.max_drawdown_pct):
            continue
        if args.prediction_source == "ensemble" and not _weights_sum_to_one(cfg.weight_6h, cfg.weight_24h, cfg.weight_72h):
            continue
        configs.append(cfg)
        if len(configs) > MAX_SWEEP_RUNS:
            raise ValueError(
                f"Sweep would generate {len(configs)}+ configurations, exceeding the hard limit of {MAX_SWEEP_RUNS}. "
                "Reduce the number of sweep values."
            )
    return configs


def per_run_table_name(prefix: str, run_id: int) -> str:
    return f"{prefix}_{run_id:04d}"


def summary_to_dict(summary_df: pd.DataFrame) -> dict[str, Any]:
    required = {"metric_name", "metric_value"}
    missing = required - set(summary_df.columns)
    if missing:
        raise KeyError(f"Summary DataFrame missing required columns: {sorted(missing)}")
    out: dict[str, Any] = {}
    for _, row in summary_df.iterrows():
        key = str(row["metric_name"])
        value = row["metric_value"]
        if pd.isna(value):
            out[key] = None
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def safe_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_text(metrics: dict[str, Any], key: str) -> str | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def derive_pnl_to_drawdown(total_pnl_usd: float | None, max_drawdown_pct: float | None) -> float | None:
    if total_pnl_usd is None or max_drawdown_pct is None:
        return None
    denom = abs(max_drawdown_pct)
    if denom <= 1e-12:
        return None
    return float(total_pnl_usd / denom)


def build_ensemble_predictions(*, args: argparse.Namespace, cfg: SweepConfig) -> str:
    if build_ensemble_predictions_mod is None:
        raise RuntimeError(
            "tradarbot.ml.ensemble.build_ensemble_predictions could not be imported. "
            "Create/restore tradarbot/ml/ensemble.py before using ensemble sweep mode."
        )
    try:
        build_ensemble_predictions_mod(
            db_path=args.db_path,
            w6=cfg.weight_6h,
            w24=cfg.weight_24h,
            w72=cfg.weight_72h,
            agreement_boost=float(args.ensemble_agreement_boost),
            threshold=float(args.ensemble_agreement_threshold),
            table_6h=args.ensemble_table_6h,
            table_24h=args.ensemble_table_24h,
            table_72h=args.ensemble_table_72h,
            output_table=args.ensemble_output_table,
        )
    except TypeError:
        build_ensemble_predictions_mod(
            db_path=args.db_path,
            w6=cfg.weight_6h,
            w24=cfg.weight_24h,
            w72=cfg.weight_72h,
            agreement_boost=float(args.ensemble_agreement_boost),
            threshold=float(args.ensemble_agreement_threshold),
        )
    return args.ensemble_output_table


def make_replay_args(args: argparse.Namespace, cfg: SweepConfig, run_id: int) -> argparse.Namespace:
    predictions_table = args.predictions_table
    if args.prediction_source == "ensemble" and args.rebuild_ensemble_per_run:
        predictions_table = args.ensemble_output_table

    return argparse.Namespace(
        db_path=args.db_path,
        predictions_table=predictions_table,
        price_table=args.price_table,
        context_table=args.context_table,
        prediction_source=args.prediction_source,
        score_column=args.score_column,
        symbol_column=args.symbol_column,
        timestamp_column=args.timestamp_column,
        price_column=args.price_column,
        prob_column=args.prob_column,
        model_name_column=args.model_name_column,
        rolling_volatility_column=args.rolling_volatility_column,
        time_to_peak_seconds_column=args.time_to_peak_seconds_column,
        prob_threshold=cfg.prob_threshold,
        min_prob_percentile=args.min_prob_percentile,
        min_rolling_volatility_24h=args.min_rolling_volatility_24h,
        max_predicted_time_to_peak_hours=args.max_predicted_time_to_peak_hours,
        top_n=cfg.top_n,
        max_positions=cfg.max_positions,
        ranking_mode=args.ranking_mode,
        prob_zscore_weight=args.prob_zscore_weight,
        percentile_weight=args.percentile_weight,
        volatility_weight=args.volatility_weight,
        time_to_peak_weight=args.time_to_peak_weight,
        rank_score_min=args.rank_score_min,
        enable_dynamic_max_positions=args.enable_dynamic_max_positions,
        min_dynamic_max_positions=args.min_dynamic_max_positions,
        dynamic_position_score_threshold=args.dynamic_position_score_threshold,
        enable_dynamic_sizing=args.enable_dynamic_sizing,
        notional_per_trade=args.notional_per_trade,
        min_notional_per_trade=args.min_notional_per_trade,
        initial_cash=args.initial_cash,
        prob_size_cap=args.prob_size_cap,
        vol_reference=args.vol_reference,
        vol_size_floor=args.vol_size_floor,
        vol_size_cap=args.vol_size_cap,
        combined_size_cap=cfg.combined_size_cap,
        enable_kelly_sizing=args.enable_kelly_sizing,
        kelly_fraction_scale=cfg.kelly_fraction_scale,
        kelly_probability_mode=args.kelly_probability_mode,
        kelly_size_cap=args.kelly_size_cap,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
        max_hold_hours=args.max_hold_hours,
        trailing_stop_pct=(0.05 if args.trailing_stop_pct is None else args.trailing_stop_pct),
        trailing_stop_activation_pct=(0.08 if args.trailing_stop_activation_pct is None else args.trailing_stop_activation_pct),
        partial_take_profit_pct=(0.10 if args.partial_take_profit_pct is None else args.partial_take_profit_pct),
        partial_take_profit_fraction=args.partial_take_profit_fraction,
        time_stop_hours=args.time_stop_hours,
        time_stop_min_return_pct=args.time_stop_min_return_pct,
        enable_risk_manager=args.enable_risk_manager,
        max_daily_loss_usd=args.max_daily_loss_usd,
        max_total_exposure_usd=args.max_total_exposure_usd,
        max_total_exposure_pct=args.max_total_exposure_pct,
        max_exposure_per_symbol_usd=args.max_exposure_per_symbol_usd,
        max_drawdown_pct=cfg.max_drawdown_pct,
        cooldown_minutes_per_symbol=args.cooldown_minutes_per_symbol,
        enable_drawdown_scaling=args.enable_drawdown_scaling,
        drawdown_full_size_pct=cfg.drawdown_full_size_pct,
        drawdown_half_size_pct=cfg.drawdown_half_size_pct,
        drawdown_quarter_size_pct=cfg.drawdown_quarter_size_pct,
        drawdown_half_size_multiplier=cfg.drawdown_half_size_multiplier,
        drawdown_quarter_size_multiplier=cfg.drawdown_quarter_size_multiplier,
        enable_regime_gating=args.enable_regime_gating,
        regime_gating_mode=args.regime_gating_mode,
        max_market_risk_off_score=cfg.max_market_risk_off_score,
        max_market_dispersion_24h=cfg.max_market_dispersion_24h,
        min_market_trend_strength_24h=cfg.min_market_trend_strength_24h,
        risk_off_size_multiplier=cfg.risk_off_size_multiplier,
        risk_off_score_raise=cfg.risk_off_score_raise,
        trades_table=per_run_table_name(args.trades_table_prefix, run_id),
        equity_table=per_run_table_name(args.equity_table_prefix, run_id),
        summary_table=per_run_table_name(args.summary_table_prefix, run_id),
        if_exists="replace",
        log_level=args.log_level,
    )


def persist_run_outputs(conn: sqlite3.Connection, trades_df: pd.DataFrame, equity_df: pd.DataFrame, summary_df: pd.DataFrame, replay_args: argparse.Namespace, run_id: int) -> None:
    trades_out = trades_df.copy()
    equity_out = equity_df.copy()
    summary_out = summary_df.copy()
    trades_out["run_id"] = run_id
    equity_out["run_id"] = run_id
    summary_out["run_id"] = run_id
    trades_out.to_sql(replay_args.trades_table, conn, if_exists="replace", index=False)
    equity_out.to_sql(replay_args.equity_table, conn, if_exists="replace", index=False)
    summary_out.to_sql(replay_args.summary_table, conn, if_exists="replace", index=False)


def run_one(args: argparse.Namespace, conn: sqlite3.Connection, run_id: int, total_runs: int, cfg: SweepConfig) -> SweepRow:
    replay_args = make_replay_args(args, cfg, run_id)
    if args.prediction_source == "ensemble" and args.rebuild_ensemble_per_run:
        replay_args.predictions_table = build_ensemble_predictions(args=args, cfg=cfg)

    LOGGER.info(
        "Sweep run %d/%d: source=%s preds=%s prob=%.4f tp=%.4f sl=%.4f top_n=%d max_pos=%d "
        "kelly=%.4f size_cap=%.4f max_dd=%.4f regime=%s risk_off=%s size_mult=%.3f score_raise=%.3f",
        run_id,
        total_runs,
        replay_args.prediction_source,
        replay_args.predictions_table,
        cfg.prob_threshold,
        cfg.take_profit_pct,
        cfg.stop_loss_pct,
        cfg.top_n,
        cfg.max_positions,
        cfg.kelly_fraction_scale,
        cfg.combined_size_cap,
        cfg.max_drawdown_pct,
        args.regime_gating_mode,
        str(cfg.max_market_risk_off_score),
        cfg.risk_off_size_multiplier,
        cfg.risk_off_score_raise,
    )

    predictions_df = replay_mod.load_predictions(conn, replay_args)
    prices_df = replay_mod.load_prices(conn, replay_args)
    context_df = replay_mod.load_context(conn, replay_args)
    predictions_df = replay_mod.enrich_predictions(predictions_df, context_df, replay_args)
    trades_df, equity_df, summary_df = replay_mod.replay_strategy(predictions_df, prices_df, replay_args)
    persist_run_outputs(conn, trades_df, equity_df, summary_df, replay_args, run_id)
    metrics = summary_to_dict(summary_df)

    trades_metric = safe_metric(metrics, "trades")
    wins_metric = safe_metric(metrics, "wins")
    losses_metric = safe_metric(metrics, "losses")
    total_pnl_metric = safe_metric(metrics, "total_pnl_usd")
    max_drawdown_metric = safe_metric(metrics, "max_drawdown_pct")

    return SweepRow(
        run_id=run_id,
        prediction_source=safe_text(metrics, "prediction_source") or replay_args.prediction_source,
        predictions_table=replay_args.predictions_table,
        regime_gating_mode=args.regime_gating_mode,
        enable_regime_gating=int(bool(args.enable_regime_gating)),
        weight_6h=cfg.weight_6h if replay_args.prediction_source == "ensemble" else None,
        weight_24h=cfg.weight_24h if replay_args.prediction_source == "ensemble" else None,
        weight_72h=cfg.weight_72h if replay_args.prediction_source == "ensemble" else None,
        prob_threshold=cfg.prob_threshold,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
        top_n=cfg.top_n,
        max_positions=cfg.max_positions,
        enable_dynamic_sizing=int(bool(args.enable_dynamic_sizing)),
        enable_kelly_sizing=int(bool(args.enable_kelly_sizing)),
        enable_dynamic_max_positions=int(bool(args.enable_dynamic_max_positions)),
        enable_risk_manager=int(bool(args.enable_risk_manager)),
        enable_drawdown_scaling=int(bool(args.enable_drawdown_scaling)),
        kelly_fraction_scale=cfg.kelly_fraction_scale,
        combined_size_cap=cfg.combined_size_cap,
        max_drawdown_pct=cfg.max_drawdown_pct,
        drawdown_full_size_pct=cfg.drawdown_full_size_pct,
        drawdown_half_size_pct=cfg.drawdown_half_size_pct,
        drawdown_quarter_size_pct=cfg.drawdown_quarter_size_pct,
        drawdown_half_size_multiplier=cfg.drawdown_half_size_multiplier,
        drawdown_quarter_size_multiplier=cfg.drawdown_quarter_size_multiplier,
        max_market_risk_off_score=cfg.max_market_risk_off_score,
        max_market_dispersion_24h=cfg.max_market_dispersion_24h,
        min_market_trend_strength_24h=cfg.min_market_trend_strength_24h,
        risk_off_size_multiplier=cfg.risk_off_size_multiplier if args.enable_regime_gating else None,
        risk_off_score_raise=cfg.risk_off_score_raise if args.enable_regime_gating else None,
        ranking_mode=safe_text(metrics, "ranking_mode") or args.ranking_mode,
        resolved_prob_column=safe_text(metrics, "resolved_prob_column"),
        resolved_score_column=safe_text(metrics, "resolved_score_column"),
        total_pnl_usd=total_pnl_metric,
        total_return_pct=safe_metric(metrics, "total_return_pct"),
        max_drawdown_pct_realized=max_drawdown_metric,
        trades=int(trades_metric) if trades_metric is not None else None,
        wins=int(wins_metric) if wins_metric is not None else None,
        losses=int(losses_metric) if losses_metric is not None else None,
        win_rate=safe_metric(metrics, "win_rate"),
        avg_pnl_usd=safe_metric(metrics, "avg_pnl_usd"),
        avg_return_pct=safe_metric(metrics, "avg_return_pct"),
        avg_hold_hours=safe_metric(metrics, "avg_hold_hours"),
        avg_notional_usd=safe_metric(metrics, "avg_notional_usd"),
        avg_prob_size_multiplier=safe_metric(metrics, "avg_prob_size_multiplier"),
        avg_vol_size_multiplier=safe_metric(metrics, "avg_vol_size_multiplier"),
        avg_kelly_fraction=safe_metric(metrics, "avg_kelly_fraction"),
        avg_kelly_multiplier=safe_metric(metrics, "avg_kelly_multiplier"),
        avg_total_size_multiplier=safe_metric(metrics, "avg_total_size_multiplier"),
        trades_blocked_by_risk=safe_metric(metrics, "trades_blocked_by_risk"),
        regime_gate_blocks=safe_metric(metrics, "regime_gate_blocks"),
        regime_scale_events=safe_metric(metrics, "regime_scale_events"),
        regime_score_raise_events=safe_metric(metrics, "regime_score_raise_events"),
        drawdown_scaling_half_count=safe_metric(metrics, "drawdown_scaling_half_count"),
        drawdown_scaling_quarter_count=safe_metric(metrics, "drawdown_scaling_quarter_count"),
        drawdown_scaling_stop_count=safe_metric(metrics, "drawdown_scaling_stop_count"),
        pnl_to_abs_drawdown=derive_pnl_to_drawdown(total_pnl_metric, max_drawdown_metric),
    )


def sort_results(results_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, asc in [
        ("total_pnl_usd", False),
        ("pnl_to_abs_drawdown", False),
        ("win_rate", False),
        ("trades", False),
        ("run_id", True),
    ]:
        if col in results_df.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        return results_df.reset_index(drop=True)
    return results_df.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    configs = build_grid(args)
    if not configs:
        raise ValueError("No sweep configurations were generated")
    if len(configs) > MAX_SWEEP_RUNS:
        raise ValueError(
            f"Sweep generated {len(configs)} configurations, exceeding the hard limit of {MAX_SWEEP_RUNS}."
        )

    LOGGER.info("Opening SQLite database: %s", args.db_path)
    LOGGER.info("Generated %d sweep configurations", len(configs))
    conn = sqlite3.connect(args.db_path)
    try:
        total_runs = len(configs)
        rows = [run_one(args, conn, idx, total_runs, cfg) for idx, cfg in enumerate(configs, start=1)]
        results_df = pd.DataFrame([asdict(r) for r in rows])
        results_df = sort_results(results_df)
        results_df.to_sql(args.results_table, conn, if_exists=args.if_exists, index=False)
        LOGGER.info("Wrote sweep results -> %s (%d rows)", args.results_table, len(results_df))
        top_cols = [
            "run_id", "prediction_source", "predictions_table", "regime_gating_mode", "enable_regime_gating",
            "resolved_prob_column", "resolved_score_column", "ranking_mode",
            "weight_6h", "weight_24h", "weight_72h",
            "prob_threshold", "take_profit_pct", "stop_loss_pct", "top_n", "max_positions",
            "kelly_fraction_scale", "combined_size_cap", "max_drawdown_pct",
            "max_market_risk_off_score", "risk_off_size_multiplier", "risk_off_score_raise",
            "total_pnl_usd", "max_drawdown_pct_realized", "win_rate", "trades",
            "regime_gate_blocks", "regime_scale_events", "regime_score_raise_events",
            "pnl_to_abs_drawdown",
        ]
        existing_top_cols = [col for col in top_cols if col in results_df.columns]
        if existing_top_cols:
            LOGGER.info("Top sweep results:\n%s", results_df[existing_top_cols].head(10).to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
