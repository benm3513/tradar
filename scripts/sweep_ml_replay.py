#!/usr/bin/env python3
"""Grid search runner for ML replay strategy.

This script sweeps parameter combinations for ``scripts/replay_ml_strategy.py``,
collects replay outputs, persists per-run artifacts, and writes ranked sweep
results to SQLite.

Phase 4.7 upgrades
------------------
- Supports replay prediction sources:
    - ensemble
    - 6h
    - 24h
    - 72h
    - direct
- Supports ensemble-weight sweeps:
    --weight-6h
    --weight-24h
    --weight-72h
- Optionally rebuilds ensemble predictions before each replay run
- Captures prediction-source and resolved column metadata in sweep results
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

LOGGER = logging.getLogger("sweep_ml_replay")

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


@dataclass
class SweepRow:
    run_id: int
    prediction_source: str
    predictions_table: str
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
    drawdown_scaling_half_count: float | None
    drawdown_scaling_quarter_count: float | None
    drawdown_scaling_stop_count: float | None
    pnl_to_abs_drawdown: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--predictions-table", default="spike_model_predictions_hgb")
    parser.add_argument("--price-table", default="spike_base_rows")
    parser.add_argument("--context-table", default="spike_training_rows")

    parser.add_argument(
        "--prediction-source",
        choices=["ensemble", "6h", "24h", "72h", "direct"],
        default="direct",
    )
    parser.add_argument("--score-column", default=None)

    parser.add_argument("--prob-thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--take-profit-pcts", nargs="+", type=float, required=True)
    parser.add_argument("--stop-loss-pcts", nargs="+", type=float, required=True)
    parser.add_argument("--top-n-values", nargs="+", type=int, default=[1])
    parser.add_argument("--max-positions-values", nargs="+", type=int, default=[1])
    parser.add_argument("--kelly-fraction-scales", nargs="+", type=float, default=[0.25])
    parser.add_argument("--combined-size-caps", nargs="+", type=float, default=[1.6])

    parser.add_argument("--enable-dynamic-sizing", action="store_true")
    parser.add_argument("--enable-kelly-sizing", action="store_true")

    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--price-column", default="price_close")
    parser.add_argument("--prob-column", default="pred_prob")
    parser.add_argument("--model-name-column", default="model_name")
    parser.add_argument("--rolling-volatility-column", default="rolling_volatility_24h")
    parser.add_argument("--time-to-peak-seconds-column", default="target_time_to_peak_seconds_24h")

    parser.add_argument("--min-prob-percentile", type=float, default=0.0)
    parser.add_argument("--min-rolling-volatility-24h", type=float, default=None)
    parser.add_argument("--max-predicted-time-to-peak-hours", type=float, default=None)
    parser.add_argument("--notional-per-trade", type=float, default=1000.0)
    parser.add_argument("--min-notional-per-trade", type=float, default=400.0)
    parser.add_argument("--initial-cash", type=float, default=100000.0)

    parser.add_argument("--ranking-mode", choices=["probability", "composite"], default="probability")
    parser.add_argument("--prob-zscore-weight", type=float, default=1.0)
    parser.add_argument("--percentile-weight", type=float, default=0.0)
    parser.add_argument("--volatility-weight", type=float, default=0.0)
    parser.add_argument("--time-to-peak-weight", type=float, default=0.0)
    parser.add_argument("--rank-score-min", type=float, default=None)

    parser.add_argument("--enable-dynamic-max-positions", action="store_true")
    parser.add_argument("--min-dynamic-max-positions", type=int, default=1)
    parser.add_argument("--dynamic-position-score-threshold", type=float, default=0.0)

    parser.add_argument("--prob-size-cap", type=float, default=2.0)
    parser.add_argument("--vol-reference", type=float, default=0.006)
    parser.add_argument("--vol_size_floor", type=float, default=0.75)
    parser.add_argument("--vol-size-floor", dest="vol_size_floor", type=float)
    parser.add_argument("--vol_size_cap", type=float, default=1.25)
    parser.add_argument("--vol-size-cap", dest="vol_size_cap", type=float)
    parser.add_argument(
        "--kelly-probability-mode",
        choices=["raw", "threshold_relative"],
        default="threshold_relative",
    )
    parser.add_argument("--kelly-size-cap", type=float, default=1.5)
    parser.add_argument("--max-hold-hours", type=float, default=24.0)

    parser.add_argument("--trailing-stop-pct", type=float, default=None)
    parser.add_argument("--trailing-stop-activation-pct", type=float, default=None)
    parser.add_argument("--partial-take-profit-pct", type=float, default=None)
    parser.add_argument("--partial-take-profit-fraction", type=float, default=0.5)
    parser.add_argument("--time-stop-hours", type=float, default=None)
    parser.add_argument("--time-stop-min-return-pct", type=float, default=0.0)

    parser.add_argument("--enable-risk-manager", action="store_true")
    parser.add_argument("--max-daily-loss-usd", type=float, default=None)
    parser.add_argument("--max-total-exposure-usd", type=float, default=None)
    parser.add_argument("--max-total-exposure-pct", type=float, default=None)
    parser.add_argument("--max-exposure-per-symbol-usd", type=float, default=None)
    parser.add_argument("--cooldown-minutes-per-symbol", type=float, default=0.0)

    parser.add_argument("--enable-drawdown-scaling", action="store_true")
    parser.add_argument("--max-drawdown-pcts", nargs="+", type=float, default=[0.08])
    parser.add_argument("--drawdown-full-size-pcts", nargs="+", type=float, default=[0.04])
    parser.add_argument("--drawdown-half-size-pcts", nargs="+", type=float, default=[0.06])
    parser.add_argument("--drawdown-quarter-size-pcts", nargs="+", type=float, default=[0.08])
    parser.add_argument("--drawdown-half-size-multipliers", nargs="+", type=float, default=[0.50])
    parser.add_argument("--drawdown-quarter-size-multipliers", nargs="+", type=float, default=[0.25])

    parser.add_argument(
        "--weight-6h",
        nargs="+",
        type=float,
        default=[0.3],
        help="Ensemble sweep weights for 6h model.",
    )
    parser.add_argument(
        "--weight-24h",
        nargs="+",
        type=float,
        default=[0.5],
        help="Ensemble sweep weights for 24h model.",
    )
    parser.add_argument(
        "--weight-72h",
        nargs="+",
        type=float,
        default=[0.2],
        help="Ensemble sweep weights for 72h model.",
    )
    parser.add_argument(
        "--rebuild-ensemble-per-run",
        action="store_true",
        help="Recompute ensemble table before each run using the current weight combination.",
    )
    parser.add_argument(
        "--ensemble-output-table",
        default="spike_model_predictions_ensemble",
        help="Destination table for rebuilt ensemble predictions.",
    )
    parser.add_argument(
        "--ensemble-table-6h",
        default="spike_model_predictions_6h",
        help="6h prediction table used by the ensemble builder.",
    )
    parser.add_argument(
        "--ensemble-table-24h",
        default="spike_model_predictions_24h",
        help="24h prediction table used by the ensemble builder.",
    )
    parser.add_argument(
        "--ensemble-table-72h",
        default="spike_model_predictions_72h",
        help="72h prediction table used by the ensemble builder.",
    )
    parser.add_argument(
        "--ensemble-agreement-threshold",
        type=float,
        default=0.8,
        help="Agreement threshold passed to the ensemble builder.",
    )
    parser.add_argument(
        "--ensemble-agreement-boost",
        type=float,
        default=0.05,
        help="Agreement boost passed to the ensemble builder.",
    )

    parser.add_argument("--results-table", default="ml_replay_sweep_results")
    parser.add_argument(
        "--summary-table-prefix",
        default="ml_replay_sweep_summary",
        help="Prefix for per-run summary tables. Final name will be '<prefix>_<run_id>'.",
    )
    parser.add_argument(
        "--trades-table-prefix",
        default="ml_replay_sweep_trades",
        help="Prefix for per-run trade tables. Final name will be '<prefix>_<run_id>'.",
    )
    parser.add_argument(
        "--equity-table-prefix",
        default="ml_replay_sweep_equity",
        help="Prefix for per-run equity tables. Final name will be '<prefix>_<run_id>'.",
    )
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="replace")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _weights_sum_to_one(weight_6h: float, weight_24h: float, weight_72h: float, tol: float = 1e-9) -> bool:
    return abs((weight_6h + weight_24h + weight_72h) - 1.0) <= tol


def build_grid(args: argparse.Namespace) -> list[SweepConfig]:
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
    ):
        cfg = SweepConfig(*values)
        if not (
            cfg.drawdown_full_size_pct <= cfg.drawdown_half_size_pct
            <= cfg.drawdown_quarter_size_pct
            <= cfg.max_drawdown_pct
        ):
            continue
        if args.prediction_source == "ensemble" and not _weights_sum_to_one(
            cfg.weight_6h,
            cfg.weight_24h,
            cfg.weight_72h,
        ):
            continue
        configs.append(cfg)
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


def build_ensemble_predictions(
    *,
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    cfg: SweepConfig,
) -> str:
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
        trades_table=per_run_table_name(args.trades_table_prefix, run_id),
        equity_table=per_run_table_name(args.equity_table_prefix, run_id),
        summary_table=per_run_table_name(args.summary_table_prefix, run_id),
        if_exists="replace",
        log_level=args.log_level,
    )


def persist_run_outputs(
    conn: sqlite3.Connection,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    replay_args: argparse.Namespace,
    run_id: int,
) -> None:
    trades_out = trades_df.copy()
    equity_out = equity_df.copy()
    summary_out = summary_df.copy()

    trades_out["run_id"] = run_id
    equity_out["run_id"] = run_id
    summary_out["run_id"] = run_id

    trades_out.to_sql(replay_args.trades_table, conn, if_exists="replace", index=False)
    equity_out.to_sql(replay_args.equity_table, conn, if_exists="replace", index=False)
    summary_out.to_sql(replay_args.summary_table, conn, if_exists="replace", index=False)


def run_one(
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    run_id: int,
    cfg: SweepConfig,
) -> SweepRow:
    replay_args = make_replay_args(args, cfg, run_id)

    if args.prediction_source == "ensemble" and args.rebuild_ensemble_per_run:
        ensemble_table = build_ensemble_predictions(conn=conn, args=args, cfg=cfg)
        replay_args.predictions_table = ensemble_table

    LOGGER.info(
        "Sweep run %d: source=%s preds=%s prob=%.4f tp=%.4f sl=%.4f top_n=%d max_pos=%d "
        "kelly=%.4f size_cap=%.4f max_dd=%.4f dd_full=%.4f dd_half=%.4f dd_qtr=%.4f "
        "half_mult=%.2f qtr_mult=%.2f w6=%.3f w24=%.3f w72=%.3f",
        run_id,
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
        cfg.drawdown_full_size_pct,
        cfg.drawdown_half_size_pct,
        cfg.drawdown_quarter_size_pct,
        cfg.drawdown_half_size_multiplier,
        cfg.drawdown_quarter_size_multiplier,
        cfg.weight_6h,
        cfg.weight_24h,
        cfg.weight_72h,
    )

    predictions_df = replay_mod.load_predictions(conn, replay_args)
    prices_df = replay_mod.load_prices(conn, replay_args)
    context_df = replay_mod.load_context(conn, replay_args)
    predictions_df = replay_mod.enrich_predictions(predictions_df, context_df, replay_args)

    trades_df, equity_df, summary_df = replay_mod.replay_strategy(
        predictions_df,
        prices_df,
        replay_args,
    )

    persist_run_outputs(conn, trades_df, equity_df, summary_df, replay_args, run_id)
    metrics = summary_to_dict(summary_df)

    trades_metric = safe_metric(metrics, "trades")
    wins_metric = safe_metric(metrics, "wins")
    losses_metric = safe_metric(metrics, "losses")
    total_pnl_metric = safe_metric(metrics, "total_pnl_usd")
    max_drawdown_metric = safe_metric(metrics, "max_drawdown_pct")

    row = SweepRow(
        run_id=run_id,
        prediction_source=safe_text(metrics, "prediction_source") or replay_args.prediction_source,
        predictions_table=replay_args.predictions_table,
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
        drawdown_scaling_half_count=safe_metric(metrics, "drawdown_scaling_half_count"),
        drawdown_scaling_quarter_count=safe_metric(metrics, "drawdown_scaling_quarter_count"),
        drawdown_scaling_stop_count=safe_metric(metrics, "drawdown_scaling_stop_count"),
        pnl_to_abs_drawdown=derive_pnl_to_drawdown(total_pnl_metric, max_drawdown_metric),
    )
    return row


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

    LOGGER.info("Opening SQLite database: %s", args.db_path)
    LOGGER.info("Generated %d sweep configurations", len(configs))
    conn = sqlite3.connect(args.db_path)
    try:
        rows = [run_one(args, conn, idx, cfg) for idx, cfg in enumerate(configs, start=1)]
        results_df = pd.DataFrame([asdict(r) for r in rows])
        results_df = sort_results(results_df)
        results_df.to_sql(args.results_table, conn, if_exists=args.if_exists, index=False)
        LOGGER.info("Wrote sweep results -> %s (%d rows)", args.results_table, len(results_df))

        top_cols = [
            "run_id",
            "prediction_source",
            "predictions_table",
            "resolved_prob_column",
            "resolved_score_column",
            "ranking_mode",
            "weight_6h",
            "weight_24h",
            "weight_72h",
            "prob_threshold",
            "take_profit_pct",
            "stop_loss_pct",
            "top_n",
            "max_positions",
            "kelly_fraction_scale",
            "combined_size_cap",
            "max_drawdown_pct",
            "drawdown_full_size_pct",
            "drawdown_half_size_pct",
            "drawdown_quarter_size_pct",
            "drawdown_half_size_multiplier",
            "drawdown_quarter_size_multiplier",
            "total_pnl_usd",
            "max_drawdown_pct_realized",
            "win_rate",
            "trades",
            "drawdown_scaling_half_count",
            "drawdown_scaling_quarter_count",
            "drawdown_scaling_stop_count",
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
