#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tradarbot.research.spikes.source_registry import SourceRegistry
except Exception:
    SourceRegistry = None

LOGGER = logging.getLogger("build_spike_base_dataset")


@dataclass
class SymbolQuality:
    symbol: str
    raw_rows: int
    resampled_rows: int
    kept_rows: int
    span_hours: float
    nonzero_volume_rate: float
    dropped_reason: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spike base dataset from candle history.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--source-table", default="candles")
    parser.add_argument("--output-table", default="spike_base_rows")
    parser.add_argument("--summary-table", default="spike_base_summary")
    parser.add_argument("--source-interval-s", type=int, default=None)
    parser.add_argument("--target-interval-s", type=int, default=3600)
    parser.add_argument("--min-bucket-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--drop-partial-buckets", action="store_true")
    parser.add_argument("--min-symbol-rows", type=int, default=168)
    parser.add_argument("--min-symbol-span-hours", type=float, default=168.0)
    parser.add_argument("--min-nonzero-volume-rate", type=float, default=0.5)
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="fail")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def load_sqlite_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def normalize_candle_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "symbol", "interval_s", "timestamp_ms",
                "price_open", "price_high", "price_low", "price_close", "volume_base",
            ]
        )

    rename_map: Dict[str, str] = {}
    cols = set(df.columns)

    if "symbol" in cols:
        rename_map["symbol"] = "symbol"
    elif "asset_id" in cols:
        rename_map["asset_id"] = "symbol"

    if "interval_s" in cols:
        rename_map["interval_s"] = "interval_s"

    if "ts_ms" in cols:
        rename_map["ts_ms"] = "timestamp_ms"
    elif "timestamp_ms" in cols:
        rename_map["timestamp_ms"] = "timestamp_ms"
    elif "first_ts_ms" in cols:
        rename_map["first_ts_ms"] = "timestamp_ms"

    if "open" in cols:
        rename_map["open"] = "price_open"
    elif "price_open" in cols:
        rename_map["price_open"] = "price_open"

    if "high" in cols:
        rename_map["high"] = "price_high"
    elif "price_high" in cols:
        rename_map["price_high"] = "price_high"

    if "low" in cols:
        rename_map["low"] = "price_low"
    elif "price_low" in cols:
        rename_map["price_low"] = "price_low"

    if "close" in cols:
        rename_map["close"] = "price_close"
    elif "price_close" in cols:
        rename_map["price_close"] = "price_close"

    if "volume" in cols:
        rename_map["volume"] = "volume_base"
    elif "volume_base" in cols:
        rename_map["volume_base"] = "volume_base"

    out = df.rename(columns=rename_map).copy()
    required = ["symbol", "timestamp_ms", "price_open", "price_high", "price_low", "price_close", "volume_base"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required candle columns after normalization: {missing}")

    if "interval_s" not in out.columns:
        out["interval_s"] = np.nan

    out = out[
        [
            "symbol", "interval_s", "timestamp_ms",
            "price_open", "price_high", "price_low", "price_close", "volume_base",
        ]
    ].copy()

    out["symbol"] = out["symbol"].astype(str)
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce").astype("Int64")
    out["interval_s"] = pd.to_numeric(out["interval_s"], errors="coerce")
    for col in ["price_open", "price_high", "price_low", "price_close", "volume_base"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["symbol", "timestamp_ms", "price_open", "price_high", "price_low", "price_close"])
    out["volume_base"] = out["volume_base"].fillna(0.0)
    return out.sort_values(["symbol", "timestamp_ms"]).reset_index(drop=True)


def infer_source_interval_s(df: pd.DataFrame, override: Optional[int]) -> int:
    if override is not None:
        return int(override)

    non_null = df["interval_s"].dropna()
    if not non_null.empty:
        return int(non_null.mode().iloc[0])

    ts = df["timestamp_ms"].dropna().astype("int64")
    if len(ts) >= 2:
        diffs = pd.Series(ts).sort_values().diff().dropna()
        positive = diffs[diffs > 0]
        if not positive.empty:
            return max(1, int(round(float(positive.median()) / 1000.0)))
    return 1


def compute_bucket_features(group: pd.DataFrame, target_interval_s: int, source_interval_s: int) -> pd.DataFrame:
    g = group.sort_values("timestamp_ms").copy()
    g["bucket_start_ms"] = (g["timestamp_ms"] // (target_interval_s * 1000)) * (target_interval_s * 1000)

    outputs: List[dict] = []
    grouped = g.groupby("bucket_start_ms", sort=True)
    expected_points = max(1.0, float(target_interval_s) / float(max(1, source_interval_s)))

    for bucket_start_ms, bucket_df in grouped:
        bucket_df = bucket_df.sort_values("timestamp_ms")
        coverage_ratio = min(1.0, bucket_df["timestamp_ms"].nunique() / expected_points)

        outputs.append(
            {
                "asset_id": str(bucket_df["symbol"].iloc[0]),
                "symbol": str(bucket_df["symbol"].iloc[0]),
                "timestamp": pd.to_datetime(bucket_start_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timestamp_ms": int(bucket_start_ms),
                "interval_s": int(target_interval_s),
                "price_open": float(bucket_df["price_open"].iloc[0]),
                "price_high": float(bucket_df["price_high"].max()),
                "price_low": float(bucket_df["price_low"].min()),
                "price_close": float(bucket_df["price_close"].iloc[-1]),
                "volume_base": float(bucket_df["volume_base"].sum()),
                "bucket_coverage_ratio": float(coverage_ratio),
                "source_row_count": int(len(bucket_df)),
                "first_ts_ms": int(bucket_df["timestamp_ms"].iloc[0]),
                "last_ts_ms": int(bucket_df["timestamp_ms"].iloc[-1]),
            }
        )

    return pd.DataFrame(outputs)


def resample_all_symbols(df: pd.DataFrame, target_interval_s: int, source_interval_s: int) -> pd.DataFrame:
    parts = [compute_bucket_features(group, target_interval_s, source_interval_s) for _, group in df.groupby("symbol", sort=True)]
    if not parts:
        return pd.DataFrame(
            columns=[
                "asset_id", "symbol", "timestamp", "timestamp_ms", "interval_s",
                "price_open", "price_high", "price_low", "price_close", "volume_base",
                "bucket_coverage_ratio", "source_row_count", "first_ts_ms", "last_ts_ms",
            ]
        )
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "timestamp_ms"]).reset_index(drop=True)


def apply_symbol_quality_filters(
    df: pd.DataFrame,
    min_symbol_rows: int,
    min_symbol_span_hours: float,
    min_nonzero_volume_rate: float,
):
    qualities: List[SymbolQuality] = []
    kept_parts: List[pd.DataFrame] = []

    for symbol, group in df.groupby("symbol", sort=True):
        group = group.sort_values("timestamp_ms").copy()
        row_count = int(len(group))
        span_hours = 0.0
        if row_count >= 2:
            span_hours = (int(group["timestamp_ms"].iloc[-1]) - int(group["timestamp_ms"].iloc[0])) / 3_600_000.0
        nonzero_volume_rate = float((group["volume_base"] > 0).mean()) if row_count else 0.0

        dropped_reason = None
        if row_count < min_symbol_rows:
            dropped_reason = f"too_few_rows<{min_symbol_rows}"
        elif span_hours < float(min_symbol_span_hours):
            dropped_reason = f"span_hours<{min_symbol_span_hours}"
        elif nonzero_volume_rate < float(min_nonzero_volume_rate):
            dropped_reason = f"nonzero_volume_rate<{min_nonzero_volume_rate}"

        qualities.append(
            SymbolQuality(
                symbol=symbol,
                raw_rows=row_count,
                resampled_rows=row_count,
                kept_rows=row_count if dropped_reason is None else 0,
                span_hours=span_hours,
                nonzero_volume_rate=nonzero_volume_rate,
                dropped_reason=dropped_reason,
            )
        )

        if dropped_reason is None:
            kept_parts.append(group)

    kept_df = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[0:0].copy()
    return kept_df, qualities


def build_summary(raw_rows, resampled_rows, bucket_eval_rows, kept_df, qualities, args):
    rows: List[dict] = [
        {"symbol": "__GLOBAL__", "metric_name": "raw_input_rows", "metric_value": float(raw_rows), "notes": "Rows loaded from the source table before resampling."},
        {"symbol": "__GLOBAL__", "metric_name": "resampled_rows", "metric_value": float(resampled_rows), "notes": "Rows after interval normalization/resampling."},
        {"symbol": "__GLOBAL__", "metric_name": "bucket_eval_rows", "metric_value": float(bucket_eval_rows), "notes": "Rows considered before optional bucket-coverage dropping."},
        {"symbol": "__GLOBAL__", "metric_name": "kept_base_rows", "metric_value": float(len(kept_df)), "notes": "Rows retained after bucket and symbol quality gates."},
        {"symbol": "__GLOBAL__", "metric_name": "symbols_after_filter", "metric_value": float(kept_df['symbol'].nunique() if not kept_df.empty else 0), "notes": "Distinct symbols retained after quality gating."},
        {"symbol": "__GLOBAL__", "metric_name": "min_bucket_coverage_ratio", "metric_value": float(args.min_bucket_coverage_ratio), "notes": "Configured minimum bucket coverage ratio."},
        {"symbol": "__GLOBAL__", "metric_name": "drop_partial_buckets", "metric_value": float(1 if args.drop_partial_buckets else 0), "notes": "Whether partial buckets were dropped."},
        {"symbol": "__GLOBAL__", "metric_name": "min_symbol_rows", "metric_value": float(args.min_symbol_rows), "notes": "Configured minimum rows per symbol."},
        {"symbol": "__GLOBAL__", "metric_name": "min_symbol_span_hours", "metric_value": float(args.min_symbol_span_hours), "notes": "Configured minimum time span per symbol."},
        {"symbol": "__GLOBAL__", "metric_name": "min_nonzero_volume_rate", "metric_value": float(args.min_nonzero_volume_rate), "notes": "Configured minimum nonzero volume rate per symbol."},
    ]

    for q in qualities:
        rows.extend(
            [
                {"symbol": q.symbol, "metric_name": "kept_rows", "metric_value": float(q.kept_rows), "notes": "Rows kept for this symbol after all filters."},
                {"symbol": q.symbol, "metric_name": "span_hours", "metric_value": float(q.span_hours), "notes": "Time span covered by retained/resampled rows."},
                {"symbol": q.symbol, "metric_name": "nonzero_volume_rate", "metric_value": float(q.nonzero_volume_rate), "notes": "Share of rows with volume_base > 0."},
                {"symbol": q.symbol, "metric_name": "dropped_flag", "metric_value": float(1 if q.dropped_reason else 0), "notes": q.dropped_reason or "kept"},
            ]
        )

    return pd.DataFrame(rows, columns=["symbol", "metric_name", "metric_value", "notes"])


def main() -> None:
    args = parse_args()
    configure_logging()

    LOGGER.info("Opening SQLite database: %s", args.db_path)

    with sqlite3.connect(args.db_path) as conn:
        if SourceRegistry is not None:
            try:
                registry = SourceRegistry()
                source = registry.get("sqlite_candles")
                LOGGER.info("Selected market source: %s (%s)", getattr(source, "name", "sqlite_candles"), getattr(source, "description", "SQLite Candles"))
            except Exception:
                LOGGER.info("Selected market source: sqlite_candles (registry unavailable)")
        else:
            LOGGER.info("Selected market source: sqlite_candles (registry import unavailable)")

        if not table_exists(conn, args.source_table):
            raise ValueError(f"Source table does not exist: {args.source_table}")

        raw_df = load_sqlite_table(conn, args.source_table)
        norm_df = normalize_candle_columns(raw_df)

        if norm_df.empty:
            LOGGER.warning("No source candles loaded from table '%s'", args.source_table)
            empty_base = pd.DataFrame(
                columns=[
                    "asset_id", "symbol", "timestamp", "timestamp_ms", "interval_s",
                    "price_open", "price_high", "price_low", "price_close", "volume_base",
                    "bucket_coverage_ratio", "source_row_count", "first_ts_ms", "last_ts_ms",
                ]
            )
            summary_df = build_summary(0, 0, 0, empty_base, [], args)
            empty_base.to_sql(args.output_table, conn, if_exists=args.if_exists, index=False)
            summary_df.to_sql(args.summary_table, conn, if_exists=args.if_exists, index=False)
            LOGGER.info("Done. Output rows=0")
            return

        source_interval_s = infer_source_interval_s(norm_df, args.source_interval_s)
        LOGGER.info("Using source interval_s=%s", source_interval_s)

        resampled_once = resample_all_symbols(norm_df, args.target_interval_s, source_interval_s)
        bucket_eval_rows = len(resampled_once)
        resampled_df = resampled_once.copy()

        if args.drop_partial_buckets:
            resampled_df = resampled_df[resampled_df["bucket_coverage_ratio"] >= float(args.min_bucket_coverage_ratio)].copy()

        kept_df, qualities = apply_symbol_quality_filters(
            resampled_df,
            min_symbol_rows=args.min_symbol_rows,
            min_symbol_span_hours=args.min_symbol_span_hours,
            min_nonzero_volume_rate=args.min_nonzero_volume_rate,
        )
        kept_df = kept_df.sort_values(["symbol", "timestamp_ms"]).reset_index(drop=True)

        summary_df = build_summary(
            raw_rows=len(norm_df),
            resampled_rows=len(resampled_once),
            bucket_eval_rows=bucket_eval_rows,
            kept_df=kept_df,
            qualities=qualities,
            args=args,
        )

        LOGGER.info("Writing base rows: %s rows across %s symbols -> %s", len(kept_df), int(kept_df["symbol"].nunique()) if not kept_df.empty else 0, args.output_table)
        kept_df.to_sql(args.output_table, conn, if_exists=args.if_exists, index=False)
        LOGGER.info("Writing base summary -> %s", args.summary_table)
        summary_df.to_sql(args.summary_table, conn, if_exists=args.if_exists, index=False)
        LOGGER.info("Base summary:\n%s", summary_df.to_string(index=False))
        LOGGER.info("Done. Output rows=%s", len(kept_df))


if __name__ == "__main__":
    main()
