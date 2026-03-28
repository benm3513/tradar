#!/usr/bin/env python3
"""Build time-aware spike labels from a historical base dataset.

This script reads an asset/timestamp price table from SQLite, computes forward
elapsed-time spike targets, and writes an output label table. It now supports:
- looser default thresholds for better recall
- explicit per-horizon threshold overrides
- percentile-based spike definitions per asset stream

Examples
--------
Absolute threshold mode:
./.venv/bin/python scripts/build_spike_labels.py \
  --db-path tradarbot.db \
  --source-table spike_base_rows \
  --output-table spike_labeled_rows

Percentile mode (top 8% spikes per symbol on 24h target):
./.venv/bin/python scripts/build_spike_labels.py \
  --db-path tradarbot.db \
  --source-table spike_base_rows \
  --output-table spike_labeled_rows \
  --label-mode percentile \
  --percentile-threshold-24h 0.92
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd

from tradarbot.research.spikes.labels import (
    DEFAULT_LABEL_CONFIGS,
    SpikeLabelConfig,
    merge_labels_onto_frame,
)

LOGGER = logging.getLogger("build_spike_labels")

DEFAULT_ASSET_KEY_COLUMNS = ["asset_id", "symbol", "exchange"]
DEFAULT_SUMMARY_TABLE = "spike_label_summary"
SUPPORTED_HORIZONS = ("6h", "24h", "72h", "7d")


class BuildSpikeLabelsError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--source-table", required=True, help="Input SQLite table containing the base asset-time rows.")
    parser.add_argument("--output-table", required=True, help="Output SQLite table name for labeled rows.")
    parser.add_argument("--price-column", default="price_close", help="Name of the price column in the source table.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Name of the timestamp column in the source table.")
    parser.add_argument(
        "--asset-key-columns",
        nargs="+",
        default=DEFAULT_ASSET_KEY_COLUMNS,
        help="Columns used to identify an asset stream. Default: asset_id symbol exchange",
    )
    parser.add_argument("--summary-table", default=DEFAULT_SUMMARY_TABLE)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--if-exists", choices=("fail", "replace", "append"), default="replace")

    parser.add_argument("--label-mode", choices=("absolute", "percentile"), default="absolute")
    parser.add_argument("--tradeable-entry-to-peak-ratio", type=float, default=0.92)
    parser.add_argument("--tradeable-max-pre-peak-drawdown", type=float, default=0.45)

    for horizon in SUPPORTED_HORIZONS:
        key = horizon.replace("d", "d").replace("h", "h")
        parser.add_argument(f"--spike-threshold-{horizon}", type=float, default=None)
        parser.add_argument(f"--tradeable-min-return-{horizon}", type=float, default=None)
        parser.add_argument(f"--percentile-threshold-{horizon}", type=float, default=None)

    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1"
    row = connection.execute(query, (table_name,)).fetchone()
    return row is not None


def load_source_frame(connection: sqlite3.Connection, source_table: str) -> pd.DataFrame:
    if not _table_exists(connection, source_table):
        raise BuildSpikeLabelsError(f"source table does not exist: {source_table}")
    query = f'SELECT * FROM "{source_table}"'
    frame = pd.read_sql_query(query, connection)
    if frame.empty:
        raise BuildSpikeLabelsError(f"source table is empty: {source_table}")
    return frame


def resolve_asset_key_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    present = [col for col in candidates if col in frame.columns]
    if present:
        return present
    raise BuildSpikeLabelsError(
        "No asset key columns were found. Provide at least one identifier such as asset_id, symbol, or exchange."
    )


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    asset_key_columns: Sequence[str],
    timestamp_column: str,
    price_column: str,
) -> None:
    required = set(asset_key_columns) | {timestamp_column, price_column}
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise BuildSpikeLabelsError(f"source table missing required columns: {missing}")

    if frame[price_column].isna().any():
        null_rows = int(frame[price_column].isna().sum())
        raise BuildSpikeLabelsError(f"price column contains {null_rows} null rows")

    non_positive_mask = frame[price_column] <= 0
    if bool(non_positive_mask.any()):
        bad_rows = int(non_positive_mask.sum())
        raise BuildSpikeLabelsError(f"price column contains {bad_rows} non-positive rows")

    duplicate_count = int(frame.duplicated(subset=[*asset_key_columns, timestamp_column]).sum())
    if duplicate_count > 0:
        raise BuildSpikeLabelsError(
            f"source table contains duplicate asset/timestamp rows: {duplicate_count} duplicates"
        )


def sort_frame(frame: pd.DataFrame, *, asset_key_columns: Sequence[str], timestamp_column: str) -> pd.DataFrame:
    sorted_frame = frame.copy()
    sorted_frame[timestamp_column] = pd.to_datetime(sorted_frame[timestamp_column], utc=True, errors="raise")
    sorted_frame = sorted_frame.sort_values([*asset_key_columns, timestamp_column]).reset_index(drop=True)
    return sorted_frame


def build_label_configs(args: argparse.Namespace) -> dict[str, SpikeLabelConfig]:
    configs: dict[str, SpikeLabelConfig] = {}
    for horizon, base_cfg in DEFAULT_LABEL_CONFIGS.items():
        spike_threshold = getattr(args, f"spike_threshold_{horizon}", None)
        tradeable_min_return = getattr(args, f"tradeable_min_return_{horizon}", None)
        configs[horizon] = SpikeLabelConfig(
            horizon=base_cfg.horizon,
            spike_threshold_return=float(base_cfg.spike_threshold_return if spike_threshold is None else spike_threshold),
            tradeable_min_return=float(base_cfg.tradeable_min_return if tradeable_min_return is None else tradeable_min_return),
            tradeable_entry_to_peak_ratio=float(args.tradeable_entry_to_peak_ratio),
            tradeable_max_pre_peak_drawdown=float(args.tradeable_max_pre_peak_drawdown),
        )
    return configs


def _resolve_label_threshold(asset_frame: pd.DataFrame, horizon: str, args: argparse.Namespace, configs: dict[str, SpikeLabelConfig]) -> float:
    if args.label_mode == "absolute":
        return float(configs[horizon].spike_threshold_return)

    percentile = getattr(args, f"percentile_threshold_{horizon}", None)
    if percentile is None:
        percentile = 0.92 if horizon == "24h" else 0.90
    percentile = float(percentile)
    if not (0.0 < percentile < 1.0):
        raise BuildSpikeLabelsError(f"percentile threshold for {horizon} must be between 0 and 1")

    target_col = f"target_future_max_return_{horizon}"
    eligible = pd.to_numeric(asset_frame[target_col], errors="coerce").dropna()
    if eligible.empty:
        return float(configs[horizon].spike_threshold_return)
    return float(eligible.quantile(percentile))


def relabel_asset_frame(asset_frame: pd.DataFrame, args: argparse.Namespace, configs: dict[str, SpikeLabelConfig]) -> pd.DataFrame:
    out = asset_frame.copy()
    for horizon in SUPPORTED_HORIZONS:
        cfg = configs[horizon]
        min_return = getattr(args, f"tradeable_min_return_{horizon}", None)
        label_threshold = _resolve_label_threshold(out, horizon, args, configs)
        tradeable_min_return = float(label_threshold if min_return is None else min_return)

        target_return_col = f"target_future_max_return_{horizon}"
        target_peak_col = f"target_future_peak_price_{horizon}"
        pre_peak_dd_col = f"target_pre_peak_drawdown_{horizon}"
        label_spike_col = f"label_spike_{horizon}"
        label_tradeable_col = f"label_tradeable_pre_spike_{horizon}"
        threshold_col = f"applied_spike_threshold_{horizon}"

        future_return = pd.to_numeric(out[target_return_col], errors="coerce")
        future_peak = pd.to_numeric(out[target_peak_col], errors="coerce")
        pre_peak_drawdown = pd.to_numeric(out[pre_peak_dd_col], errors="coerce")
        current_price = pd.to_numeric(out["price_close"], errors="coerce")

        spike_mask = future_return.notna()
        out[label_spike_col] = pd.Series(pd.NA, index=out.index, dtype="object")
        out.loc[spike_mask, label_spike_col] = (future_return[spike_mask] >= label_threshold).astype(int)

        tradeable_mask = future_return.notna() & future_peak.notna() & pre_peak_drawdown.notna() & current_price.notna()
        early_enough = current_price <= (float(cfg.tradeable_entry_to_peak_ratio) * future_peak)
        return_ok = future_return >= tradeable_min_return
        drawdown_ok = pre_peak_drawdown <= float(cfg.tradeable_max_pre_peak_drawdown)
        tradeable = (early_enough & return_ok & drawdown_ok).astype(int)

        out[label_tradeable_col] = pd.Series(pd.NA, index=out.index, dtype="object")
        out.loc[tradeable_mask, label_tradeable_col] = tradeable[tradeable_mask]
        out[threshold_col] = float(label_threshold)

    return out


def build_labels_for_all_assets(
    frame: pd.DataFrame,
    *,
    asset_key_columns: Sequence[str],
    timestamp_column: str,
    price_column: str,
    configs: dict[str, SpikeLabelConfig],
    args: argparse.Namespace,
) -> pd.DataFrame:
    output_groups: list[pd.DataFrame] = []

    for asset_key, asset_frame in frame.groupby(list(asset_key_columns), sort=False, dropna=False):
        asset_frame = asset_frame.sort_values(timestamp_column).reset_index(drop=True)
        labeled = merge_labels_onto_frame(asset_frame, price_column=price_column, configs=configs)
        labeled = relabel_asset_frame(labeled, args, configs)
        output_groups.append(labeled)

        asset_key_tuple = asset_key if isinstance(asset_key, tuple) else (asset_key,)
        LOGGER.debug("Labeled asset stream %s with %d rows", asset_key_tuple, len(asset_frame))

    if not output_groups:
        raise BuildSpikeLabelsError("No asset groups were produced from the source frame")

    return pd.concat(output_groups, ignore_index=True)


def build_summary_frame(labeled_frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    label_columns = sorted(col for col in labeled_frame.columns if col.startswith("label_"))
    target_columns = sorted(col for col in labeled_frame.columns if col.startswith("target_future_max_return_"))
    applied_threshold_columns = sorted(col for col in labeled_frame.columns if col.startswith("applied_spike_threshold_"))

    for column in label_columns:
        non_null = labeled_frame[column].notna()
        eligible_rows = int(non_null.sum())
        positive_rows = int(pd.to_numeric(labeled_frame.loc[non_null, column], errors="coerce").fillna(0).sum()) if eligible_rows else 0
        positive_rate = (positive_rows / eligible_rows) if eligible_rows else None
        summary_rows.append(
            {
                "metric_type": "label",
                "metric_name": column,
                "eligible_rows": eligible_rows,
                "positive_rows": positive_rows,
                "positive_rate": positive_rate,
                "mean_value": None,
                "median_value": None,
            }
        )

    for column in target_columns + applied_threshold_columns:
        series = pd.to_numeric(labeled_frame[column], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric_type": "target" if column.startswith("target_") else "threshold",
                "metric_name": column,
                "eligible_rows": int(series.shape[0]),
                "positive_rows": None,
                "positive_rate": None,
                "mean_value": float(series.mean()) if not series.empty else None,
                "median_value": float(series.median()) if not series.empty else None,
            }
        )

    return pd.DataFrame(summary_rows)


def write_frame_to_sqlite(connection: sqlite3.Connection, frame: pd.DataFrame, *, table_name: str, if_exists: str) -> None:
    frame_to_write = frame.copy()
    for col in frame_to_write.columns:
        if pd.api.types.is_datetime64_any_dtype(frame_to_write[col]):
            frame_to_write[col] = frame_to_write[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame_to_write.to_sql(table_name, connection, if_exists=if_exists, index=False)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Opening SQLite database: %s", db_path)
    with sqlite3.connect(db_path) as connection:
        source_frame = load_source_frame(connection, args.source_table)
        asset_key_columns = resolve_asset_key_columns(source_frame, args.asset_key_columns)

        validate_source_frame(
            source_frame,
            asset_key_columns=asset_key_columns,
            timestamp_column=args.timestamp_column,
            price_column=args.price_column,
        )

        source_frame = sort_frame(
            source_frame,
            asset_key_columns=asset_key_columns,
            timestamp_column=args.timestamp_column,
        )
        configs = build_label_configs(args)

        LOGGER.info(
            "Building labels for %d rows across %d asset streams (mode=%s)",
            len(source_frame),
            source_frame.groupby(asset_key_columns, dropna=False).ngroups,
            args.label_mode,
        )
        labeled_frame = build_labels_for_all_assets(
            source_frame,
            asset_key_columns=asset_key_columns,
            timestamp_column=args.timestamp_column,
            price_column=args.price_column,
            configs=configs,
            args=args,
        )

        LOGGER.info("Writing labeled rows to table: %s", args.output_table)
        write_frame_to_sqlite(connection, labeled_frame, table_name=args.output_table, if_exists=args.if_exists)

        if not args.skip_summary:
            summary_frame = build_summary_frame(labeled_frame)
            LOGGER.info("Writing label summary to table: %s", args.summary_table)
            write_frame_to_sqlite(connection, summary_frame, table_name=args.summary_table, if_exists="replace")
            LOGGER.info("Label summary:\n%s", summary_frame.to_string(index=False))

        LOGGER.info("Done. Output rows=%d, columns=%d", len(labeled_frame), labeled_frame.shape[1])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildSpikeLabelsError as exc:
        LOGGER.error("Build failed: %s", exc)
        raise SystemExit(2) from exc
