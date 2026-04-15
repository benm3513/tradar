#!/usr/bin/env python3
"""Export a hard-gated spike training set for Phase 4 research.

This script reads the enriched feature table and exports only rows that pass the
current Phase 4 quality gates. The intent is to create a training-ready subset
that is honest about sparse history, missing forward labels, and minimum
coverage requirements.

Phase 4.7 upgrades
------------------
- Supports upgraded multi-horizon label schema
- Preserves stable alias labels:
    - spike_6h_label
    - spike_24h_label
    - spike_72h_label
    - tradeable_pre_spike_6h_label
    - ...
- Adds per-horizon gate diagnostics
- Supports optional horizon-required export behavior
- Keeps backward compatibility with legacy label_spike_* columns

Hard gates
----------
A row is exported only if:
- has_min_history_24h = 1
- is_sparse_stream = 0
- is_tail_unlabelable = 0

Additional safety behavior
--------------------------
- If label payload columns exist, rows must also have has_label_row = 1 when the
  column is present.
- Optionally require presence of specific horizon labels.
- Rows are kept in full unless an explicit column subset is added later.
- A summary table is written to make empty exports or low retention obvious.

Typical usage
-------------
python scripts/export_spike_training_set.py \
    --db-path tradarbot.db \
    --feature-table spike_feature_rows \
    --output-table spike_training_rows \
    --summary-table spike_training_summary

Require 24h labels:
python scripts/export_spike_training_set.py \
    --db-path tradarbot.db \
    --feature-table spike_feature_rows \
    --output-table spike_training_rows_24h \
    --required-horizons 24h
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("export_spike_training_set")

DEFAULT_FEATURE_TABLE = "spike_feature_rows"
DEFAULT_OUTPUT_TABLE = "spike_training_rows"
DEFAULT_SUMMARY_TABLE = "spike_training_summary"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_REQUIRED_FLAG_COLUMNS = [
    "has_min_history_24h",
    "is_sparse_stream",
    "is_tail_unlabelable",
]
OPTIONAL_FLAG_COLUMNS = ["has_label_row"]
SUPPORTED_HORIZONS = ("6h", "24h", "72h", "7d")


class ExportSpikeTrainingSetError(RuntimeError):
    """Raised when the training export cannot proceed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a hard-gated spike training set into SQLite."
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database.")
    parser.add_argument(
        "--feature-table",
        default=DEFAULT_FEATURE_TABLE,
        help=f"Input feature table. Default: {DEFAULT_FEATURE_TABLE}",
    )
    parser.add_argument(
        "--output-table",
        default=DEFAULT_OUTPUT_TABLE,
        help=f"Output training table. Default: {DEFAULT_OUTPUT_TABLE}",
    )
    parser.add_argument(
        "--summary-table",
        default=DEFAULT_SUMMARY_TABLE,
        help=f"Summary table for export diagnostics. Default: {DEFAULT_SUMMARY_TABLE}",
    )
    parser.add_argument(
        "--timestamp-column",
        default=DEFAULT_TIMESTAMP_COLUMN,
        help=f"Timestamp column to normalize on write. Default: {DEFAULT_TIMESTAMP_COLUMN}",
    )
    parser.add_argument(
        "--if-exists",
        choices=("fail", "replace", "append"),
        default="replace",
        help="How to write the exported training table. Default: replace",
    )
    parser.add_argument(
        "--allow-empty-export",
        action="store_true",
        help=(
            "Allow a zero-row training export without treating it as a failure. "
            "Useful during research when quality gates are expected to reject all rows."
        ),
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not write the training export summary table.",
    )
    parser.add_argument(
        "--required-horizons",
        nargs="*",
        default=[],
        choices=SUPPORTED_HORIZONS,
        help=(
            "Optional horizons whose spike labels must be present for a row to be exported. "
            "Example: --required-horizons 24h or --required-horizons 6h 24h 72h"
        ),
    )
    parser.add_argument(
        "--require-tradeable-labels",
        action="store_true",
        help=(
            "When used with --required-horizons, also require tradeable_pre_spike labels "
            "for the same horizons."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def load_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        raise ExportSpikeTrainingSetError(f"table does not exist: {table_name}")
    frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
    if frame.empty:
        raise ExportSpikeTrainingSetError(f"table is empty: {table_name}")
    return frame


def validate_feature_frame(frame: pd.DataFrame) -> None:
    missing = [col for col in DEFAULT_REQUIRED_FLAG_COLUMNS if col not in frame.columns]
    if missing:
        raise ExportSpikeTrainingSetError(
            f"feature table missing required quality columns: {missing}"
        )


def normalize_flag_series(series: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        raise ExportSpikeTrainingSetError(f"quality flag column is fully null: {name}")
    return numeric.fillna(0).astype(int)


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((col for col in candidates if col in frame.columns), None)


def _spike_label_candidates(horizon: str) -> list[str]:
    return [f"spike_{horizon}_label", f"label_spike_{horizon}"]


def _tradeable_label_candidates(horizon: str) -> list[str]:
    return [
        f"tradeable_pre_spike_{horizon}_label",
        f"label_tradeable_pre_spike_{horizon}",
    ]


def _present_any_label_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for horizon in SUPPORTED_HORIZONS:
        cols.extend([c for c in _spike_label_candidates(horizon) if c in frame.columns])
        cols.extend([c for c in _tradeable_label_candidates(horizon) if c in frame.columns])
    return cols


def build_gate_mask(
    frame: pd.DataFrame,
    *,
    required_horizons: Sequence[str],
    require_tradeable_labels: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    gate_details = pd.DataFrame(index=frame.index)

    gate_details["gate_has_min_history_24h"] = (
        normalize_flag_series(frame["has_min_history_24h"], name="has_min_history_24h") == 1
    )
    gate_details["gate_not_sparse_stream"] = (
        normalize_flag_series(frame["is_sparse_stream"], name="is_sparse_stream") == 0
    )
    gate_details["gate_not_tail_unlabelable"] = (
        normalize_flag_series(frame["is_tail_unlabelable"], name="is_tail_unlabelable") == 0
    )

    if "has_label_row" in frame.columns:
        gate_details["gate_has_label_row"] = (
            normalize_flag_series(frame["has_label_row"], name="has_label_row") == 1
        )

    for horizon in required_horizons:
        spike_col = _first_present_column(frame, _spike_label_candidates(horizon))
        if spike_col is None:
            raise ExportSpikeTrainingSetError(
                f"required horizon {horizon} has no spike label column in feature table"
            )
        gate_details[f"gate_has_spike_label_{horizon}"] = frame[spike_col].notna()

        if require_tradeable_labels:
            tradeable_col = _first_present_column(frame, _tradeable_label_candidates(horizon))
            if tradeable_col is None:
                raise ExportSpikeTrainingSetError(
                    f"required horizon {horizon} has no tradeable label column in feature table"
                )
            gate_details[f"gate_has_tradeable_label_{horizon}"] = frame[tradeable_col].notna()

    gate_details["passes_all_training_gates"] = gate_details.all(axis=1)
    return gate_details["passes_all_training_gates"], gate_details


def build_summary_frame(
    feature_frame: pd.DataFrame,
    export_frame: pd.DataFrame,
    gate_details: pd.DataFrame,
) -> pd.DataFrame:
    total_rows = len(feature_frame)
    exported_rows = len(export_frame)
    rejected_rows = total_rows - exported_rows

    summary_rows: list[dict[str, object]] = [
        {
            "metric_type": "count",
            "metric_name": "input_rows",
            "metric_value": float(total_rows),
            "rows": int(total_rows),
            "rate": 1.0 if total_rows else None,
            "notes": "Rows read from spike_feature_rows-like input.",
        },
        {
            "metric_type": "count",
            "metric_name": "exported_rows",
            "metric_value": float(exported_rows),
            "rows": int(exported_rows),
            "rate": float(exported_rows / total_rows) if total_rows else None,
            "notes": "Rows passing all hard quality gates.",
        },
        {
            "metric_type": "count",
            "metric_name": "rejected_rows",
            "metric_value": float(rejected_rows),
            "rows": int(rejected_rows),
            "rate": float(rejected_rows / total_rows) if total_rows else None,
            "notes": "Rows failing at least one hard quality gate.",
        },
    ]

    gate_note_map = {
        "gate_has_min_history_24h": "Requires has_min_history_24h = 1.",
        "gate_not_sparse_stream": "Requires is_sparse_stream = 0.",
        "gate_not_tail_unlabelable": "Requires is_tail_unlabelable = 0.",
        "gate_has_label_row": "Requires has_label_row = 1 when labels are joined.",
        "passes_all_training_gates": "All hard quality gates combined.",
    }

    for horizon in SUPPORTED_HORIZONS:
        gate_note_map[f"gate_has_spike_label_{horizon}"] = f"Requires a non-null spike label for {horizon}."
        gate_note_map[f"gate_has_tradeable_label_{horizon}"] = (
            f"Requires a non-null tradeable pre-spike label for {horizon}."
        )

    for column in gate_details.columns:
        passed = int(gate_details[column].sum())
        summary_rows.append(
            {
                "metric_type": "gate",
                "metric_name": column,
                "metric_value": float(passed),
                "rows": passed,
                "rate": float(passed / total_rows) if total_rows else None,
                "notes": gate_note_map.get(column, "Gate pass count."),
            }
        )

    label_columns = _present_any_label_columns(feature_frame)
    for column in label_columns:
        present_count = int(feature_frame[column].notna().sum())
        summary_rows.append(
            {
                "metric_type": "label_presence",
                "metric_name": column,
                "metric_value": float(present_count),
                "rows": present_count,
                "rate": float(present_count / total_rows) if total_rows else None,
                "notes": "Rows with non-null label payload for this column.",
            }
        )

    if "symbol" in feature_frame.columns:
        symbol_total = feature_frame.groupby("symbol", dropna=False).size().rename("input_rows")
        symbol_export = export_frame.groupby("symbol", dropna=False).size().rename("exported_rows")
        symbol_summary = pd.concat([symbol_total, symbol_export], axis=1).fillna(0).reset_index()
        for _, row in symbol_summary.iterrows():
            input_rows = int(row["input_rows"])
            exported_symbol_rows = int(row["exported_rows"])
            summary_rows.append(
                {
                    "metric_type": "symbol",
                    "metric_name": str(row["symbol"]),
                    "metric_value": float(exported_symbol_rows),
                    "rows": exported_symbol_rows,
                    "rate": float(exported_symbol_rows / input_rows) if input_rows else None,
                    "notes": f"Export retention for symbol; input_rows={input_rows}.",
                }
            )

    return pd.DataFrame(summary_rows)


def write_frame_to_sqlite(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    table_name: str,
    if_exists: str,
) -> None:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_sql(table_name, connection, if_exists=if_exists, index=False)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    db_path = Path(args.db_path)
    LOGGER.info("Opening SQLite database: %s", db_path)

    with sqlite3.connect(db_path) as connection:
        feature_frame = load_table(connection, args.feature_table)
        validate_feature_frame(feature_frame)

        if args.timestamp_column in feature_frame.columns:
            feature_frame[args.timestamp_column] = pd.to_datetime(
                feature_frame[args.timestamp_column], utc=True, errors="coerce"
            )

        gate_mask, gate_details = build_gate_mask(
            feature_frame,
            required_horizons=args.required_horizons,
            require_tradeable_labels=args.require_tradeable_labels,
        )
        export_frame = feature_frame.loc[gate_mask].copy()

        LOGGER.info(
            "Hard-gated export retained %d/%d rows",
            len(export_frame),
            len(feature_frame),
        )

        if export_frame.empty and not args.allow_empty_export:
            raise ExportSpikeTrainingSetError(
                "training export is empty after hard quality gates; rerun with "
                "--allow-empty-export if this is expected"
            )

        LOGGER.info("Writing training rows to table: %s", args.output_table)
        write_frame_to_sqlite(
            connection,
            export_frame,
            table_name=args.output_table,
            if_exists=args.if_exists,
        )

        if not args.skip_summary:
            summary_frame = build_summary_frame(feature_frame, export_frame, gate_details)
            LOGGER.info("Writing training export summary to table: %s", args.summary_table)
            write_frame_to_sqlite(
                connection,
                summary_frame,
                table_name=args.summary_table,
                if_exists="replace",
            )
            LOGGER.info("Training export summary:\n%s", summary_frame.to_string(index=False))

        LOGGER.info(
            "Done. Exported rows=%d, columns=%d",
            len(export_frame),
            export_frame.shape[1],
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportSpikeTrainingSetError as exc:
        LOGGER.error("Export failed: %s", exc)
        raise SystemExit(2) from exc