from __future__ import annotations

"""
build_spike_quality_report.py

Summarize Phase 4.2 coverage quality from spike_feature_rows-like tables.

Purpose
-------
This script answers two questions:
1. Which symbols have usable training coverage?
2. Exactly why are rows failing the hard quality gates?

It reads the feature-layer output, computes global and per-symbol diagnostics,
and writes SQLite report tables for easy inspection.

Default expected inputs
-----------------------
- spike_feature_rows

Default outputs
---------------
- spike_quality_report_overall
- spike_quality_report_by_symbol
- spike_quality_report_failures
"""

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

LOGGER = logging.getLogger("build_spike_quality_report")

REQUIRED_COLUMNS = {
    "symbol",
    "timestamp",
    "timestamp_ms",
    "has_min_history_24h",
    "is_sparse_stream",
    "is_tail_unlabelable",
}

OPTIONAL_NUMERIC_COLUMNS = [
    "hours_since_prev_row",
    "rows_in_last_24h",
    "coverage_hours_24h",
    "coverage_ratio_24h",
]

OPTIONAL_GATE_COLUMNS = [
    "has_label_row",
]


@dataclass(frozen=True)
class ReportConfig:
    db_path: str
    feature_table: str
    overall_table: str
    by_symbol_table: str
    failure_table: str
    if_exists: str
    verbose: bool


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> ReportConfig:
    parser = argparse.ArgumentParser(
        description="Build a coverage / quality report from spike feature rows."
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database.")
    parser.add_argument(
        "--feature-table",
        default="spike_feature_rows",
        help="Input feature table.",
    )
    parser.add_argument(
        "--overall-table",
        default="spike_quality_report_overall",
        help="Output table for global quality metrics.",
    )
    parser.add_argument(
        "--by-symbol-table",
        default="spike_quality_report_by_symbol",
        help="Output table for per-symbol quality metrics.",
    )
    parser.add_argument(
        "--failure-table",
        default="spike_quality_report_failures",
        help="Output table for exact failure reason combinations.",
    )
    parser.add_argument(
        "--if-exists",
        default="fail",
        choices=("fail", "replace", "append"),
        help="SQLite write mode for output tables.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    configure_logging(args.verbose)
    return ReportConfig(
        db_path=args.db_path,
        feature_table=args.feature_table,
        overall_table=args.overall_table,
        by_symbol_table=args.by_symbol_table,
        failure_table=args.failure_table,
        if_exists=args.if_exists,
        verbose=args.verbose,
    )


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    )
    return cur.fetchone() is not None


def read_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(conn, table_name):
        raise ValueError(f"table does not exist: {table_name}")
    frame = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"table '{table_name}' missing required columns: {missing}")

    for col in REQUIRED_COLUMNS:
        if col in {"symbol", "timestamp"}:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    for col in OPTIONAL_NUMERIC_COLUMNS + OPTIONAL_GATE_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="raise").astype("int64")
    frame = frame.sort_values(["symbol", "timestamp_ms"], kind="stable").reset_index(drop=True)
    return frame


def fraction_true(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((valid > 0).mean())


def safe_mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def safe_median(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.median())


def compute_gate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["gate_has_min_history_24h"] = (work["has_min_history_24h"].fillna(0) == 1).astype("int64")
    work["gate_not_sparse_stream"] = (work["is_sparse_stream"].fillna(1) == 0).astype("int64")
    work["gate_not_tail_unlabelable"] = (work["is_tail_unlabelable"].fillna(0) == 0).astype("int64")

    if "has_label_row" in work.columns:
        work["gate_has_label_row"] = (work["has_label_row"].fillna(0) == 1).astype("int64")
    else:
        work["gate_has_label_row"] = 1

    gate_cols = [
        "gate_has_min_history_24h",
        "gate_not_sparse_stream",
        "gate_not_tail_unlabelable",
        "gate_has_label_row",
    ]
    work["passes_all_training_gates"] = work[gate_cols].all(axis=1).astype("int64")

    def reasons(row: pd.Series) -> str:
        items: List[str] = []
        if row["gate_has_min_history_24h"] != 1:
            items.append("min_history_24h")
        if row["gate_not_sparse_stream"] != 1:
            items.append("sparse_stream")
        if row["gate_not_tail_unlabelable"] != 1:
            items.append("tail_unlabelable")
        if row["gate_has_label_row"] != 1:
            items.append("missing_label_row")
        return "PASS" if not items else ",".join(items)

    work["failure_reasons"] = work.apply(reasons, axis=1)
    return work


def build_overall_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    input_rows = len(frame)

    def add(metric_name: str, metric_value: float | int | str | None, notes: str, rows_count: int | None = None, rate: float | None = None) -> None:
        rows.append(
            {
                "metric_type": "overall",
                "metric_name": metric_name,
                "metric_value": metric_value,
                "rows": input_rows if rows_count is None else rows_count,
                "rate": rate,
                "notes": notes,
            }
        )

    add("input_rows", input_rows, "Rows read from spike_feature_rows-like input.", rows_count=input_rows, rate=1.0 if input_rows else 0.0)
    add(
        "exportable_rows",
        int(frame["passes_all_training_gates"].sum()),
        "Rows passing all hard training gates.",
        rows_count=int(frame["passes_all_training_gates"].sum()),
        rate=fraction_true(frame["passes_all_training_gates"]),
    )
    add(
        "pct_has_min_history_24h",
        None,
        "Share of rows with has_min_history_24h = 1.",
        rate=fraction_true(frame["gate_has_min_history_24h"]),
    )
    add(
        "pct_not_sparse_stream",
        None,
        "Share of rows with is_sparse_stream = 0.",
        rate=fraction_true(frame["gate_not_sparse_stream"]),
    )
    add(
        "pct_not_tail_unlabelable",
        None,
        "Share of rows with is_tail_unlabelable = 0.",
        rate=fraction_true(frame["gate_not_tail_unlabelable"]),
    )
    add(
        "pct_has_label_row",
        None,
        "Share of rows with has_label_row = 1 (or assumed 1 if absent).",
        rate=fraction_true(frame["gate_has_label_row"]),
    )

    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in frame.columns:
            add(f"mean_{col}", safe_mean(frame[col]), f"Mean of {col} across all rows.")
            add(f"median_{col}", safe_median(frame[col]), f"Median of {col} across all rows.")

    failure_counts = frame.loc[frame["failure_reasons"] != "PASS", "failure_reasons"].value_counts()
    for reason, count in failure_counts.items():
        add(
            f"failure_combo::{reason}",
            int(count),
            "Rows failing this exact combination of gates.",
            rows_count=int(count),
            rate=float(count) / input_rows if input_rows else 0.0,
        )

    return pd.DataFrame(rows)


def build_symbol_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for symbol, grp in frame.groupby("symbol", sort=True):
        row: Dict[str, object] = {
            "symbol": symbol,
            "input_rows": int(len(grp)),
            "first_timestamp": grp["timestamp"].min(),
            "last_timestamp": grp["timestamp"].max(),
            "exportable_rows": int(grp["passes_all_training_gates"].sum()),
            "pct_exportable": fraction_true(grp["passes_all_training_gates"]),
            "pct_has_min_history_24h": fraction_true(grp["gate_has_min_history_24h"]),
            "pct_not_sparse_stream": fraction_true(grp["gate_not_sparse_stream"]),
            "pct_not_tail_unlabelable": fraction_true(grp["gate_not_tail_unlabelable"]),
            "pct_has_label_row": fraction_true(grp["gate_has_label_row"]),
            "failed_min_history_24h_rows": int((grp["gate_has_min_history_24h"] != 1).sum()),
            "failed_sparse_stream_rows": int((grp["gate_not_sparse_stream"] != 1).sum()),
            "failed_tail_unlabelable_rows": int((grp["gate_not_tail_unlabelable"] != 1).sum()),
            "failed_missing_label_rows": int((grp["gate_has_label_row"] != 1).sum()),
            "top_failure_reason": grp.loc[grp["failure_reasons"] != "PASS", "failure_reasons"].mode().iloc[0]
            if (grp["failure_reasons"] != "PASS").any()
            else "PASS",
        }

        for col in OPTIONAL_NUMERIC_COLUMNS:
            if col in grp.columns:
                row[f"mean_{col}"] = safe_mean(grp[col])
                row[f"median_{col}"] = safe_median(grp[col])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["pct_exportable", "symbol"], ascending=[False, True], kind="stable")


def build_failure_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (symbol, reasons), grp in frame.groupby(["symbol", "failure_reasons"], dropna=False, sort=True):
        rows.append(
            {
                "symbol": symbol,
                "failure_reasons": reasons,
                "row_count": int(len(grp)),
                "first_timestamp": grp["timestamp"].min(),
                "last_timestamp": grp["timestamp"].max(),
                "pct_of_symbol": float(len(grp)) / float(len(frame.loc[frame["symbol"] == symbol])) if len(frame.loc[frame["symbol"] == symbol]) else 0.0,
                "pct_of_dataset": float(len(grp)) / float(len(frame)) if len(frame) else 0.0,
                "mean_coverage_ratio_24h": safe_mean(grp["coverage_ratio_24h"]) if "coverage_ratio_24h" in grp.columns else float("nan"),
                "median_hours_since_prev_row": safe_median(grp["hours_since_prev_row"]) if "hours_since_prev_row" in grp.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["row_count", "symbol", "failure_reasons"], ascending=[False, True, True], kind="stable")


def write_frame_sqlite(conn: sqlite3.Connection, frame: pd.DataFrame, table_name: str, if_exists: str) -> None:
    frame.to_sql(table_name, conn, if_exists=if_exists, index=False)


def main() -> int:
    config = parse_args()
    LOGGER.info("Opening SQLite database: %s", config.db_path)
    conn = sqlite3.connect(config.db_path)
    try:
        features = read_table(conn, config.feature_table)
        features = compute_gate_columns(features)

        overall = build_overall_report(features)
        by_symbol = build_symbol_report(features)
        failures = build_failure_report(features)

        LOGGER.info("Writing overall report -> %s", config.overall_table)
        write_frame_sqlite(conn, overall, config.overall_table, config.if_exists)
        LOGGER.info("Writing per-symbol report -> %s", config.by_symbol_table)
        write_frame_sqlite(conn, by_symbol, config.by_symbol_table, config.if_exists)
        LOGGER.info("Writing failure report -> %s", config.failure_table)
        write_frame_sqlite(conn, failures, config.failure_table, config.if_exists)

        LOGGER.info("Overall quality report:\n%s", overall.to_string(index=False))
        if not by_symbol.empty:
            LOGGER.info("Per-symbol quality report:\n%s", by_symbol.to_string(index=False))
        if not failures.empty:
            LOGGER.info("Failure combinations:\n%s", failures.to_string(index=False))

        conn.commit()
        return 0
    except Exception as exc:
        LOGGER.error("Quality report build failed: %s", exc)
        conn.rollback()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
