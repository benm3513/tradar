"""Build ensemble spike predictions from multi-horizon model outputs.

Phase 4.7 ensemble layer
------------------------
This module combines separate horizon-specific prediction tables into a single
ensemble prediction table for downstream replay and sweep workflows.

Compatible inputs
-----------------
Expected per-horizon tables (defaults):
- spike_model_predictions_6h
- spike_model_predictions_24h
- spike_model_predictions_72h

Each table should contain at minimum:
- symbol
- timestamp
- pred_prob

Optional columns are preserved when available:
- model_name
- horizon
- target_column

Compatible outputs
------------------
The output table contains at minimum:
- symbol
- timestamp
- prob_6h
- prob_24h
- prob_72h
- prob_ensemble
- ensemble_score
- agreement_count
- agreement_boost_applied
- model_name

Design notes
------------
- Deterministic and SQLite-friendly
- No external dependencies beyond pandas/stdlib
- Safe with either legacy or upgraded training outputs
- Inner-joins only rows present at all three horizons
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger("ensemble")


@dataclass(frozen=True)
class EnsembleBuildResult:
    output_table: str
    rows_written: int
    weight_6h: float
    weight_24h: float
    weight_72h: float
    agreement_threshold: float
    agreement_boost: float


def _validate_weights(w6: float, w24: float, w72: float, tol: float = 1e-9) -> None:
    for name, value in (("w6", w6), ("w24", w24), ("w72", w72)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    total = w6 + w24 + w72
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"ensemble weights must sum to 1.0; got {total:.12f} "
            f"(w6={w6}, w24={w24}, w72={w72})"
        )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _load_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not _table_exists(conn, table_name):
        raise ValueError(f"prediction table does not exist: {table_name}")
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    if df.empty:
        raise ValueError(f"prediction table is empty: {table_name}")
    return df


def _resolve_prob_column(df: pd.DataFrame, preferred: str = "pred_prob") -> str:
    candidates = [
        preferred,
        "prob_ensemble",
        "ensemble_score",
        "prob_6h",
        "prob_24h",
        "prob_72h",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"could not resolve probability column; tried {candidates} against columns={list(df.columns)}"
    )


def _prepare_horizon_frame(
    df: pd.DataFrame,
    *,
    horizon_name: str,
    symbol_column: str,
    timestamp_column: str,
    preferred_prob_column: str,
    model_name_column: str,
) -> pd.DataFrame:
    required = [symbol_column, timestamp_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{horizon_name} table missing required columns: {missing}")

    prob_col = _resolve_prob_column(df, preferred=preferred_prob_column)

    out = df.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce")
    out = out.dropna(subset=[symbol_column, timestamp_column, prob_col]).copy()

    duplicate_count = int(out.duplicated(subset=[symbol_column, timestamp_column]).sum())
    if duplicate_count > 0:
        raise ValueError(
            f"{horizon_name} table contains duplicate symbol/timestamp rows: {duplicate_count}"
        )

    keep_cols = [symbol_column, timestamp_column]
    renamed_prob_col = f"prob_{horizon_name}"
    out = out[keep_cols + [prob_col]].rename(columns={prob_col: renamed_prob_col})

    if model_name_column in df.columns:
        model_name_src = df[[symbol_column, timestamp_column, model_name_column]].copy()
        model_name_src[timestamp_column] = pd.to_datetime(
            model_name_src[timestamp_column], utc=True, errors="raise"
        )
        model_name_src = model_name_src.drop_duplicates([symbol_column, timestamp_column])
        model_name_src = model_name_src.rename(
            columns={model_name_column: f"model_name_{horizon_name}"}
        )
        out = out.merge(
            model_name_src,
            on=[symbol_column, timestamp_column],
            how="left",
            validate="one_to_one",
        )

    return out.sort_values([timestamp_column, symbol_column]).reset_index(drop=True)


def _build_model_name(row: pd.Series) -> str:
    names = []
    for col in ("model_name_6h", "model_name_24h", "model_name_72h"):
        value = row.get(col)
        if pd.notna(value):
            names.append(str(value))
    if not names:
        return "ensemble"
    unique_names = []
    seen = set()
    for name in names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    return "ensemble[" + "|".join(unique_names) + "]"


def build_ensemble_predictions(
    *,
    db_path: str,
    w6: float = 0.3,
    w24: float = 0.5,
    w72: float = 0.2,
    agreement_boost: float = 0.05,
    threshold: float = 0.8,
    table_6h: str = "spike_model_predictions_6h",
    table_24h: str = "spike_model_predictions_24h",
    table_72h: str = "spike_model_predictions_72h",
    output_table: str = "spike_model_predictions_ensemble",
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
    prob_column: str = "pred_prob",
    model_name_column: str = "model_name",
) -> pd.DataFrame:
    """Build and persist ensemble predictions.

    Parameters
    ----------
    db_path:
        SQLite database path.
    w6, w24, w72:
        Ensemble weights for the 6h, 24h, and 72h models. Must sum to 1.
    agreement_boost:
        Additive boost applied when at least two horizon probabilities exceed
        ``threshold``.
    threshold:
        Agreement threshold used to count horizon agreement.
    table_6h, table_24h, table_72h:
        Source prediction tables.
    output_table:
        Destination table written with ``if_exists='replace'``.

    Returns
    -------
    pandas.DataFrame
        The ensemble prediction frame that was written to SQLite.
    """
    _validate_weights(w6, w24, w72)
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in [0, 1]")
    if agreement_boost < 0:
        raise ValueError("agreement_boost must be >= 0")

    LOGGER.info(
        "Building ensemble predictions: db=%s w6=%.3f w24=%.3f w72=%.3f threshold=%.3f boost=%.3f",
        db_path,
        w6,
        w24,
        w72,
        threshold,
        agreement_boost,
    )

    conn = sqlite3.connect(db_path)
    try:
        df6_raw = _load_table(conn, table_6h)
        df24_raw = _load_table(conn, table_24h)
        df72_raw = _load_table(conn, table_72h)

        df6 = _prepare_horizon_frame(
            df6_raw,
            horizon_name="6h",
            symbol_column=symbol_column,
            timestamp_column=timestamp_column,
            preferred_prob_column=prob_column,
            model_name_column=model_name_column,
        )
        df24 = _prepare_horizon_frame(
            df24_raw,
            horizon_name="24h",
            symbol_column=symbol_column,
            timestamp_column=timestamp_column,
            preferred_prob_column=prob_column,
            model_name_column=model_name_column,
        )
        df72 = _prepare_horizon_frame(
            df72_raw,
            horizon_name="72h",
            symbol_column=symbol_column,
            timestamp_column=timestamp_column,
            preferred_prob_column=prob_column,
            model_name_column=model_name_column,
        )

        merged = df6.merge(
            df24,
            on=[symbol_column, timestamp_column],
            how="inner",
            validate="one_to_one",
        )
        merged = merged.merge(
            df72,
            on=[symbol_column, timestamp_column],
            how="inner",
            validate="one_to_one",
        )

        if merged.empty:
            raise ValueError(
                "ensemble merge produced zero rows; check that 6h/24h/72h predictions overlap on symbol/timestamp"
            )

        for col in ("prob_6h", "prob_24h", "prob_72h"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").clip(lower=0.0, upper=1.0)

        merged = merged.dropna(subset=["prob_6h", "prob_24h", "prob_72h"]).copy()
        if merged.empty:
            raise ValueError("ensemble merge contains no valid non-null probability rows after cleaning")

        merged["prob_ensemble"] = (
            float(w6) * merged["prob_6h"]
            + float(w24) * merged["prob_24h"]
            + float(w72) * merged["prob_72h"]
        )

        agreement_mask_6h = merged["prob_6h"] >= float(threshold)
        agreement_mask_24h = merged["prob_24h"] >= float(threshold)
        agreement_mask_72h = merged["prob_72h"] >= float(threshold)

        merged["agreement_count"] = (
            agreement_mask_6h.astype(int)
            + agreement_mask_24h.astype(int)
            + agreement_mask_72h.astype(int)
        )
        merged["agreement_boost_applied"] = (merged["agreement_count"] >= 2).astype(int) * float(
            agreement_boost
        )

        merged["ensemble_score"] = (
            merged["prob_ensemble"] + merged["agreement_boost_applied"]
        ).clip(lower=0.0, upper=1.0)

        merged["weight_6h"] = float(w6)
        merged["weight_24h"] = float(w24)
        merged["weight_72h"] = float(w72)
        merged["agreement_threshold"] = float(threshold)
        merged["agreement_boost"] = float(agreement_boost)
        merged["model_name"] = merged.apply(_build_model_name, axis=1)
        merged["target_column"] = "ensemble_score"
        merged["horizon"] = "ensemble"

        output_columns = [
            symbol_column,
            timestamp_column,
            "prob_6h",
            "prob_24h",
            "prob_72h",
            "prob_ensemble",
            "ensemble_score",
            "agreement_count",
            "agreement_boost_applied",
            "weight_6h",
            "weight_24h",
            "weight_72h",
            "agreement_threshold",
            "agreement_boost",
            "model_name",
            "target_column",
            "horizon",
        ]

        if "model_name_6h" in merged.columns:
            output_columns.append("model_name_6h")
        if "model_name_24h" in merged.columns:
            output_columns.append("model_name_24h")
        if "model_name_72h" in merged.columns:
            output_columns.append("model_name_72h")

        out = merged[output_columns].copy()

        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        out.to_sql(output_table, conn, if_exists="replace", index=False)

        result = EnsembleBuildResult(
            output_table=output_table,
            rows_written=int(len(out)),
            weight_6h=float(w6),
            weight_24h=float(w24),
            weight_72h=float(w72),
            agreement_threshold=float(threshold),
            agreement_boost=float(agreement_boost),
        )
        LOGGER.info("Wrote ensemble predictions -> %s (%d rows)", result.output_table, result.rows_written)

        return merged[output_columns].copy()

    finally:
        conn.close()


__all__ = [
    "EnsembleBuildResult",
    "build_ensemble_predictions",
]