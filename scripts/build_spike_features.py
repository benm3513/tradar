#!/usr/bin/env python3
"""Build time-aware market features for Phase 4 spike-regime research.

This script computes backward-looking market features from the base research
rows and optionally merges them with the labeled dataset to produce a single
feature table suitable for downstream inspection and training export.

Signal-quality upgrades
-----------------------
Adds four new feature families:
- momentum acceleration
- volume spikes vs baseline
- cross-asset correlation / relative strength
- regime detection

Design goals
------------
- No trading-engine changes
- Strictly no future leakage
- Per-symbol computation only for asset-level features
- Cross-asset features computed only from same-timestamp or prior information
- Time-aware windows instead of row-count assumptions
- SQLite -> pandas -> SQLite flow

Typical usage
-------------
python scripts/build_spike_features.py \
    --db-path tradarbot.db \
    --base-table spike_base_rows \
    --label-table spike_labeled_rows \
    --output-table spike_feature_rows \
    --summary-table spike_feature_summary
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("build_spike_features")

DEFAULT_ASSET_KEY_COLUMNS = ["asset_id", "symbol", "exchange"]
DEFAULT_TIME_COLUMN = "timestamp"
DEFAULT_PRICE_COLUMN = "price_close"
DEFAULT_OUTPUT_TABLE = "spike_feature_rows"
DEFAULT_SUMMARY_TABLE = "spike_feature_summary"
SUPPORTED_HORIZONS = ("6h", "24h", "72h", "7d")

FEATURE_COLUMNS = [
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "rolling_volatility_24h",
    "range_pct_24h",
    "volume_ratio_1h_vs_24h_avg",
    "volume_ratio_current_vs_24h_avg",
    "drawup_from_recent_low_24h",
    "hours_since_prev_row",
    "rows_in_last_24h",
    "coverage_hours_24h",
    "coverage_ratio_24h",
    # momentum acceleration
    "momentum_accel_1h_vs_6h",
    "momentum_accel_6h_vs_24h",
    "price_zscore_24h",
    # volume spikes vs baseline
    "volume_zscore_24h",
    "volume_spike_ratio_24h",
    "volume_spike_ratio_7d",
    # cross-asset relative / correlation
    "market_ret_1h_ex_self",
    "market_ret_24h_ex_self",
    "relative_strength_1h",
    "relative_strength_24h",
    "corr_to_market_24h",
    "beta_to_market_24h",
    # regime detection
    "market_breadth_up_1h",
    "market_breadth_up_24h",
    "market_dispersion_1h",
    "market_dispersion_24h",
    "market_trend_strength_24h",
    "market_volume_regime_24h",
    "market_risk_off_score",
]

QUALITY_FLAG_COLUMNS = [
    "has_min_history_24h",
    "is_sparse_stream",
    "is_tail_unlabelable",
    "has_label_row",
]

MULTI_HORIZON_LABEL_COLUMNS = [
    "spike_6h_label",
    "spike_24h_label",
    "spike_72h_label",
    "spike_7d_label",
    "tradeable_pre_spike_6h_label",
    "tradeable_pre_spike_24h_label",
    "tradeable_pre_spike_72h_label",
    "tradeable_pre_spike_7d_label",
]

LEGACY_LABEL_COLUMNS = [
    "label_spike_6h",
    "label_spike_24h",
    "label_spike_72h",
    "label_spike_7d",
    "label_tradeable_pre_spike_6h",
    "label_tradeable_pre_spike_24h",
    "label_tradeable_pre_spike_72h",
    "label_tradeable_pre_spike_7d",
]


class BuildSpikeFeaturesError(RuntimeError):
    """Raised when the feature build cannot proceed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spike market features into SQLite.")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database.")
    parser.add_argument(
        "--base-table",
        default="spike_base_rows",
        help="Base research table containing normalized OHLCV rows.",
    )
    parser.add_argument(
        "--label-table",
        default="spike_labeled_rows",
        help=(
            "Optional labeled table to merge in. Pass an empty string to disable merging. "
            "Default: spike_labeled_rows"
        ),
    )
    parser.add_argument(
        "--output-table",
        default=DEFAULT_OUTPUT_TABLE,
        help=f"Destination table for enriched feature rows. Default: {DEFAULT_OUTPUT_TABLE}",
    )
    parser.add_argument(
        "--summary-table",
        default=DEFAULT_SUMMARY_TABLE,
        help=f"Destination table for compact feature summary. Default: {DEFAULT_SUMMARY_TABLE}",
    )
    parser.add_argument(
        "--timestamp-column",
        default=DEFAULT_TIME_COLUMN,
        help="Timestamp column name. Default: timestamp",
    )
    parser.add_argument(
        "--price-column",
        default=DEFAULT_PRICE_COLUMN,
        help="Close-price column name. Default: price_close",
    )
    parser.add_argument(
        "--asset-key-columns",
        nargs="+",
        default=DEFAULT_ASSET_KEY_COLUMNS,
        help="Asset key columns. Present columns will be used.",
    )
    parser.add_argument(
        "--if-exists",
        choices=("fail", "replace", "append"),
        default="replace",
        help="How to write the output table. Default: replace",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not write the summary table.",
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
    if not table_name:
        return pd.DataFrame()
    if not table_exists(connection, table_name):
        raise BuildSpikeFeaturesError(f"table does not exist: {table_name}")
    frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
    if frame.empty:
        raise BuildSpikeFeaturesError(f"table is empty: {table_name}")
    return frame


def resolve_asset_key_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    present = [col for col in candidates if col in frame.columns]
    if present:
        return present
    raise BuildSpikeFeaturesError(
        "No asset key columns found. Provide identifiers such as asset_id, symbol, or exchange."
    )


def validate_base_frame(
    frame: pd.DataFrame,
    *,
    asset_key_columns: Sequence[str],
    timestamp_column: str,
    price_column: str,
) -> None:
    required = set(asset_key_columns) | {
        timestamp_column,
        price_column,
        "price_high",
        "price_low",
        "volume_base",
    }
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise BuildSpikeFeaturesError(f"base table missing required columns: {missing}")

    duplicate_count = int(frame.duplicated(subset=[*asset_key_columns, timestamp_column]).sum())
    if duplicate_count > 0:
        raise BuildSpikeFeaturesError(
            f"base table contains duplicate asset/timestamp rows: {duplicate_count}"
        )

    if frame[price_column].isna().any():
        raise BuildSpikeFeaturesError(
            f"price column contains {int(frame[price_column].isna().sum())} null rows"
        )

    if (frame[price_column] <= 0).any():
        raise BuildSpikeFeaturesError(
            f"price column contains {int((frame[price_column] <= 0).sum())} non-positive rows"
        )


def prepare_base_frame(
    frame: pd.DataFrame,
    *,
    asset_key_columns: Sequence[str],
    timestamp_column: str,
) -> pd.DataFrame:
    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    out = out.sort_values([*asset_key_columns, timestamp_column]).reset_index(drop=True)
    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_group_zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((numeric - numeric.mean()) / std).fillna(0.0)


def _compute_time_aware_returns(
    asset_frame: pd.DataFrame,
    *,
    timestamp_column: str,
    price_column: str,
    horizon: str,
) -> pd.Series:
    horizon_delta = pd.to_timedelta(horizon)
    lookup = asset_frame[[timestamp_column, price_column]].copy()
    lookup = lookup.rename(
        columns={timestamp_column: "lookup_timestamp", price_column: f"lookup_{price_column}"}
    )

    probe = asset_frame[[timestamp_column]].copy()
    probe["lookup_timestamp"] = probe[timestamp_column] - horizon_delta

    merged = pd.merge_asof(
        probe.sort_values("lookup_timestamp"),
        lookup.sort_values("lookup_timestamp"),
        on="lookup_timestamp",
        direction="backward",
    ).sort_index()

    historical_price = merged[f"lookup_{price_column}"]
    current_price = asset_frame[price_column].astype(float)
    return _safe_divide(current_price, historical_price) - 1.0


def _compute_rolling_corr_beta(
    asset_df: pd.DataFrame,
    *,
    timestamp_column: str,
    asset_return_column: str,
    market_return_column: str,
    window: int = 24,
    min_periods: int = 12,
) -> pd.DataFrame:
    out = asset_df.sort_values(timestamp_column).copy()
    x = pd.to_numeric(out[asset_return_column], errors="coerce")
    y = pd.to_numeric(out[market_return_column], errors="coerce")

    corr_values: list[float] = []
    beta_values: list[float] = []

    for idx in range(len(out)):
        start = max(0, idx - window + 1)
        xw = x.iloc[start: idx + 1]
        yw = y.iloc[start: idx + 1]

        valid = xw.notna() & yw.notna()
        if int(valid.sum()) < min_periods:
            corr_values.append(np.nan)
            beta_values.append(np.nan)
            continue

        xwv = xw[valid].astype(float)
        ywv = yw[valid].astype(float)

        x_std = float(xwv.std(ddof=0))
        y_std = float(ywv.std(ddof=0))
        if x_std <= 1e-12 or y_std <= 1e-12:
            corr_values.append(0.0)
        else:
            corr_values.append(float(xwv.corr(ywv)))

        market_var = float(ywv.var(ddof=0))
        if market_var <= 1e-12:
            beta_values.append(0.0)
        else:
            cov = float(((xwv - xwv.mean()) * (ywv - ywv.mean())).mean())
            beta_values.append(cov / market_var)

    out["corr_to_market_24h"] = corr_values
    out["beta_to_market_24h"] = beta_values
    return out


def build_features_for_asset(
    asset_frame: pd.DataFrame,
    *,
    timestamp_column: str,
    price_column: str,
) -> pd.DataFrame:
    asset_frame = asset_frame.sort_values(timestamp_column).reset_index(drop=True).copy()

    # Time-aware lagged returns.
    asset_frame["ret_1h"] = _compute_time_aware_returns(
        asset_frame,
        timestamp_column=timestamp_column,
        price_column=price_column,
        horizon="1h",
    )
    asset_frame["ret_6h"] = _compute_time_aware_returns(
        asset_frame,
        timestamp_column=timestamp_column,
        price_column=price_column,
        horizon="6h",
    )
    asset_frame["ret_24h"] = _compute_time_aware_returns(
        asset_frame,
        timestamp_column=timestamp_column,
        price_column=price_column,
        horizon="24h",
    )

    indexed = asset_frame.set_index(timestamp_column)
    one_step_return = indexed[price_column].astype(float).pct_change()
    trailing_window = "24h"

    asset_frame["rolling_volatility_24h"] = (
        one_step_return.rolling(trailing_window, min_periods=2).std().to_numpy()
    )

    trailing_high = indexed["price_high"].astype(float).rolling(trailing_window, min_periods=1).max()
    trailing_low = indexed["price_low"].astype(float).rolling(trailing_window, min_periods=1).min()
    current_close = indexed[price_column].astype(float)

    asset_frame["range_pct_24h"] = (
        _safe_divide(trailing_high - trailing_low, current_close).to_numpy()
    )

    current_volume = indexed["volume_base"].astype(float)
    trailing_vol_1h = current_volume.rolling("1h", min_periods=1).sum()
    trailing_vol_24h = current_volume.rolling(trailing_window, min_periods=1).sum()
    trailing_vol_7d = current_volume.rolling("168h", min_periods=24).sum()
    trailing_avg_hourly_vol_24h = trailing_vol_24h / 24.0
    trailing_avg_hourly_vol_7d = trailing_vol_7d / 168.0
    trailing_mean_bar_vol_24h = current_volume.rolling(trailing_window, min_periods=1).mean()
    trailing_std_bar_vol_24h = current_volume.rolling(trailing_window, min_periods=6).std()

    asset_frame["volume_ratio_1h_vs_24h_avg"] = _safe_divide(
        trailing_vol_1h, trailing_avg_hourly_vol_24h
    ).to_numpy()
    asset_frame["volume_ratio_current_vs_24h_avg"] = _safe_divide(
        current_volume, trailing_mean_bar_vol_24h
    ).to_numpy()

    asset_frame["drawup_from_recent_low_24h"] = (
        _safe_divide(current_close, trailing_low).to_numpy() - 1.0
    )

    timestamps = asset_frame[timestamp_column]
    prev_timestamps = timestamps.shift(1)
    asset_frame["hours_since_prev_row"] = (
        (timestamps - prev_timestamps).dt.total_seconds() / 3600.0
    )

    rows_in_last_24h: list[int] = []
    coverage_hours_24h: list[float] = []
    ts_values = timestamps.to_list()
    left = 0
    window_24h = pd.Timedelta(hours=24)

    for right, current_ts in enumerate(ts_values):
        window_start = current_ts - window_24h
        while left < right and ts_values[left] < window_start:
            left += 1

        rows_in_window = right - left + 1
        earliest_ts = ts_values[left]
        covered_hours = min(24.0, max(0.0, (current_ts - earliest_ts).total_seconds() / 3600.0))

        rows_in_last_24h.append(int(rows_in_window))
        coverage_hours_24h.append(float(covered_hours))

    asset_frame["rows_in_last_24h"] = rows_in_last_24h
    asset_frame["coverage_hours_24h"] = coverage_hours_24h
    asset_frame["coverage_ratio_24h"] = asset_frame["coverage_hours_24h"] / 24.0
    asset_frame["has_min_history_24h"] = (
        (asset_frame["coverage_ratio_24h"] >= 0.7)
        & (asset_frame["rows_in_last_24h"] >= 12)
    ).astype(int)
    asset_frame["is_sparse_stream"] = (
        (asset_frame["hours_since_prev_row"] > 6.0)
        | (asset_frame["coverage_ratio_24h"] < 0.5)
    ).fillna(asset_frame["coverage_ratio_24h"] < 0.5).astype(int)

    # Signal-quality upgrades

    # 1) Momentum acceleration
    asset_frame["momentum_accel_1h_vs_6h"] = (
        asset_frame["ret_1h"] - (asset_frame["ret_6h"] / 6.0)
    )
    asset_frame["momentum_accel_6h_vs_24h"] = (
        asset_frame["ret_6h"] - (asset_frame["ret_24h"] / 4.0)
    )
    rolling_mean_close_24h = current_close.rolling("24h", min_periods=6).mean()
    rolling_std_close_24h = current_close.rolling("24h", min_periods=6).std()
    asset_frame["price_zscore_24h"] = _safe_divide(
        current_close - rolling_mean_close_24h,
        rolling_std_close_24h,
    ).to_numpy()

    # 2) Volume spikes vs baseline
    asset_frame["volume_zscore_24h"] = _safe_divide(
        current_volume - trailing_mean_bar_vol_24h,
        trailing_std_bar_vol_24h,
    ).to_numpy()
    asset_frame["volume_spike_ratio_24h"] = _safe_divide(
        current_volume,
        trailing_mean_bar_vol_24h,
    ).to_numpy()
    asset_frame["volume_spike_ratio_7d"] = _safe_divide(
        trailing_vol_1h,
        trailing_avg_hourly_vol_7d,
    ).to_numpy()

    return asset_frame.reset_index(drop=True)


def add_cross_asset_and_regime_features(
    feature_frame: pd.DataFrame,
    *,
    timestamp_column: str,
    symbol_column: str,
) -> pd.DataFrame:
    out = feature_frame.copy()

    required = [timestamp_column, symbol_column, "ret_1h", "ret_24h", "volume_ratio_1h_vs_24h_avg"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise BuildSpikeFeaturesError(f"feature frame missing columns for regime/correlation features: {missing}")

    # Cross-sectional market aggregates at each timestamp
    grouped = out.groupby(timestamp_column, dropna=False)

    out["market_mean_ret_1h"] = grouped["ret_1h"].transform("mean")
    out["market_mean_ret_24h"] = grouped["ret_24h"].transform("mean")
    out["market_sum_ret_1h"] = grouped["ret_1h"].transform("sum")
    out["market_sum_ret_24h"] = grouped["ret_24h"].transform("sum")
    out["market_asset_count"] = grouped[symbol_column].transform("count").astype(float)

    denom = (out["market_asset_count"] - 1.0).replace(0.0, np.nan)
    out["market_ret_1h_ex_self"] = _safe_divide(
        out["market_sum_ret_1h"] - pd.to_numeric(out["ret_1h"], errors="coerce"),
        denom,
    )
    out["market_ret_24h_ex_self"] = _safe_divide(
        out["market_sum_ret_24h"] - pd.to_numeric(out["ret_24h"], errors="coerce"),
        denom,
    )

    out["relative_strength_1h"] = (
        pd.to_numeric(out["ret_1h"], errors="coerce")
        - pd.to_numeric(out["market_ret_1h_ex_self"], errors="coerce")
    )
    out["relative_strength_24h"] = (
        pd.to_numeric(out["ret_24h"], errors="coerce")
        - pd.to_numeric(out["market_ret_24h_ex_self"], errors="coerce")
    )

    out["market_breadth_up_1h"] = grouped["ret_1h"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").gt(0).mean()
    )
    out["market_breadth_up_24h"] = grouped["ret_24h"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").gt(0).mean()
    )
    out["market_dispersion_1h"] = grouped["ret_1h"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)
    )
    out["market_dispersion_24h"] = grouped["ret_24h"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)
    )
    out["market_trend_strength_24h"] = grouped["ret_24h"].transform("mean")
    out["market_volume_regime_24h"] = grouped["volume_ratio_1h_vs_24h_avg"].transform("mean")

    # Risk-off style score: weak breadth + negative trend + high dispersion
    breadth_term = 0.5 - pd.to_numeric(out["market_breadth_up_24h"], errors="coerce")
    trend_term = -pd.to_numeric(out["market_trend_strength_24h"], errors="coerce")
    dispersion_term = pd.to_numeric(out["market_dispersion_1h"], errors="coerce")
    out["market_risk_off_score"] = (
        breadth_term.fillna(0.0)
        + trend_term.fillna(0.0)
        + dispersion_term.fillna(0.0)
    )

    # Rolling correlation / beta to market per asset stream
    groups: list[pd.DataFrame] = []
    for _, asset_df in out.groupby(symbol_column, sort=False, dropna=False):
        enhanced = _compute_rolling_corr_beta(
            asset_df,
            timestamp_column=timestamp_column,
            asset_return_column="ret_1h",
            market_return_column="market_ret_1h_ex_self",
            window=24,
            min_periods=12,
        )
        groups.append(enhanced)

    out = pd.concat(groups, ignore_index=True)
    return out.sort_values([timestamp_column, symbol_column]).reset_index(drop=True)


def build_feature_frame(
    base_frame: pd.DataFrame,
    *,
    asset_key_columns: Sequence[str],
    timestamp_column: str,
    price_column: str,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []

    for asset_key, asset_frame in base_frame.groupby(list(asset_key_columns), sort=False, dropna=False):
        featured = build_features_for_asset(
            asset_frame,
            timestamp_column=timestamp_column,
            price_column=price_column,
        )
        groups.append(featured)
        asset_key_tuple = asset_key if isinstance(asset_key, tuple) else (asset_key,)
        LOGGER.debug("Built features for asset stream %s with %d rows", asset_key_tuple, len(featured))

    if not groups:
        raise BuildSpikeFeaturesError("No asset groups were produced from the base frame")

    feature_frame = pd.concat(groups, ignore_index=True)
    feature_frame = add_cross_asset_and_regime_features(
        feature_frame,
        timestamp_column=timestamp_column,
        symbol_column="symbol",
    )
    return feature_frame


def _preferred_tail_label_column(columns: Sequence[str]) -> str | None:
    for candidate in ("spike_6h_label", "label_spike_6h", "spike_24h_label", "label_spike_24h"):
        if candidate in columns:
            return candidate
    return None


def _present_label_columns(columns: Sequence[str]) -> list[str]:
    return [
        col for col in (
            MULTI_HORIZON_LABEL_COLUMNS
            + LEGACY_LABEL_COLUMNS
            + [f"has_full_horizon_{h}" for h in SUPPORTED_HORIZONS]
            + [f"threshold_spike_return_{h}" for h in SUPPORTED_HORIZONS]
            + [f"threshold_tradeable_return_{h}" for h in SUPPORTED_HORIZONS]
            + [f"applied_spike_threshold_{h}" for h in SUPPORTED_HORIZONS]
            + [f"applied_tradeable_min_return_{h}" for h in SUPPORTED_HORIZONS]
        )
        if col in columns
    ]


def merge_with_labels(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    asset_key_columns: Sequence[str],
    timestamp_column: str,
) -> pd.DataFrame:
    if label_frame.empty:
        out = feature_frame.copy()
        out["has_label_row"] = 0
        out["is_tail_unlabelable"] = np.nan
        for horizon in SUPPORTED_HORIZONS:
            out[f"has_spike_label_{horizon}"] = 0
            out[f"has_tradeable_label_{horizon}"] = 0
        return out

    join_columns = [*asset_key_columns, timestamp_column]
    missing = [col for col in join_columns if col not in label_frame.columns]
    if missing:
        raise BuildSpikeFeaturesError(f"label table missing join columns: {missing}")

    label_frame = label_frame.copy()
    label_frame[timestamp_column] = pd.to_datetime(label_frame[timestamp_column], utc=True, errors="raise")

    duplicate_count = int(label_frame.duplicated(subset=join_columns).sum())
    if duplicate_count > 0:
        raise BuildSpikeFeaturesError(
            f"label table contains duplicate asset/timestamp rows: {duplicate_count}"
        )

    label_columns_to_keep = _present_label_columns(label_frame.columns)
    label_payload = label_frame[[*join_columns, *label_columns_to_keep]].copy()

    merged = feature_frame.merge(label_payload, on=join_columns, how="left", validate="one_to_one")

    label_columns = [
        col for col in merged.columns
        if col.endswith("_label") or col.startswith("label_")
    ]
    merged["has_label_row"] = merged[label_columns].notna().any(axis=1).astype(int) if label_columns else 0

    tail_label_col = _preferred_tail_label_column(merged.columns)
    if tail_label_col is not None:
        merged["is_tail_unlabelable"] = merged[tail_label_col].isna().astype(int)
    else:
        merged["is_tail_unlabelable"] = np.nan

    for horizon in SUPPORTED_HORIZONS:
        spike_candidates = [f"spike_{horizon}_label", f"label_spike_{horizon}"]
        tradeable_candidates = [
            f"tradeable_pre_spike_{horizon}_label",
            f"label_tradeable_pre_spike_{horizon}",
        ]

        spike_col = next((c for c in spike_candidates if c in merged.columns), None)
        tradeable_col = next((c for c in tradeable_candidates if c in merged.columns), None)

        merged[f"has_spike_label_{horizon}"] = (
            merged[spike_col].notna().astype(int) if spike_col is not None else 0
        )
        merged[f"has_tradeable_label_{horizon}"] = (
            merged[tradeable_col].notna().astype(int) if tradeable_col is not None else 0
        )

    return merged


def _build_value_summary_row(feature_frame: pd.DataFrame, column: str) -> dict[str, object]:
    series = pd.to_numeric(feature_frame[column], errors="coerce")
    non_null = series.dropna()
    return {
        "metric_type": "feature",
        "metric_name": column,
        "eligible_rows": int(non_null.shape[0]),
        "positive_rows": None,
        "positive_rate": None,
        "mean_value": float(non_null.mean()) if not non_null.empty else None,
        "median_value": float(non_null.median()) if not non_null.empty else None,
        "missing_rows": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()) if len(series) else None,
        "min_value": float(non_null.min()) if not non_null.empty else None,
        "max_value": float(non_null.max()) if not non_null.empty else None,
    }


def _build_binary_summary_row(feature_frame: pd.DataFrame, metric_type: str, column: str) -> dict[str, object]:
    series = pd.to_numeric(feature_frame[column], errors="coerce")
    non_null = series.dropna()
    return {
        "metric_type": metric_type,
        "metric_name": column,
        "eligible_rows": int(non_null.shape[0]),
        "positive_rows": int(non_null.sum()) if not non_null.empty else 0,
        "positive_rate": float(non_null.mean()) if not non_null.empty else None,
        "mean_value": None,
        "median_value": None,
        "missing_rows": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()) if len(series) else None,
        "min_value": None,
        "max_value": None,
    }


def build_summary_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []

    for column in FEATURE_COLUMNS:
        if column not in feature_frame.columns:
            continue
        summary_rows.append(_build_value_summary_row(feature_frame, column))

    quality_columns = QUALITY_FLAG_COLUMNS + [
        f"has_spike_label_{h}" for h in SUPPORTED_HORIZONS
    ] + [
        f"has_tradeable_label_{h}" for h in SUPPORTED_HORIZONS
    ]

    for column in quality_columns:
        if column not in feature_frame.columns:
            continue
        metric_type = (
            "join"
            if column in {
                "has_label_row",
                *[f"has_spike_label_{h}" for h in SUPPORTED_HORIZONS],
                *[f"has_tradeable_label_{h}" for h in SUPPORTED_HORIZONS],
            }
            else "quality"
        )
        summary_rows.append(_build_binary_summary_row(feature_frame, metric_type, column))

    alias_rows: list[tuple[str, str]] = [
        ("quality", "pct_has_min_history_24h"),
        ("quality", "pct_is_sparse_stream"),
        ("quality", "pct_is_tail_unlabelable"),
    ]
    alias_to_source = {
        "pct_has_min_history_24h": "has_min_history_24h",
        "pct_is_sparse_stream": "is_sparse_stream",
        "pct_is_tail_unlabelable": "is_tail_unlabelable",
    }
    for metric_type, alias_name in alias_rows:
        source_col = alias_to_source[alias_name]
        if source_col not in feature_frame.columns:
            continue
        series = pd.to_numeric(feature_frame[source_col], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric_type": metric_type,
                "metric_name": alias_name,
                "eligible_rows": int(series.shape[0]),
                "positive_rows": None,
                "positive_rate": float(series.mean()) if not series.empty else None,
                "mean_value": None,
                "median_value": None,
                "missing_rows": int(pd.to_numeric(feature_frame[source_col], errors="coerce").isna().sum()),
                "missing_rate": float(pd.to_numeric(feature_frame[source_col], errors="coerce").isna().mean()) if len(feature_frame) else None,
                "min_value": None,
                "max_value": None,
            }
        )

    if "hours_since_prev_row" in feature_frame.columns:
        series = pd.to_numeric(feature_frame["hours_since_prev_row"], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric_type": "quality",
                "metric_name": "mean_hours_since_prev_row",
                "eligible_rows": int(series.shape[0]),
                "positive_rows": None,
                "positive_rate": None,
                "mean_value": float(series.mean()) if not series.empty else None,
                "median_value": None,
                "missing_rows": int(pd.to_numeric(feature_frame["hours_since_prev_row"], errors="coerce").isna().sum()),
                "missing_rate": float(pd.to_numeric(feature_frame["hours_since_prev_row"], errors="coerce").isna().mean()) if len(feature_frame) else None,
                "min_value": None,
                "max_value": None,
            }
        )
        summary_rows.append(
            {
                "metric_type": "quality",
                "metric_name": "median_hours_since_prev_row",
                "eligible_rows": int(series.shape[0]),
                "positive_rows": None,
                "positive_rate": None,
                "mean_value": None,
                "median_value": float(series.median()) if not series.empty else None,
                "missing_rows": int(pd.to_numeric(feature_frame["hours_since_prev_row"], errors="coerce").isna().sum()),
                "missing_rate": float(pd.to_numeric(feature_frame["hours_since_prev_row"], errors="coerce").isna().mean()) if len(feature_frame) else None,
                "min_value": None,
                "max_value": None,
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
        base_frame = load_table(connection, args.base_table)
        asset_key_columns = resolve_asset_key_columns(base_frame, args.asset_key_columns)
        validate_base_frame(
            base_frame,
            asset_key_columns=asset_key_columns,
            timestamp_column=args.timestamp_column,
            price_column=args.price_column,
        )
        base_frame = prepare_base_frame(
            base_frame,
            asset_key_columns=asset_key_columns,
            timestamp_column=args.timestamp_column,
        )

        LOGGER.info(
            "Building features for %d rows across %d asset streams",
            len(base_frame),
            base_frame.groupby(asset_key_columns, dropna=False).ngroups,
        )
        feature_frame = build_feature_frame(
            base_frame,
            asset_key_columns=asset_key_columns,
            timestamp_column=args.timestamp_column,
            price_column=args.price_column,
        )

        label_table = (args.label_table or "").strip()
        if label_table:
            label_frame = load_table(connection, label_table)
            feature_frame = merge_with_labels(
                feature_frame,
                label_frame,
                asset_key_columns=asset_key_columns,
                timestamp_column=args.timestamp_column,
            )
            LOGGER.info(
                "Merged features with label table %s; rows with label payload=%d/%d",
                label_table,
                int(feature_frame["has_label_row"].sum()) if "has_label_row" in feature_frame.columns else 0,
                len(feature_frame),
            )
        else:
            feature_frame["has_label_row"] = 0
            feature_frame["is_tail_unlabelable"] = np.nan
            for horizon in SUPPORTED_HORIZONS:
                feature_frame[f"has_spike_label_{horizon}"] = 0
                feature_frame[f"has_tradeable_label_{horizon}"] = 0

        LOGGER.info("Writing feature rows to table: %s", args.output_table)
        write_frame_to_sqlite(
            connection,
            feature_frame,
            table_name=args.output_table,
            if_exists=args.if_exists,
        )

        if not args.skip_summary:
            summary_frame = build_summary_frame(feature_frame)
            LOGGER.info("Writing feature summary to table: %s", args.summary_table)
            write_frame_to_sqlite(
                connection,
                summary_frame,
                table_name=args.summary_table,
                if_exists="replace",
            )
            LOGGER.info("Feature summary:\n%s", summary_frame.to_string(index=False))

        LOGGER.info(
            "Done. Output rows=%d, columns=%d",
            len(feature_frame),
            feature_frame.shape[1],
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildSpikeFeaturesError as exc:
        LOGGER.error("Build failed: %s", exc)
        raise SystemExit(2) from exc