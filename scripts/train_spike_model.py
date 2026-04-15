#!/usr/bin/env python3
"""
Train spike-prediction models from spike_training_rows.

Phase 4.7 upgrades
------------------
- Supports multi-horizon targets:
    - 6h
    - 24h
    - 72h
- Can train one target or all supported horizons in one run
- Preserves backward compatibility with legacy label columns
- Writes horizon-aware metrics and predictions to SQLite
- Uses time-aware split (no shuffle)
- Supports LogisticRegression or HistGradientBoostingClassifier

Examples
--------
Train the 24h model only:
./.venv/bin/python scripts/train_spike_model.py \
  --db-path tradarbot.db \
  --input-table spike_training_rows \
  --target-column spike_24h_label \
  --metrics-table spike_model_metrics_24h \
  --predictions-table spike_model_predictions_24h \
  --model hgb

Train all horizons:
./.venv/bin/python scripts/train_spike_model.py \
  --db-path tradarbot.db \
  --input-table spike_training_rows \
  --train-all-horizons \
  --metrics-table spike_model_metrics_multi \
  --predictions-table-prefix spike_model_predictions \
  --model hgb
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("train_spike_model")

SUPPORTED_HORIZONS = ("6h", "24h", "72h")

DEFAULT_FEATURE_COLUMNS = [
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
]

DEFAULT_TARGET_COLUMN_BY_HORIZON = {
    "6h": "spike_6h_label",
    "24h": "spike_24h_label",
    "72h": "spike_72h_label",
}

LEGACY_TARGET_COLUMN_BY_HORIZON = {
    "6h": "label_spike_6h",
    "24h": "label_spike_24h",
    "72h": "label_spike_72h",
}


@dataclass
class TrainResult:
    model_name: str
    horizon: str
    target_column: str
    train_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_test: float
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    best_threshold_f1: float | None
    precision_at_best_f1: float | None
    recall_at_best_f1: float | None
    f1_at_best_f1: float | None
    classification_report_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--input-table", default="spike_training_rows")
    parser.add_argument("--target-column", default=None)
    parser.add_argument(
        "--horizon",
        choices=SUPPORTED_HORIZONS,
        default="24h",
        help="Target horizon to train when not using --train-all-horizons.",
    )
    parser.add_argument(
        "--train-all-horizons",
        action="store_true",
        help="Train separate models for 6h, 24h, and 72h in one run.",
    )
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--model", choices=["logistic", "hgb"], default="logistic")
    parser.add_argument("--metrics-table", default="spike_model_metrics")
    parser.add_argument(
        "--predictions-table",
        default="spike_model_predictions",
        help="Used for single-target training.",
    )
    parser.add_argument(
        "--predictions-table-prefix",
        default="spike_model_predictions",
        help="Used for --train-all-horizons. Outputs like spike_model_predictions_6h.",
    )
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="replace")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_frame(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    query = f'SELECT * FROM "{table}"'
    return pd.read_sql_query(query, conn)


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def resolve_target_column(df: pd.DataFrame, *, horizon: str, explicit_target_column: str | None) -> str:
    if explicit_target_column is not None:
        if explicit_target_column not in df.columns:
            raise KeyError(f"Explicit target column not found: {explicit_target_column}")
        return explicit_target_column

    preferred = DEFAULT_TARGET_COLUMN_BY_HORIZON[horizon]
    legacy = LEGACY_TARGET_COLUMN_BY_HORIZON[horizon]

    if preferred in df.columns:
        return preferred
    if legacy in df.columns:
        return legacy

    raise KeyError(
        f"No target column found for horizon={horizon}. "
        f"Tried {preferred!r} and {legacy!r}."
    )


def prepare_frame(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    timestamp_column: str,
    symbol_column: str,
) -> pd.DataFrame:
    validate_columns(df, list(feature_columns) + [target_column, timestamp_column, symbol_column])

    out = df.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    out = out.sort_values([symbol_column, timestamp_column]).reset_index(drop=True)

    # Keep only rows with an actual binary label.
    out = out[out[target_column].isin([0, 1])].copy()

    if out.empty:
        raise ValueError(f"No rows remain after filtering to binary target values for {target_column}")

    return out


def time_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.5 <= train_fraction < 1.0):
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    split_idx = int(len(df) * train_fraction)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition")
    return train_df, test_df


def build_model(model_name: str, numeric_features: Sequence[str]) -> Pipeline:
    numeric_preproc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_preproc, list(numeric_features))],
        remainder="drop",
    )

    if model_name == "logistic":
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "hgb":
        estimator = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def choose_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    denom = precision_t + recall_t
    f1 = np.zeros_like(precision_t, dtype=float)
    np.divide(
        2 * precision_t * recall_t,
        denom,
        out=f1,
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1))

    return (
        float(thresholds[best_idx]),
        float(precision_t[best_idx]),
        float(recall_t[best_idx]),
        float(f1[best_idx]),
    )


def evaluate(
    *,
    model_name: str,
    horizon: str,
    target_column: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_prob: np.ndarray,
) -> TrainResult:
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None
    brier = brier_score_loss(y_test, y_prob)

    best_threshold, best_precision, best_recall, best_f1 = choose_best_f1_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return TrainResult(
        model_name=model_name,
        horizon=horizon,
        target_column=target_column,
        train_rows=int(len(y_train)),
        test_rows=int(len(y_test)),
        positive_rate_train=float(np.mean(y_train)),
        positive_rate_test=float(np.mean(y_test)),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        pr_auc=float(pr_auc) if pr_auc is not None else None,
        brier_score=float(brier),
        best_threshold_f1=float(best_threshold),
        precision_at_best_f1=float(best_precision),
        recall_at_best_f1=float(best_recall),
        f1_at_best_f1=float(best_f1),
        classification_report_json=json.dumps(report),
    )


def build_predictions_frame(
    *,
    test_df: pd.DataFrame,
    symbol_column: str,
    timestamp_column: str,
    target_column: str,
    y_prob: np.ndarray,
    result: TrainResult,
) -> pd.DataFrame:
    out = test_df[[symbol_column, timestamp_column, target_column]].copy()
    out["pred_prob"] = y_prob
    out["pred_label_at_best_f1"] = (y_prob >= result.best_threshold_f1).astype(int)
    out["model_name"] = result.model_name
    out["horizon"] = result.horizon
    out["target_column"] = result.target_column
    return out


def write_table(conn: sqlite3.Connection, df: pd.DataFrame, table: str, if_exists: str) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_sql(table, conn, if_exists=if_exists, index=False)


def train_one_horizon(
    *,
    df: pd.DataFrame,
    horizon: str,
    explicit_target_column: str | None,
    feature_columns: Sequence[str],
    timestamp_column: str,
    symbol_column: str,
    train_fraction: float,
    model_name: str,
) -> tuple[TrainResult, pd.DataFrame]:
    target_column = resolve_target_column(
        df,
        horizon=horizon,
        explicit_target_column=explicit_target_column,
    )

    prepared = prepare_frame(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        timestamp_column=timestamp_column,
        symbol_column=symbol_column,
    )
    LOGGER.info(
        "Prepared %d rows for horizon=%s target=%s",
        len(prepared),
        horizon,
        target_column,
    )

    train_df, test_df = time_split(prepared, train_fraction)
    LOGGER.info(
        "Time split complete for horizon=%s: train=%d test=%d",
        horizon,
        len(train_df),
        len(test_df),
    )

    X_train = train_df[list(feature_columns)]
    y_train = train_df[target_column].astype(int).to_numpy()
    X_test = test_df[list(feature_columns)]
    y_test = test_df[target_column].astype(int).to_numpy()

    model = build_model(model_name, feature_columns)
    LOGGER.info("Training model=%s horizon=%s", model_name, horizon)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    result = evaluate(
        model_name=model_name,
        horizon=horizon,
        target_column=target_column,
        y_train=y_train,
        y_test=y_test,
        y_prob=y_prob,
    )

    LOGGER.info(
        "Metrics horizon=%s: roc_auc=%s pr_auc=%s brier=%s best_threshold=%s f1=%s precision=%s recall=%s",
        horizon,
        result.roc_auc,
        result.pr_auc,
        result.brier_score,
        result.best_threshold_f1,
        result.f1_at_best_f1,
        result.precision_at_best_f1,
        result.recall_at_best_f1,
    )

    predictions_df = build_predictions_frame(
        test_df=test_df,
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
        y_prob=y_prob,
        result=result,
    )
    return result, predictions_df


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    LOGGER.info("Opening SQLite database: %s", args.db_path)
    conn = sqlite3.connect(args.db_path)
    try:
        df = load_frame(conn, args.input_table)
        LOGGER.info("Loaded %d rows from %s", len(df), args.input_table)

        if args.train_all_horizons:
            metrics_rows: list[dict[str, object]] = []

            for idx, horizon in enumerate(SUPPORTED_HORIZONS):
                result, predictions_df = train_one_horizon(
                    df=df,
                    horizon=horizon,
                    explicit_target_column=None,
                    feature_columns=args.feature_columns,
                    timestamp_column=args.timestamp_column,
                    symbol_column=args.symbol_column,
                    train_fraction=args.train_fraction,
                    model_name=args.model,
                )

                metrics_rows.append(asdict(result))
                predictions_table = f"{args.predictions_table_prefix}_{horizon}"

                write_mode = args.if_exists if idx == 0 else "replace"
                write_table(conn, predictions_df, predictions_table, write_mode)
                LOGGER.info("Wrote predictions for horizon=%s -> %s", horizon, predictions_table)

            metrics_df = pd.DataFrame(metrics_rows)
            write_table(conn, metrics_df, args.metrics_table, args.if_exists)
            LOGGER.info("Wrote multi-horizon metrics -> %s", args.metrics_table)

        else:
            horizon = args.horizon
            result, predictions_df = train_one_horizon(
                df=df,
                horizon=horizon,
                explicit_target_column=args.target_column,
                feature_columns=args.feature_columns,
                timestamp_column=args.timestamp_column,
                symbol_column=args.symbol_column,
                train_fraction=args.train_fraction,
                model_name=args.model,
            )

            metrics_df = pd.DataFrame([asdict(result)])

            write_table(conn, metrics_df, args.metrics_table, args.if_exists)
            write_table(conn, predictions_df, args.predictions_table, args.if_exists)
            LOGGER.info(
                "Wrote metrics -> %s and predictions -> %s",
                args.metrics_table,
                args.predictions_table,
            )

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())