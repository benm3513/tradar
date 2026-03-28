#!/usr/bin/env python3
"""
Train a baseline spike-prediction model from spike_training_rows.

Phase 4 baseline:
- target: label_spike_24h
- time-aware split (no shuffle)
- per-row sample weights optional
- baseline model: LogisticRegression
- optional second model: HistGradientBoostingClassifier
- writes metrics + predictions to SQLite

Example:
./.venv/bin/python scripts/train_spike_model.py \
  --db-path tradarbot.db \
  --input-table spike_training_rows \
  --target-column label_spike_24h \
  --metrics-table spike_model_metrics \
  --predictions-table spike_model_predictions \
  --model logistic
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Sequence

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


@dataclass
class TrainResult:
    model_name: str
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
    report_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--input-table", default="spike_training_rows")
    parser.add_argument("--target-column", default="label_spike_24h")
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
    parser.add_argument("--predictions-table", default="spike_model_predictions")
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="replace")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_frame(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    query = f"SELECT * FROM {table}"
    return pd.read_sql_query(query, conn)


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def prepare_frame(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    validate_columns(df, list(feature_columns) + [target_column, timestamp_column])

    out = df.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    out = out.sort_values(["symbol", timestamp_column]).reset_index(drop=True)

    # Only keep rows with an actual binary label.
    out = out[out[target_column].isin([0, 1])].copy()

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
    model_name: str,
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
        report_json=json.dumps(report),
    )


def write_table(conn: sqlite3.Connection, df: pd.DataFrame, table: str, if_exists: str) -> None:
    df.to_sql(table, conn, if_exists=if_exists, index=False)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    LOGGER.info("Opening SQLite database: %s", args.db_path)
    conn = sqlite3.connect(args.db_path)
    try:
        df = load_frame(conn, args.input_table)
        LOGGER.info("Loaded %d rows from %s", len(df), args.input_table)

        prepared = prepare_frame(
            df=df,
            feature_columns=args.feature_columns,
            target_column=args.target_column,
            timestamp_column=args.timestamp_column,
        )
        LOGGER.info("Prepared %d rows with target=%s", len(prepared), args.target_column)

        train_df, test_df = time_split(prepared, args.train_fraction)
        LOGGER.info("Time split complete: train=%d test=%d", len(train_df), len(test_df))

        X_train = train_df[list(args.feature_columns)]
        y_train = train_df[args.target_column].astype(int).to_numpy()
        X_test = test_df[list(args.feature_columns)]
        y_test = test_df[args.target_column].astype(int).to_numpy()

        model = build_model(args.model, args.feature_columns)
        LOGGER.info("Training model=%s", args.model)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        result = evaluate(
            model_name=args.model,
            target_column=args.target_column,
            y_train=y_train,
            y_test=y_test,
            y_prob=y_prob,
        )

        LOGGER.info(
            "Metrics: roc_auc=%s pr_auc=%s brier=%s best_threshold=%s f1=%s precision=%s recall=%s",
            result.roc_auc,
            result.pr_auc,
            result.brier_score,
            result.best_threshold_f1,
            result.f1_at_best_f1,
            result.precision_at_best_f1,
            result.recall_at_best_f1,
        )

        metrics_df = pd.DataFrame(
            [
                {
                    "model_name": result.model_name,
                    "target_column": result.target_column,
                    "train_rows": result.train_rows,
                    "test_rows": result.test_rows,
                    "positive_rate_train": result.positive_rate_train,
                    "positive_rate_test": result.positive_rate_test,
                    "roc_auc": result.roc_auc,
                    "pr_auc": result.pr_auc,
                    "brier_score": result.brier_score,
                    "best_threshold_f1": result.best_threshold_f1,
                    "precision_at_best_f1": result.precision_at_best_f1,
                    "recall_at_best_f1": result.recall_at_best_f1,
                    "f1_at_best_f1": result.f1_at_best_f1,
                    "classification_report_json": result.report_json,
                }
            ]
        )

        predictions_df = test_df[[args.symbol_column, args.timestamp_column, args.target_column]].copy()
        predictions_df["pred_prob"] = y_prob
        predictions_df["pred_label_at_best_f1"] = (y_prob >= result.best_threshold_f1).astype(int)
        predictions_df["model_name"] = args.model

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