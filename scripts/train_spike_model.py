#!/usr/bin/env python3
"""
Train spike-prediction models from spike_training_rows.

Phase 4.8 upgrades
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
- Adds signal-quality feature set:
    - momentum acceleration
    - volume spikes vs baseline
    - cross-asset relative strength / correlation
    - regime detection
- Adds safeguards:
    - replace inf/-inf with NaN
    - drop rows missing required features/target before training
    - optional low-variance feature filtering
- Adds feature-importance outputs:
    - impurity-based importances for HGB
    - per-horizon SQLite tables
    - top-feature logging

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
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
    # Core returns
    "ret_1h",
    "ret_6h",
    "ret_24h",

    # Volatility / range
    "rolling_volatility_24h",
    "range_pct_24h",

    # Volume baseline features
    "volume_ratio_1h_vs_24h_avg",
    "volume_ratio_current_vs_24h_avg",

    # Price structure
    "drawup_from_recent_low_24h",

    # Data quality / timing
    "hours_since_prev_row",
    "rows_in_last_24h",
    "coverage_hours_24h",
    "coverage_ratio_24h",

    # Momentum acceleration
    "momentum_accel_1h_vs_6h",
    "momentum_accel_6h_vs_24h",

    # Z-score normalization
    "price_zscore_24h",
    "volume_zscore_24h",

    # Volume spike detection
    "volume_spike_ratio_24h",
    "volume_spike_ratio_7d",

    # Cross-asset relative strength
    "relative_strength_1h",
    "relative_strength_24h",

    # Market correlation / beta
    "corr_to_market_24h",
    "beta_to_market_24h",

    # Market regime features
    "market_breadth_up_1h",
    "market_breadth_up_24h",
    "market_dispersion_1h",
    "market_dispersion_24h",
    "market_trend_strength_24h",
    "market_volume_regime_24h",
    "market_risk_off_score",
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
    feature_count_requested: int
    feature_count_used: int
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
    parser.add_argument(
        "--drop-rows-with-missing-features",
        action="store_true",
        default=True,
        help="Drop rows with missing requested features before training. Default: enabled.",
    )
    parser.add_argument(
        "--no-drop-rows-with-missing-features",
        action="store_false",
        dest="drop_rows_with_missing_features",
        help="Do not drop rows with missing features before training; rely on imputation instead.",
    )
    parser.add_argument(
        "--enable-variance-threshold",
        action="store_true",
        help="Enable low-variance feature filtering before model training.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-6,
        help="Threshold used by sklearn VarianceThreshold when enabled.",
    )
    parser.add_argument(
        "--feature-importance-table-prefix",
        default="spike_feature_importance",
        help="Prefix for per-horizon feature importance tables.",
    )
    parser.add_argument(
        "--top-feature-count",
        type=int,
        default=10,
        help="How many top features to log.",
    )
    parser.add_argument(
        "--enable-permutation-importance",
        action="store_true",
        help="Also compute permutation importance on the test split for the trained model.",
    )
    parser.add_argument(
        "--permutation-importance-table-prefix",
        default="spike_permutation_importance",
        help="Prefix for per-horizon permutation-importance tables.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=5,
        help="Number of repeats for permutation importance.",
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
    *,
    drop_rows_with_missing_features: bool,
) -> pd.DataFrame:
    validate_columns(df, list(feature_columns) + [target_column, timestamp_column, symbol_column])

    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    out = out.sort_values([symbol_column, timestamp_column]).reset_index(drop=True)

    out = out[out[target_column].isin([0, 1])].copy()

    if drop_rows_with_missing_features:
        before = len(out)
        out = out.dropna(subset=list(feature_columns) + [target_column]).copy()
        dropped = before - len(out)
        if dropped > 0:
            LOGGER.info(
                "Dropped %d rows with missing required features/target for %s",
                dropped,
                target_column,
            )

    if out.empty:
        raise ValueError(f"No rows remain after filtering/cleaning for {target_column}")

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


def build_model(
    model_name: str,
    numeric_features: Sequence[str],
    *,
    enable_variance_threshold: bool,
    variance_threshold: float,
) -> Pipeline:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if enable_variance_threshold:
        numeric_steps.append(("variance_filter", VarianceThreshold(threshold=variance_threshold)))

    numeric_steps.append(("scaler", StandardScaler()))
    numeric_preproc = Pipeline(steps=numeric_steps)

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
    feature_count_requested: int,
    feature_count_used: int,
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
        feature_count_requested=int(feature_count_requested),
        feature_count_used=int(feature_count_used),
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


def _estimate_feature_count_used(model: Pipeline, requested_feature_count: int) -> int:
    try:
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transformers_[0][1]
        if "variance_filter" in transformed.named_steps:
            selector = transformed.named_steps["variance_filter"]
            return int(selector.get_support().sum())
    except Exception:
        pass
    return int(requested_feature_count)


def _extract_post_variance_feature_names(
    model: Pipeline,
    feature_columns: Sequence[str],
) -> list[str]:
    feature_names = list(feature_columns)
    try:
        preprocessor = model.named_steps["preprocessor"]
        numeric_pipeline = preprocessor.transformers_[0][1]
        if "variance_filter" in numeric_pipeline.named_steps:
            selector = numeric_pipeline.named_steps["variance_filter"]
            mask = selector.get_support()
            feature_names = [name for name, keep in zip(feature_columns, mask) if keep]
    except Exception:
        pass
    return feature_names


def build_feature_importance_frame(
    *,
    model: Pipeline,
    feature_columns: Sequence[str],
    horizon: str,
    target_column: str,
) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            columns=["feature", "importance", "horizon", "target_column", "importance_type", "rank"]
        )

    feature_names = _extract_post_variance_feature_names(model, feature_columns)
    importances = np.asarray(estimator.feature_importances_, dtype=float)

    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)

    fi_df["horizon"] = horizon
    fi_df["target_column"] = target_column
    fi_df["importance_type"] = "model_feature_importance"
    fi_df["rank"] = np.arange(1, len(fi_df) + 1)
    return fi_df


def build_permutation_importance_frame(
    *,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: Sequence[str],
    horizon: str,
    target_column: str,
    n_repeats: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="average_precision",
    )
    pi_df = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)

    pi_df["horizon"] = horizon
    pi_df["target_column"] = target_column
    pi_df["importance_type"] = "permutation_average_precision"
    pi_df["rank"] = np.arange(1, len(pi_df) + 1)
    return pi_df


def train_one_horizon(
    *,
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    horizon: str,
    explicit_target_column: str | None,
    feature_columns: Sequence[str],
    timestamp_column: str,
    symbol_column: str,
    train_fraction: float,
    model_name: str,
    drop_rows_with_missing_features: bool,
    enable_variance_threshold: bool,
    variance_threshold: float,
    feature_importance_table_prefix: str,
    top_feature_count: int,
    enable_permutation_importance: bool,
    permutation_importance_table_prefix: str,
    permutation_repeats: int,
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
        drop_rows_with_missing_features=drop_rows_with_missing_features,
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

    model = build_model(
        model_name,
        feature_columns,
        enable_variance_threshold=enable_variance_threshold,
        variance_threshold=variance_threshold,
    )
    LOGGER.info("Training model=%s horizon=%s", model_name, horizon)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    feature_count_used = _estimate_feature_count_used(model, len(feature_columns))

    result = evaluate(
        model_name=model_name,
        horizon=horizon,
        target_column=target_column,
        feature_count_requested=len(feature_columns),
        feature_count_used=feature_count_used,
        y_train=y_train,
        y_test=y_test,
        y_prob=y_prob,
    )

    LOGGER.info(
        "Metrics horizon=%s: roc_auc=%s pr_auc=%s brier=%s best_threshold=%s f1=%s precision=%s recall=%s features=%d/%d",
        horizon,
        result.roc_auc,
        result.pr_auc,
        result.brier_score,
        result.best_threshold_f1,
        result.f1_at_best_f1,
        result.precision_at_best_f1,
        result.recall_at_best_f1,
        result.feature_count_used,
        result.feature_count_requested,
    )

    if model_name == "hgb":
        fi_df = build_feature_importance_frame(
            model=model,
            feature_columns=feature_columns,
            horizon=horizon,
            target_column=target_column,
        )
        fi_table = f"{feature_importance_table_prefix}_{horizon}"
        write_table(conn, fi_df, fi_table, "replace")
        LOGGER.info(
            "Top %d features (%s):\n%s",
            top_feature_count,
            horizon,
            fi_df.head(top_feature_count).to_string(index=False),
        )

        if enable_permutation_importance:
            pi_df = build_permutation_importance_frame(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_columns=feature_columns,
                horizon=horizon,
                target_column=target_column,
                n_repeats=permutation_repeats,
            )
            pi_table = f"{permutation_importance_table_prefix}_{horizon}"
            write_table(conn, pi_df, pi_table, "replace")
            LOGGER.info(
                "Top %d permutation features (%s):\n%s",
                top_feature_count,
                horizon,
                pi_df.head(top_feature_count).to_string(index=False),
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
                    conn=conn,
                    df=df,
                    horizon=horizon,
                    explicit_target_column=None,
                    feature_columns=args.feature_columns,
                    timestamp_column=args.timestamp_column,
                    symbol_column=args.symbol_column,
                    train_fraction=args.train_fraction,
                    model_name=args.model,
                    drop_rows_with_missing_features=args.drop_rows_with_missing_features,
                    enable_variance_threshold=args.enable_variance_threshold,
                    variance_threshold=args.variance_threshold,
                    feature_importance_table_prefix=args.feature_importance_table_prefix,
                    top_feature_count=args.top_feature_count,
                    enable_permutation_importance=args.enable_permutation_importance,
                    permutation_importance_table_prefix=args.permutation_importance_table_prefix,
                    permutation_repeats=args.permutation_repeats,
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
                conn=conn,
                df=df,
                horizon=horizon,
                explicit_target_column=args.target_column,
                feature_columns=args.feature_columns,
                timestamp_column=args.timestamp_column,
                symbol_column=args.symbol_column,
                train_fraction=args.train_fraction,
                model_name=args.model,
                drop_rows_with_missing_features=args.drop_rows_with_missing_features,
                enable_variance_threshold=args.enable_variance_threshold,
                variance_threshold=args.variance_threshold,
                feature_importance_table_prefix=args.feature_importance_table_prefix,
                top_feature_count=args.top_feature_count,
                enable_permutation_importance=args.enable_permutation_importance,
                permutation_importance_table_prefix=args.permutation_importance_table_prefix,
                permutation_repeats=args.permutation_repeats,
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