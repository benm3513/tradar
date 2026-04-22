from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class PredictorConfig:
    mode: str = "heuristic"  # heuristic | db_latest
    predictions_table: Optional[str] = None
    symbol_column: str = "symbol"
    timestamp_column: str = "timestamp"
    prob_column: str = "pred_prob"
    score_column: Optional[str] = None
    prediction_source: str = "direct"  # direct | ensemble | 6h | 24h | 72h
    max_prediction_age_minutes: Optional[float] = 180.0


class LivePredictor:
    """Phase 5.0 live predictor.

    Requirements satisfied:
    - does NOT load real model artifacts yet
    - can simulate predictions from live features
    - can optionally read latest predictions from SQLite
    - returns a normalized per-symbol payload that MLStrategy can rank/sort

    Output payload per symbol:
    {
        "symbol": ...,
        "prob": float,
        "score": float,
        "entry_score": float,
        "pred_prob": float,
        "prob_percentile_rank": float | None,
        "rolling_volatility_24h": float | None,
        "target_time_to_peak_seconds_24h": float | None,
        ...
    }
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = dict(cfg or {})
        self.mode = str(self.cfg.get("mode", "heuristic")).strip().lower()
        self.predictions_table = self.cfg.get("predictions_table")
        self.symbol_column = self.cfg.get("symbol_column", "symbol")
        self.timestamp_column = self.cfg.get("timestamp_column", "timestamp")
        self.prob_column = self.cfg.get("prob_column", "pred_prob")
        self.score_column = self.cfg.get("score_column")
        self.prediction_source = str(self.cfg.get("prediction_source", "direct")).strip().lower()
        self.max_prediction_age_minutes = self.cfg.get("max_prediction_age_minutes", 180.0)

    # -------------------------
    # public API
    # -------------------------

    def predict(self, feature_frame: pd.DataFrame, ctx: Any = None) -> Dict[str, Dict[str, Any]]:
        if feature_frame is None or feature_frame.empty:
            return {}

        if self.mode == "db_latest":
            db_payload = self._predict_from_db_latest(feature_frame, ctx)
            if db_payload:
                return db_payload
            # safe fallback if DB mode is enabled but no usable rows exist
            return self._predict_heuristic(feature_frame)

        return self._predict_heuristic(feature_frame)

    # -------------------------
    # heuristic mode
    # -------------------------

    def _predict_heuristic(self, feature_frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        df = feature_frame.copy()

        if "prob_percentile_rank" not in df.columns and "prob_proxy" in df.columns:
            df["prob_percentile_rank"] = df["prob_proxy"].rank(method="average", pct=True)

        outputs: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            symbol = row.get("symbol")
            if not symbol:
                continue

            prob, score = self._score_row_heuristic(row)

            payload = row.to_dict()
            payload["symbol"] = symbol
            payload["prob"] = prob
            payload["pred_prob"] = prob
            payload["score"] = score
            payload["entry_score"] = score
            payload["prediction_source"] = "heuristic_live"
            payload["model_name"] = "phase5_heuristic"
            payload["prob_percentile_rank"] = _safe_float(
                payload.get("prob_percentile_rank"),
                default=None if payload.get("prob_percentile_rank") is None else 0.0,
            )

            outputs[str(symbol)] = payload

        return outputs

    def _score_row_heuristic(self, row: pd.Series) -> tuple[float, float]:
        prob_proxy = _safe_float(row.get("prob_proxy"), 0.0)
        ret_6h = _safe_float(row.get("return_6h"), 0.0)
        ret_24h = _safe_float(row.get("return_24h"), 0.0)
        accel = _safe_float(row.get("momentum_accel_6h_vs_24h"), 0.0)
        vol_z = _safe_float(row.get("volume_zscore_24h"), 0.0)
        vol_ratio = _safe_float(row.get("volume_spike_ratio_7d"), 0.0)
        roll_vol = _safe_float(row.get("rolling_volatility_24h"), 0.0)
        range_pct = _safe_float(row.get("range_pct_24h"), 0.0)
        drawup = _safe_float(row.get("drawup_from_recent_low_24h"), 0.0)
        trend_strength = _safe_float(row.get("market_trend_strength_24h"), _safe_float(row.get("trend_strength_local_24h"), 0.0))
        risk_off = _safe_float(row.get("market_risk_off_score"), 0.5)
        dispersion = _safe_float(row.get("market_dispersion_24h"), 0.0)
        volume_regime = _safe_float(row.get("market_volume_regime_24h"), vol_z)
        percentile = _safe_float(row.get("prob_percentile_rank"), 0.5)
        time_to_peak_h = _safe_float(
            row.get("predicted_time_to_peak_hours", row.get("time_to_peak_hours", 24.0)),
            24.0,
        )

        raw = (
            1.20 * prob_proxy
            + 0.85 * accel
            + 0.70 * ret_6h
            + 0.35 * ret_24h
            + 0.25 * vol_z
            + 0.25 * volume_regime
            + 0.20 * vol_ratio
            + 0.20 * drawup
            + 0.15 * range_pct
            + 0.20 * percentile
            + 0.15 * trend_strength
            - 0.60 * risk_off
            - 0.35 * dispersion
            - 0.30 * roll_vol
            - 0.08 * min(time_to_peak_h / 24.0, 1.0)
        )

        prob = _bounded(_sigmoid(3.0 * raw), 0.0, 1.0)

        # ranking score: similar spirit to replay composite ranking,
        # but generated from live features for Phase 5.0
        score = (
            1.00 * prob
            + 0.25 * percentile
            + 0.20 * trend_strength
            + 0.15 * max(0.0, vol_z)
            + 0.10 * max(0.0, vol_ratio - 1.0)
            - 0.25 * risk_off
            - 0.10 * dispersion
        )

        return float(prob), float(score)

    # -------------------------
    # db-latest mode
    # -------------------------

    def _predict_from_db_latest(self, feature_frame: pd.DataFrame, ctx: Any = None) -> Dict[str, Dict[str, Any]]:
        conn = self._get_connection(ctx)
        if conn is None or not self.predictions_table:
            return {}

        symbol_col = self.symbol_column
        ts_col = self.timestamp_column

        feature_df = feature_frame.copy()
        symbols = [str(x) for x in feature_df["symbol"].dropna().astype(str).unique().tolist()]
        if not symbols:
            return {}

        db_df = self._load_latest_predictions(conn, symbols)
        if db_df.empty:
            return {}

        merged = feature_df.merge(db_df, on="symbol", how="left", suffixes=("", "_db"))
        merged = merged.dropna(subset=["_resolved_prob", "_resolved_score"])

        if merged.empty:
            return {}

        outputs: Dict[str, Dict[str, Any]] = {}
        for _, row in merged.iterrows():
            symbol = str(row["symbol"])
            payload = row.to_dict()
            payload["prob"] = _safe_float(row["_resolved_prob"], 0.0)
            payload["pred_prob"] = _safe_float(row["_resolved_prob"], 0.0)
            payload["score"] = _safe_float(row["_resolved_score"], 0.0)
            payload["entry_score"] = _safe_float(row["_resolved_score"], 0.0)
            payload["prediction_source"] = f"db_latest:{self.prediction_source}"
            payload["model_name"] = payload.get("model_name")
            outputs[symbol] = payload

        # percentile context should remain live cross-sectional
        if outputs:
            probs = pd.Series({sym: payload["prob"] for sym, payload in outputs.items()})
            pct = probs.rank(method="average", pct=True).to_dict()
            for sym, payload in outputs.items():
                payload["prob_percentile_rank"] = _safe_float(pct.get(sym), 0.5)

        return outputs

    def _load_latest_predictions(self, conn: sqlite3.Connection, symbols: Iterable[str]) -> pd.DataFrame:
        query = f'SELECT * FROM "{self.predictions_table}"'
        try:
            df = pd.read_sql_query(query, conn)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        required = [self.symbol_column]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        if self.timestamp_column in df.columns:
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], utc=True, errors="coerce")

        prob_col, score_col = self._resolve_prediction_columns(df)

        if prob_col is None or score_col is None:
            return pd.DataFrame()

        df["_resolved_prob"] = pd.to_numeric(df[prob_col], errors="coerce")
        df["_resolved_score"] = pd.to_numeric(df[score_col], errors="coerce")
        df = df[df[self.symbol_column].astype(str).isin([str(s) for s in symbols])].copy()
        df = df.dropna(subset=[self.symbol_column, "_resolved_prob", "_resolved_score"])

        if df.empty:
            return df

        if self.timestamp_column in df.columns:
            df = df.sort_values([self.symbol_column, self.timestamp_column])
            latest = df.groupby(self.symbol_column, as_index=False).tail(1).copy()
            if self.max_prediction_age_minutes is not None:
                max_age = pd.Timedelta(minutes=float(self.max_prediction_age_minutes))
                now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow().tz_convert("UTC")
                latest = latest[
                    latest[self.timestamp_column].notna()
                    & ((now - latest[self.timestamp_column]) <= max_age)
                ].copy()
        else:
            latest = df.groupby(self.symbol_column, as_index=False).tail(1).copy()

        latest = latest.rename(columns={self.symbol_column: "symbol"})
        return latest

    def _resolve_prediction_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        source = self.prediction_source
        explicit_score = self.score_column

        if source == "ensemble":
            prob_candidates = ["prob_ensemble", "ensemble_score", self.prob_column]
            score_candidates = [explicit_score] if explicit_score else ["ensemble_score", "prob_ensemble", self.prob_column]
        elif source in {"6h", "24h", "72h"}:
            horizon_prob = f"prob_{source}"
            prob_candidates = [horizon_prob, self.prob_column]
            score_candidates = [explicit_score] if explicit_score else [horizon_prob, self.prob_column]
        else:
            prob_candidates = [self.prob_column]
            score_candidates = [explicit_score] if explicit_score else [self.prob_column]

        prob_col = next((c for c in prob_candidates if c and c in df.columns), None)
        score_col = next((c for c in score_candidates if c and c in df.columns), None)
        return prob_col, score_col

    def _get_connection(self, ctx: Any = None) -> Optional[sqlite3.Connection]:
        # First preference: runtime store connection from ctx
        if ctx is not None:
            store = getattr(ctx, "store", None)
            if store is not None:
                for attr in ("conn", "_conn", "connection"):
                    conn = getattr(store, attr, None)
                    if isinstance(conn, sqlite3.Connection):
                        return conn
                db_path = getattr(store, "db_path", None)
                if db_path:
                    try:
                        return sqlite3.connect(db_path)
                    except Exception:
                        pass

        # Second preference: explicit db_path in config
        db_path = self.cfg.get("db_path")
        if db_path:
            try:
                return sqlite3.connect(db_path)
            except Exception:
                return None

        return None


def predict_from_features(feature_frame: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None, ctx: Any = None) -> Dict[str, Dict[str, Any]]:
    predictor = LivePredictor(cfg or {})
    return predictor.predict(feature_frame=feature_frame, ctx=ctx)
