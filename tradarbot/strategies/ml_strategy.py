from __future__ import annotations

import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd

from tradarbot.core.events import OrderIntent
from tradarbot.execution.order_router import OrderRouter
from tradarbot.ml.live_features import build_live_feature_frame
from tradarbot.ml.live_predictor import LivePredictor
from tradarbot.ml.calibration import ProbabilityCalibrator
from tradarbot.ml.signal_throttle import SignalThrottle
from tradarbot.ml.shadow_eval import (
    ShadowDecisionRecord,
    ShadowPredictionRecord,
    ShadowSignalRecord,
    record_shadow_decision,
    record_shadow_prediction,
    record_shadow_signal,
)
from scripts.replay_ml_strategy import (
    build_runtime_args,
    compute_size_multipliers,
    filter_ranked_candidates,
)

log = logging.getLogger("tradarbot.ml_strategy")


class MLStrategy:
    """
    Phase 5.0/5.3 live ML strategy with replay-parity filtering/ranking/sizing.

    Debug upgrade:
    - Adds ML_DEBUG logs around each early-return checkpoint in on_candle.
    - Does not change thresholds, ranking, sizing, or order logic.
    """

    name = "ml_strategy"

    def __init__(self, cfg: Dict):
        self.cfg = dict(cfg or {})
        raw_mode = str(self.cfg.get("mode") or "").strip().lower()
        # Backward compatibility: older config used ml_live.mode as predictor mode
        # (heuristic/db_latest/artifacts). Treat those as paper execution modes and
        # leave predictor_mode to LivePredictor.
        if raw_mode in {"off", "shadow", "paper", "live"}:
            self.ml_mode = raw_mode
        elif self.cfg.get("enabled", True) is False:
            self.ml_mode = "off"
        else:
            self.ml_mode = "paper"
            if raw_mode and "predictor_mode" not in self.cfg:
                self.cfg["predictor_mode"] = raw_mode
        self.predictor = LivePredictor(self.cfg)
        self.calibrator = ProbabilityCalibrator(self.cfg)
        self.signal_throttle = SignalThrottle(self.cfg)
        self.router = None

        self.last_signal_ts: Dict[str, int] = {}
        self.last_eval_ts_by_symbol: Dict[str, int] = {}
        self.last_inference_bucket_ts: int | None = None
        self.last_inference_ts_ms: int | None = None
        self.price_history = defaultdict(list)

        self.lookback_bars = int(
            self.cfg.get("feature_lookback_bars", self.cfg.get("warmup_candles", 200))
        )
        self.evaluation_interval_s = int(self.cfg.get("evaluation_interval_s", 1))
        self.throttle_inference = bool(self.cfg.get("throttle_inference", True))
        self.inference_interval_s = int(
            self.cfg.get(
                "inference_interval_s",
                self.cfg.get("min_inference_interval_seconds", max(60, self.evaluation_interval_s)),
            )
        )
        self.inference_interval_s = max(1, self.inference_interval_s)
        self.inference_on_candle_close_only = bool(self.cfg.get("inference_on_candle_close_only", False))
        self.candle_interval_s = int(
            self.cfg.get("candle_interval_s", self.cfg.get("feature_interval_s", 3600))
        )
        self.min_ready_bars = int(
            self.cfg.get("min_ready_bars", self.cfg.get("warmup_candles", self.lookback_bars))
        )
        self.use_centralized_feature_state = bool(self.cfg.get("use_centralized_feature_state", True))

        self.order_notional_buffer = float(
            self.cfg.get("order_notional_buffer", self.cfg.get("live_order_notional_buffer", 0.99))
            or 0.99
        )
        if self.order_notional_buffer <= 0.0 or self.order_notional_buffer > 1.0:
            self.order_notional_buffer = 0.99

        self.runtime_args = self._build_runtime_args()

        log.info(
            "MLStrategy initialized ml_mode=%s predictor_mode=%s min_ready_bars=%s lookback_bars=%s centralized_feature_state=%s inference_interval_s=%s throttle_inference=%s",
            self.ml_mode,
            getattr(self.predictor, "mode", self.cfg.get("predictor_mode", self.cfg.get("mode"))),
            self.min_ready_bars,
            self.lookback_bars,
            self.use_centralized_feature_state,
            self.inference_interval_s,
            self.throttle_inference,
        )

    def on_listing(self, ev, ctx):
        return []

    def on_candle(self, e, ctx):
        if self.ml_mode == "off":
            log.debug("ML_MODE_OFF symbol=%s ts_ms=%s", getattr(e, "symbol", None), getattr(e, "ts_ms", None))
            return []

        self._record_candle(e)
        self._sync_state_symbol(e.symbol, ctx)

        log.info(
            "ML_DEBUG on_candle symbol=%s ts_ms=%s local_bars=%s min_ready_bars=%s lookback_bars=%s",
            e.symbol,
            e.ts_ms,
            len(self.price_history.get(e.symbol, [])),
            self.min_ready_bars,
            self.lookback_bars,
        )

        last_eval = self.last_eval_ts_by_symbol.get(e.symbol)
        if last_eval is not None:
            min_gap_ms = max(1, self.evaluation_interval_s) * 1000
            gap_ms = int(e.ts_ms) - int(last_eval)
            if gap_ms < min_gap_ms:
                log.info(
                    "ML_DEBUG return=eval_interval symbol=%s gap_ms=%s min_gap_ms=%s",
                    e.symbol,
                    gap_ms,
                    min_gap_ms,
                )
                return []

        symbols = sorted(set(getattr(ctx.state, "active_symbols", set()) or {e.symbol}))
        if e.symbol not in symbols:
            symbols.append(e.symbol)

        log.debug("ML_DEBUG active_symbols_before_filter=%s", symbols)

        symbols = self._filter_tradable_symbols(symbols, ctx)
        log.debug("ML_DEBUG symbols_after_filter=%s", symbols)

        if not symbols:
            log.debug("ML_DEBUG return=no_symbols_after_filter")
            return []

        ready_symbols = self._ready_symbols(symbols, ctx)
        log.info(
            "ML_DEBUG ready_symbols=%s min_ready_bars=%s local_counts=%s",
            ready_symbols,
            self.min_ready_bars,
            {s: len(self.price_history.get(s, [])) for s in symbols},
        )

        if not ready_symbols:
            fs = self._get_feature_state(ctx)
            fs_symbols = []
            fs_ready = []
            try:
                if fs is not None and hasattr(fs, "symbols"):
                    fs_symbols = fs.symbols()
                if fs is not None and hasattr(fs, "ready_symbols"):
                    fs_ready = fs.ready_symbols()
            except Exception as ex:
                log.debug("ML_DEBUG feature_state_inspect_failed=%s", ex)

            log.info(
                "ML_DEBUG return=no_ready_symbols feature_state_present=%s feature_state_symbols=%s feature_state_ready=%s",
                fs is not None,
                fs_symbols,
                fs_ready,
            )
            return []

        if not self._should_run_inference(e, ctx, ready_symbols):
            return []

        feature_df = self._build_feature_frame(ready_symbols, ctx)
        log.info(
            "ML_DEBUG feature_df rows=%s cols=%s",
            0 if feature_df is None else len(feature_df),
            [] if feature_df is None or feature_df.empty else list(feature_df.columns),
        )

        if feature_df.empty:
            log.debug("ML_DEBUG return=empty_feature_df ready_symbols=%s", ready_symbols)
            return []

        prediction_map = self.predictor.predict(feature_df, ctx=ctx)
        prediction_map = self._calibrate_predictions(prediction_map)
        log.info(
            "ML_DEBUG prediction_map symbols=%s",
            list(prediction_map.keys()) if prediction_map else [],
        )

        if prediction_map:
            for symbol, payload in prediction_map.items():
                log.info(
                    "ML_DEBUG prediction symbol=%s prob=%s pred_prob=%s score=%s source=%s model=%s",
                    symbol,
                    payload.get("prob"),
                    payload.get("pred_prob"),
                    payload.get("score"),
                    payload.get("prediction_source"),
                    payload.get("model_name"),
                )

        self._persist_shadow_predictions(ctx, prediction_map, int(e.ts_ms))

        candidate_df = self._merge_predictions(feature_df, prediction_map)
        log.info(
            "ML_DEBUG candidate_df rows=%s cols=%s",
            0 if candidate_df is None else len(candidate_df),
            [] if candidate_df is None or candidate_df.empty else list(candidate_df.columns),
        )

        if candidate_df.empty:
            log.info(
                "ML_DEBUG return=empty_candidate_df feature_symbols=%s",
                feature_df["symbol"].astype(str).tolist() if "symbol" in feature_df.columns else [],
            )
            return []

        args = self.runtime_args
        ranked = self._filter_ranked(candidate_df, args)
        log.info(
            "ML_DEBUG ranked rows=%s symbols=%s",
            0 if ranked is None else len(ranked),
            [] if ranked is None or ranked.empty or "symbol" not in ranked.columns else ranked["symbol"].astype(str).tolist(),
        )

        self._update_state_with_rankings(ctx, ranked, int(e.ts_ms))
        self._persist_shadow_candidate_decisions(ctx, candidate_df, ranked, int(e.ts_ms))

        if ranked.empty:
            self.last_eval_ts_by_symbol[e.symbol] = int(e.ts_ms)
            log.info(
                "ML_DEBUG return=empty_ranked prob_threshold=%s min_prob_percentile=%s top_n=%s",
                getattr(args, "prob_threshold", None),
                getattr(args, "min_prob_percentile", None),
                getattr(args, "top_n", None),
            )
            return []

        intents = self._build_entry_intents(ranked, args, ctx, ts_ms=int(e.ts_ms))
        if self.ml_mode == "shadow":
            self._persist_shadow_signals(ctx, ranked, intents, int(e.ts_ms))
            self._increment_state_counter(ctx, "ml_shadow_blocked_execution_count", len(intents))
            for intent in intents:
                log.info(
                    "ML_SHADOW_BLOCKED_EXECUTION symbol=%s side=%s qty=%.8f limit_px=%.8f",
                    intent.symbol,
                    intent.side,
                    float(intent.qty),
                    float(intent.limit_px),
                )
            self.last_eval_ts_by_symbol[e.symbol] = int(e.ts_ms)
            log.debug("ML_DEBUG shadow_intents_blocked=%s", len(intents))
            return []

        self.last_eval_ts_by_symbol[e.symbol] = int(e.ts_ms)
        log.debug("ML_DEBUG intents_count=%s", len(intents))
        return intents


    def _should_run_inference(self, e, ctx, ready_symbols: List[str]) -> bool:
        """Gate expensive live ML inference before feature/predict/rank/persist work.

        Phase 5.8.5 initially suppressed duplicate signals after prediction. This
        method moves the gate earlier so shadow mode does not burn CPU, logs, and
        SQLite writes on every symbol/tick update. The default is one inference
        pass per inference_interval_s bucket, shared across symbols.
        """
        if not self.throttle_inference:
            return True

        ts_ms = int(getattr(e, "ts_ms", 0) or 0)
        interval_ms = max(1, int(self.inference_interval_s)) * 1000

        if self.inference_on_candle_close_only:
            candle_ms = max(1, int(self.candle_interval_s or self.inference_interval_s)) * 1000
            if candle_ms > 0 and ts_ms % candle_ms != 0:
                log.info(
                    "ML_INFERENCE_SKIPPED_CANDLE_CLOSE_ONLY symbol=%s ts_ms=%s candle_interval_s=%s ready_symbols=%s",
                    getattr(e, "symbol", None),
                    ts_ms,
                    self.candle_interval_s,
                    ready_symbols,
                )
                self._increment_state_counter(ctx, "ml_inference_skipped_count", 1)
                return False

        bucket_ts = (ts_ms // interval_ms) * interval_ms
        if self.last_inference_bucket_ts == bucket_ts:
            log.info(
                "ML_INFERENCE_SKIPPED_INTERVAL symbol=%s ts_ms=%s bucket_ts=%s inference_interval_s=%s ready_symbols=%s",
                getattr(e, "symbol", None),
                ts_ms,
                bucket_ts,
                self.inference_interval_s,
                ready_symbols,
            )
            self._increment_state_counter(ctx, "ml_inference_skipped_count", 1)
            return False

        if self.last_inference_ts_ms is not None and ts_ms - int(self.last_inference_ts_ms) < interval_ms:
            gap_ms = ts_ms - int(self.last_inference_ts_ms)
            log.info(
                "ML_INFERENCE_SKIPPED_MIN_GAP symbol=%s ts_ms=%s gap_ms=%s min_gap_ms=%s ready_symbols=%s",
                getattr(e, "symbol", None),
                ts_ms,
                gap_ms,
                interval_ms,
                ready_symbols,
            )
            self._increment_state_counter(ctx, "ml_inference_skipped_count", 1)
            return False

        self.last_inference_bucket_ts = bucket_ts
        self.last_inference_ts_ms = ts_ms
        log.info(
            "ML_INFERENCE_ALLOWED symbol=%s ts_ms=%s bucket_ts=%s inference_interval_s=%s ready_symbols=%s",
            getattr(e, "symbol", None),
            ts_ms,
            bucket_ts,
            self.inference_interval_s,
            ready_symbols,
        )
        return True


    def _calibrate_predictions(self, prediction_map: Dict[str, Dict]) -> Dict[str, Dict]:
        if not prediction_map:
            return {}
        out: Dict[str, Dict] = {}
        for symbol, payload in prediction_map.items():
            try:
                out[str(symbol)] = self.calibrator.calibrate_payload(str(symbol), dict(payload or {}))
            except Exception:
                log.exception("ML_PROBABILITY_CALIBRATION_FAILED symbol=%s", symbol)
                out[str(symbol)] = dict(payload or {})
        return out

    def _increment_state_counter(self, ctx, name: str, amount: int = 1) -> None:
        state = getattr(ctx, "state", None)
        if state is None:
            return
        try:
            setattr(state, name, int(getattr(state, name, 0) or 0) + int(amount))
        except Exception:
            pass

    def _persist_shadow_predictions(self, ctx, prediction_map: Dict[str, Dict], ts_ms: int) -> None:
        if self.ml_mode not in {"shadow", "paper", "live"} or not prediction_map:
            return
        store = getattr(ctx, "store", None)
        for symbol, payload in prediction_map.items():
            try:
                record = ShadowPredictionRecord.from_payload(ts_ms=ts_ms, symbol=symbol, payload=dict(payload or {}), mode=self.ml_mode)
                record_shadow_prediction(store, record)
                self._increment_state_counter(ctx, "ml_shadow_prediction_count", 1)
                log.info(
                    "ML_SHADOW_PREDICTION mode=%s symbol=%s prob=%s score=%s source=%s model=%s",
                    self.ml_mode,
                    symbol,
                    payload.get("prob", payload.get("pred_prob")),
                    payload.get("score", payload.get("entry_score")),
                    payload.get("prediction_source"),
                    payload.get("model_name"),
                )
            except Exception:
                log.exception("ML_SHADOW_PREDICTION_PERSIST_FAILED symbol=%s", symbol)

    def _persist_shadow_candidate_decisions(self, ctx, candidate_df: pd.DataFrame, ranked: pd.DataFrame, ts_ms: int) -> None:
        if self.ml_mode not in {"shadow", "paper", "live"} or candidate_df is None or candidate_df.empty:
            return
        store = getattr(ctx, "store", None)
        ranked_symbols = [] if ranked is None or ranked.empty or "symbol" not in ranked.columns else ranked["symbol"].astype(str).tolist()
        rank_map = {symbol: idx + 1 for idx, symbol in enumerate(ranked_symbols)}
        for _, row in candidate_df.iterrows():
            symbol = str(row.get("symbol"))
            accepted = symbol in rank_map
            reject_reason = None if accepted else "filtered_by_replay_compatible_path"
            record = ShadowDecisionRecord(
                ts_ms=ts_ms,
                symbol=symbol,
                mode=self.ml_mode,
                accepted=accepted,
                reject_reason=reject_reason,
                would_trade=False,
                prob=row.get("prob"),
                pred_prob=row.get("pred_prob", row.get("prob")),
                score=row.get("score"),
                entry_score=row.get("entry_score"),
                prob_percentile_rank=row.get("prob_percentile_rank"),
                rolling_volatility_24h=row.get("rolling_volatility_24h"),
                predicted_time_to_peak_hours=row.get("predicted_time_to_peak_hours"),
                market_risk_off_score=row.get("market_risk_off_score"),
                prediction_source=row.get("prediction_source"),
                model_name=row.get("model_name"),
                regime_size_multiplier=row.get("_regime_size_multiplier"),
                top_n_rank=rank_map.get(symbol),
                payload=row.to_dict(),
            )
            record_shadow_decision(store, record)
            self._increment_state_counter(ctx, "ml_shadow_candidate_count", 1)
            log.info(
                "ML_SHADOW_CANDIDATE mode=%s symbol=%s accepted=%s reject_reason=%s prob=%s score=%s rank=%s",
                self.ml_mode,
                symbol,
                accepted,
                reject_reason,
                row.get("pred_prob", row.get("prob")),
                row.get("entry_score", row.get("score")),
                rank_map.get(symbol),
            )

    def _persist_shadow_signals(self, ctx, ranked: pd.DataFrame, intents: List[OrderIntent], ts_ms: int) -> None:
        if self.ml_mode != "shadow" or ranked is None or ranked.empty:
            return
        store = getattr(ctx, "store", None)
        row_by_symbol = {str(row.get("symbol")): row for _, row in ranked.iterrows()}
        for intent in intents:
            row = row_by_symbol.get(str(intent.symbol), {})
            notional = float(intent.qty) * float(intent.limit_px)
            record_shadow_signal(
                store,
                ShadowSignalRecord(
                    ts_ms=ts_ms,
                    symbol=str(intent.symbol),
                    mode=self.ml_mode,
                    action="would_buy",
                    side=str(intent.side),
                    qty=float(intent.qty),
                    limit_px=float(intent.limit_px),
                    notional_usd=notional,
                    blocked_execution=True,
                    prediction_source=row.get("prediction_source") if hasattr(row, "get") else None,
                    model_name=row.get("model_name") if hasattr(row, "get") else None,
                    prob=row.get("pred_prob", row.get("prob")) if hasattr(row, "get") else None,
                    score=row.get("score") if hasattr(row, "get") else None,
                    entry_score=row.get("entry_score") if hasattr(row, "get") else None,
                    payload=row.to_dict() if hasattr(row, "to_dict") else {},
                ),
            )
            record_shadow_decision(
                store,
                ShadowDecisionRecord(
                    ts_ms=ts_ms,
                    symbol=str(intent.symbol),
                    mode=self.ml_mode,
                    accepted=True,
                    reject_reason=None,
                    would_trade=True,
                    would_side=str(intent.side),
                    would_qty=float(intent.qty),
                    would_limit_px=float(intent.limit_px),
                    would_notional_usd=notional,
                    prob=row.get("prob") if hasattr(row, "get") else None,
                    pred_prob=row.get("pred_prob", row.get("prob")) if hasattr(row, "get") else None,
                    score=row.get("score") if hasattr(row, "get") else None,
                    entry_score=row.get("entry_score") if hasattr(row, "get") else None,
                    prob_percentile_rank=row.get("prob_percentile_rank") if hasattr(row, "get") else None,
                    rolling_volatility_24h=row.get("rolling_volatility_24h") if hasattr(row, "get") else None,
                    predicted_time_to_peak_hours=row.get("predicted_time_to_peak_hours") if hasattr(row, "get") else None,
                    market_risk_off_score=row.get("market_risk_off_score") if hasattr(row, "get") else None,
                    prediction_source=row.get("prediction_source") if hasattr(row, "get") else None,
                    model_name=row.get("model_name") if hasattr(row, "get") else None,
                    regime_size_multiplier=row.get("_regime_size_multiplier") if hasattr(row, "get") else None,
                    total_size_multiplier=row.get("total_size_multiplier") if hasattr(row, "get") else None,
                    payload=row.to_dict() if hasattr(row, "to_dict") else {},
                ),
            )
            self._increment_state_counter(ctx, "ml_shadow_signal_count", 1)
            self._increment_state_counter(ctx, "ml_shadow_would_trade_count", 1)
            log.info(
                "ML_SHADOW_SIGNAL symbol=%s side=%s qty=%.8f limit_px=%.8f notional=%.2f",
                intent.symbol,
                intent.side,
                float(intent.qty),
                float(intent.limit_px),
                notional,
            )

    def _record_candle(self, e):
        arr = self.price_history[e.symbol]
        arr.append(e)
        max_keep = max(self.lookback_bars + 5, 512)
        if len(arr) > max_keep:
            del arr[:-max_keep]

    def _sync_state_symbol(self, symbol: str, ctx) -> None:
        if hasattr(ctx.state, "get_ml_symbol_state"):
            ctx.state.get_ml_symbol_state(symbol)

    def _filter_tradable_symbols(self, symbols: List[str], ctx) -> List[str]:
        root_cfg = getattr(ctx, "cfg", {}) if ctx is not None else {}
        exec_cfg = root_cfg.get("execution_live", {}) if isinstance(root_cfg, dict) else {}
        ml_cfg = root_cfg.get("ml_live", {}) if isinstance(root_cfg, dict) else {}

        raw = (
            self.cfg.get("tradable_symbols")
            or ml_cfg.get("tradable_symbols")
            or exec_cfg.get("tradable_symbols")
            or exec_cfg.get("allowlist")
        )
        if not raw:
            return list(symbols)

        allowed = {str(s).upper().replace("/", "") for s in raw}
        out = []
        for symbol in symbols:
            normalized = str(symbol).upper().replace("/", "")
            alt = normalized[:-1] if normalized.endswith("T") else normalized
            if normalized in allowed or alt in allowed:
                out.append(symbol)
        skipped = sorted(set(symbols) - set(out))
        if skipped:
            if getattr(self, "_last_filter_log", None) != tuple(skipped):
                log.debug("ML_TRADABLE_FILTER skipped=%s allowed=%s", skipped, allowed)
                self._last_filter_log = tuple(skipped)
        return out

    def _get_feature_state(self, ctx) -> Any:
        state = getattr(ctx, "state", None)
        if state is None or not self.use_centralized_feature_state:
            return None
        return getattr(state, "feature_state", None) or getattr(state, "rolling_feature_state", None)

    def _ready_symbols(self, symbols: List[str], ctx) -> List[str]:
        feature_state = self._get_feature_state(ctx)
        if feature_state is not None and hasattr(feature_state, "ready_symbols"):
            try:
                ready = [str(s) for s in feature_state.ready_symbols()]
                allowed = set(symbols)
                ready = [s for s in ready if s in allowed]
                if ready:
                    return ready
            except Exception:
                log.exception("MLStrategy failed to read centralized feature-state readiness")

        min_bars = max(1, int(self.min_ready_bars or self.lookback_bars))
        return [s for s in symbols if len(self.price_history.get(s, [])) >= min_bars]

    def _build_feature_frame(self, symbols: List[str], ctx) -> pd.DataFrame:
        candles_by_symbol = {}

        feature_state = self._get_feature_state(ctx)
        if feature_state is not None and hasattr(feature_state, "frames_by_symbol"):
            try:
                frames = feature_state.frames_by_symbol() or {}
                for symbol in symbols:
                    frame = frames.get(symbol)
                    if isinstance(frame, pd.DataFrame) and len(frame) >= self.min_ready_bars:
                        candles_by_symbol[symbol] = frame.tail(self.lookback_bars).copy()
            except Exception:
                log.exception("MLStrategy failed to build frame from centralized feature state")
                candles_by_symbol = {}

        if not candles_by_symbol:
            for symbol in symbols:
                rows = self.price_history.get(symbol, [])
                if len(rows) < self.min_ready_bars:
                    continue

                subset = rows[-self.lookback_bars :]
                candles_by_symbol[symbol] = pd.DataFrame(
                    {
                        "symbol": [r.symbol for r in subset],
                        "ts_ms": [r.ts_ms for r in subset],
                        "timestamp": pd.to_datetime([r.ts_ms for r in subset], unit="ms", utc=True),
                        "open": [r.open for r in subset],
                        "high": [r.high for r in subset],
                        "low": [r.low for r in subset],
                        "close": [r.close for r in subset],
                        "volume": [r.volume for r in subset],
                    }
                )

        if not candles_by_symbol:
            return pd.DataFrame()

        feature_df = build_live_feature_frame(
            symbols=list(candles_by_symbol.keys()),
            ctx=ctx,
            lookback_bars=self.lookback_bars,
            interval_s=self.candle_interval_s,
            candles_by_symbol=candles_by_symbol,
        )
        return feature_df if isinstance(feature_df, pd.DataFrame) else pd.DataFrame()

    def _merge_predictions(self, feature_df: pd.DataFrame, prediction_map: Dict[str, Dict]) -> pd.DataFrame:
        rows = []
        for _, row in feature_df.iterrows():
            symbol = row["symbol"]
            pred = prediction_map.get(symbol)
            if not pred:
                continue

            merged = row.to_dict()
            merged.update(pred)
            merged["pred_prob"] = float(merged.get("pred_prob", merged.get("prob", 0.0)))
            merged["entry_score"] = float(merged.get("entry_score", merged.get("score", merged["pred_prob"])))
            merged["_resolved_score_col_name"] = "entry_score"
            rows.append(merged)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def _build_runtime_args(self):
        args = SimpleNamespace(**dict(self.cfg or {}))

        defaults = {
            "prob_column": "pred_prob",
            "rolling_volatility_column": "rolling_volatility_24h",
            "symbol_column": "symbol",
            "rank_score_min": None,
            "prob_threshold": float(self.cfg.get("prob_threshold", 0.0) or 0.0),
            "min_prob_percentile": self.cfg.get("min_prob_percentile"),
            "top_n": int(self.cfg.get("top_n", 1) or 1),
            "max_positions": int(self.cfg.get("max_positions", 1) or 1),
            "notional_per_trade": float(self.cfg.get("notional_per_trade", 0.0) or 0.0),
            "min_notional_per_trade": float(self.cfg.get("min_notional_per_trade", 0.0) or 0.0),
            "prediction_source": self.cfg.get("prediction_source", "ensemble"),
            "ranking_mode": self.cfg.get("ranking_mode", "composite"),
            "prob_zscore_weight": float(self.cfg.get("prob_zscore_weight", 1.0) or 1.0),
            "percentile_weight": float(self.cfg.get("percentile_weight", 0.2) or 0.2),
            "volatility_weight": float(self.cfg.get("volatility_weight", 0.35) or 0.35),
            "time_to_peak_weight": float(self.cfg.get("time_to_peak_weight", 0.25) or 0.25),
            "min_rolling_volatility_24h": self.cfg.get("min_rolling_volatility_24h"),
            "max_predicted_time_to_peak_hours": self.cfg.get("max_predicted_time_to_peak_hours"),
            "prob_size_cap": float(self.cfg.get("prob_size_cap", 2.0) or 2.0),
            "vol_reference": float(self.cfg.get("vol_reference", 0.006) or 0.006),
            "vol_size_floor": float(self.cfg.get("vol_size_floor", 0.75) or 0.75),
            "vol_size_cap": float(self.cfg.get("vol_size_cap", 1.25) or 1.25),
            "combined_size_cap": float(self.cfg.get("combined_size_cap", 2.0) or 2.0),
            "enable_dynamic_sizing": bool(self.cfg.get("enable_dynamic_sizing", True)),
            "enable_kelly_sizing": bool(self.cfg.get("enable_kelly_sizing", True)),
            "kelly_fraction_scale": float(self.cfg.get("kelly_fraction_scale", 0.25) or 0.25),
            "kelly_size_cap": float(self.cfg.get("kelly_size_cap", 1.5) or 1.5),
            "regime_gating_mode": self.cfg.get("regime_gating_mode", "scale"),
            "regime_risk_off_column": self.cfg.get("regime_risk_off_column", "market_risk_off_score"),
            "regime_disable_threshold": self.cfg.get("regime_disable_threshold"),
            "regime_scale_threshold": self.cfg.get("regime_scale_threshold"),
            "regime_scale_min_multiplier": float(self.cfg.get("regime_scale_min_multiplier", 0.25) or 0.25),
            "regime_score_raise_threshold": self.cfg.get("regime_score_raise_threshold"),
            "regime_score_raise_multiplier": float(self.cfg.get("regime_score_raise_multiplier", 1.0) or 1.0),
            "regime_score_raise_max": self.cfg.get("regime_score_raise_max"),
        }
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        return args

    def _filter_ranked(self, candidate_df: pd.DataFrame, args) -> pd.DataFrame:
        try:
            diagnostics = SimpleNamespace(
                candidate_rows_seen=0,
                candidate_rows_after_prob_threshold=0,
                candidate_rows_after_percentile=0,
                candidate_rows_after_volatility=0,
                candidate_rows_after_time_to_peak=0,
                candidate_rows_after_rank_score=0,
                candidate_rows_after_regime_gate=0,
                regime_gate_blocks=0,
                regime_scale_events=0,
                regime_score_raise_events=0,
            )
            ranked = filter_ranked_candidates(candidate_df, args, diagnostics)
            return ranked if isinstance(ranked, pd.DataFrame) else pd.DataFrame()
        except Exception as ex:
            log.warning("MLStrategy replay filter failed, falling back to simple sort: %s", ex)
            out = candidate_df.copy()
            prob_col = getattr(args, "prob_column", "pred_prob")
            if prob_col in out.columns:
                out = out[out[prob_col] >= float(getattr(args, "prob_threshold", 0.0))]
            if "entry_score" in out.columns:
                out = out.sort_values(["entry_score", prob_col], ascending=[False, False])
            top_n = max(int(getattr(args, "top_n", 1)), 0)
            if top_n:
                out = out.head(top_n)
            return out.reset_index(drop=True)

    def _get_router(self, ctx) -> OrderRouter:
        if self.router is None:
            self.router = OrderRouter(getattr(ctx, "cfg", {}) or {})
        return self.router

    def _build_entry_intents(self, ranked: pd.DataFrame, args, ctx, ts_ms: int):
        intents = []
        max_positions = int(getattr(args, "max_positions", 1) or 1)
        router = self._get_router(ctx)

        broker_positions = getattr(ctx.broker, "positions", {}) or {}
        open_symbols = {
            sym
            for sym, pos in broker_positions.items()
            if float(getattr(pos, "qty", 0.0) or 0.0) > 0.0
        }
        mapped_open_symbols = set(open_symbols)
        for sym in list(open_symbols):
            try:
                mapped_open_symbols.add(router.to_source_symbol(sym))
                mapped_open_symbols.add(router.to_venue_symbol(sym))
            except Exception:
                pass
        open_count = len(open_symbols)

        fee_bps = float(getattr(ctx.broker, "fee_bps", ctx.cfg.get("execution", {}).get("fee_bps", 0.0)) or 0.0)
        cash_buffer = float(self.cfg.get("order_notional_buffer", self.order_notional_buffer) or self.order_notional_buffer)

        for _, row in ranked.iterrows():
            symbol = str(row["symbol"])

            if symbol in mapped_open_symbols:
                continue
            if open_count >= max_positions:
                break

            ms = ctx.state.market.get(symbol)
            ask = getattr(ms, "ask", None) if ms is not None else None
            if ask is None or float(ask) <= 0.0:
                continue

            prob_mult, vol_mult, kelly_fraction, kelly_mult, total_mult = compute_size_multipliers(row, args)

            regime_mult = float(row.get("_regime_size_multiplier", 1.0) or 1.0)
            target_notional = float(getattr(args, "notional_per_trade", self.cfg.get("notional_per_trade", 0.0)))
            target_notional *= float(total_mult) * regime_mult

            min_notional = float(getattr(args, "min_notional_per_trade", 0.0) or 0.0)
            if target_notional < min_notional:
                continue

            cash = float(getattr(ctx.broker, "cash", 0.0) or 0.0)
            if cash <= 0.0:
                continue

            slip = float(ctx.cfg.get("execution", {}).get("entry_slippage_cap_pct", 0.0) or 0.0)
            raw_limit_px = float(ask) * (1.0 + slip)
            limit_px = router.normalize_price(symbol, raw_limit_px)
            if limit_px <= 0.0:
                continue

            desired_qty = target_notional / max(limit_px, 1e-12)
            qty = router.clamp_buy_quantity_to_cash(
                symbol=symbol,
                desired_qty=desired_qty,
                price=limit_px,
                cash=cash,
                fee_bps=fee_bps,
                cash_buffer=cash_buffer,
            )
            if qty <= 0.0:
                continue

            routed_notional = qty * limit_px
            if routed_notional < min_notional:
                continue

            prob_value = float(row.get("pred_prob", row.get("prob", 0.0)) or 0.0)
            score_value = float(row.get("entry_score", row.get("score", prob_value)) or prob_value)
            throttle_decision = self.signal_throttle.should_emit_signal(
                symbol=symbol,
                side="BUY",
                ts_ms=ts_ms,
                prob=prob_value,
                score=score_value,
                mode=self.ml_mode,
            )
            if not throttle_decision.allowed:
                log.info(
                    "ML_SIGNAL_SUPPRESSED_%s symbol=%s side=BUY prob=%.6f score=%.6f details=%s",
                    str(throttle_decision.reason).upper(),
                    symbol,
                    prob_value,
                    score_value,
                    throttle_decision.details,
                )
                self._increment_state_counter(ctx, "ml_signal_suppressed_count", 1)
                continue

            self.signal_throttle.update_signal_state(
                symbol=symbol,
                side="BUY",
                ts_ms=ts_ms,
                prob=prob_value,
                score=score_value,
            )
            log.info(
                "ML_SIGNAL_ALLOWED symbol=%s side=BUY prob=%.6f score=%.6f reason=%s",
                symbol,
                prob_value,
                score_value,
                throttle_decision.reason,
            )

            self.last_signal_ts[symbol] = ts_ms
            self._update_state_for_signal(
                ctx=ctx,
                symbol=symbol,
                ts_ms=ts_ms,
                row=row,
                notional=routed_notional,
                total_mult=total_mult,
            )

            log.info(
                "%s %s prob=%.4f score=%.4f size_mult=%.4f target_notional=%.2f routed_notional=%.2f qty=%.8f limit_px=%.8f",
                "ML_SHADOW_WOULD_BUY" if self.ml_mode == "shadow" else "ML SIGNAL BUY",
                symbol,
                prob_value,
                score_value,
                float(total_mult),
                float(target_notional),
                float(routed_notional),
                float(qty),
                float(limit_px),
            )

            intents.append(
                OrderIntent(
                    side="BUY",
                    symbol=symbol,
                    qty=float(qty),
                    limit_px=float(limit_px),
                    tif="IOC",
                )
            )
            open_count += 1

        return intents

    def _update_state_with_rankings(self, ctx, ranked: pd.DataFrame, ts_ms: int) -> None:
        if ranked is None or ranked.empty:
            return
        if hasattr(ctx.state, "set_ml_ranking_batch"):
            ranked_symbols = ranked["symbol"].astype(str).tolist()
            ctx.state.set_ml_ranking_batch(
                ts_ms=ts_ms,
                ranked_symbols=ranked_symbols,
                top_n_symbols=ranked_symbols,
                metadata={"strategy": self.name},
            )
        if hasattr(ctx.state, "set_ml_candidate_snapshot"):
            for _, row in ranked.iterrows():
                ctx.state.set_ml_candidate_snapshot(str(row["symbol"]), ts_ms, row.to_dict())

    def _update_state_for_signal(self, ctx, symbol: str, ts_ms: int, row: pd.Series, notional: float, total_mult: float) -> None:
        payload = row.to_dict()
        payload.update(
            {
                "symbol": symbol,
                "notional_usd": float(notional),
                "total_size_multiplier": float(total_mult),
                "strategy": self.name,
            }
        )
        if hasattr(ctx.state, "set_ml_signal_snapshot"):
            ctx.state.set_ml_signal_snapshot(symbol, ts_ms, payload)
