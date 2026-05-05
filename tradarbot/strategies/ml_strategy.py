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
        self.predictor = LivePredictor(self.cfg)
        self.router = None

        self.last_signal_ts: Dict[str, int] = {}
        self.last_eval_ts_by_symbol: Dict[str, int] = {}
        self.price_history = defaultdict(list)

        self.lookback_bars = int(
            self.cfg.get("feature_lookback_bars", self.cfg.get("warmup_candles", 200))
        )
        self.evaluation_interval_s = int(self.cfg.get("evaluation_interval_s", 1))
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
            "MLStrategy predictor initialized mode=%s min_ready_bars=%s lookback_bars=%s centralized_feature_state=%s",
            getattr(self.predictor, "mode", self.cfg.get("predictor_mode", self.cfg.get("mode"))),
            self.min_ready_bars,
            self.lookback_bars,
            self.use_centralized_feature_state,
        )

    def on_listing(self, ev, ctx):
        return []

    def on_candle(self, e, ctx):
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

        log.info("ML_DEBUG active_symbols_before_filter=%s", symbols)

        symbols = self._filter_tradable_symbols(symbols, ctx)
        log.info("ML_DEBUG symbols_after_filter=%s", symbols)

        if not symbols:
            log.info("ML_DEBUG return=no_symbols_after_filter")
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
                log.info("ML_DEBUG feature_state_inspect_failed=%s", ex)

            log.info(
                "ML_DEBUG return=no_ready_symbols feature_state_present=%s feature_state_symbols=%s feature_state_ready=%s",
                fs is not None,
                fs_symbols,
                fs_ready,
            )
            return []

        feature_df = self._build_feature_frame(ready_symbols, ctx)
        log.info(
            "ML_DEBUG feature_df rows=%s cols=%s",
            0 if feature_df is None else len(feature_df),
            [] if feature_df is None or feature_df.empty else list(feature_df.columns),
        )

        if feature_df.empty:
            log.info("ML_DEBUG return=empty_feature_df ready_symbols=%s", ready_symbols)
            return []

        prediction_map = self.predictor.predict(feature_df, ctx=ctx)
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
        self.last_eval_ts_by_symbol[e.symbol] = int(e.ts_ms)
        log.info("ML_DEBUG intents_count=%s", len(intents))
        return intents

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
                log.info("ML_TRADABLE_FILTER skipped=%s allowed=%s", skipped, allowed)
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
                "ML SIGNAL BUY %s prob=%.4f score=%.4f size_mult=%.4f target_notional=%.2f routed_notional=%.2f qty=%.8f limit_px=%.8f",
                symbol,
                float(row.get("pred_prob", row.get("prob", 0.0))),
                float(row.get("entry_score", row.get("score", 0.0))),
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
