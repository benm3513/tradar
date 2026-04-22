from __future__ import annotations

import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd

from tradarbot.core.events import OrderIntent
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
    Phase 5.0 live ML strategy with replay-parity filtering/ranking/sizing.

    Keeps the integration-safe fixes:
    - has `name` for StrategyEngine
    - uses in-memory candle history instead of replay DB tables
    - reuses replay helper math for filtering/ranking/sizing
    - guards replay helper calls that assume diagnostics objects
    """

    name = "ml_strategy"

    def __init__(self, cfg: Dict):
        self.cfg = dict(cfg or {})
        self.predictor = LivePredictor(self.cfg)

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

    def on_listing(self, ev, ctx):
        return []

    def on_candle(self, e, ctx):
        self._record_candle(e)
        self._sync_state_symbol(e.symbol, ctx)

        last_eval = self.last_eval_ts_by_symbol.get(e.symbol)
        if last_eval is not None:
            min_gap_ms = max(1, self.evaluation_interval_s) * 1000
            if int(e.ts_ms) - int(last_eval) < min_gap_ms:
                return []

        symbols = sorted(set(getattr(ctx.state, "active_symbols", set()) or {e.symbol}))
        if e.symbol not in symbols:
            symbols.append(e.symbol)

        ready_symbols = [s for s in symbols if len(self.price_history.get(s, [])) >= self.lookback_bars]
        if not ready_symbols:
            return []

        feature_df = self._build_feature_frame(ready_symbols, ctx)
        if feature_df.empty:
            return []

        prediction_map = self.predictor.predict(feature_df, ctx=ctx)
        candidate_df = self._merge_predictions(feature_df, prediction_map)
        if candidate_df.empty:
            return []

        args = self._build_runtime_args()
        ranked = self._filter_ranked(candidate_df, args)
        self._update_state_with_rankings(ctx, ranked, int(e.ts_ms))

        if ranked.empty:
            self.last_eval_ts_by_symbol[e.symbol] = int(e.ts_ms)
            return []

        intents = self._build_entry_intents(ranked, args, ctx, ts_ms=int(e.ts_ms))
        self.last_eval_ts_by_symbol[e.symbol] = int(e.ts_ms)
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

    def _build_feature_frame(self, symbols: List[str], ctx) -> pd.DataFrame:
        candles_by_symbol = {}
        for symbol in symbols:
            rows = self.price_history.get(symbol, [])
            if len(rows) < self.lookback_bars:
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
        args = build_runtime_args(overrides=self.cfg, section="ml_live")
        if not hasattr(args, "prob_column"):
            args.prob_column = "pred_prob"
        if not hasattr(args, "rolling_volatility_column"):
            args.rolling_volatility_column = "rolling_volatility_24h"
        if not hasattr(args, "symbol_column"):
            args.symbol_column = "symbol"
        if not hasattr(args, "rank_score_min"):
            args.rank_score_min = None
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

    def _build_entry_intents(self, ranked: pd.DataFrame, args, ctx, ts_ms: int):
        intents = []
        max_positions = int(getattr(args, "max_positions", 1) or 1)

        broker_positions = getattr(ctx.broker, "positions", {}) or {}
        open_symbols = {
            sym
            for sym, pos in broker_positions.items()
            if float(getattr(pos, "qty", 0.0) or 0.0) > 0.0
        }
        open_count = len(open_symbols)

        for _, row in ranked.iterrows():
            symbol = str(row["symbol"])

            if symbol in open_symbols:
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
            if target_notional > cash:
                target_notional = cash

            qty = target_notional / float(ask)
            if qty <= 0.0:
                continue

            slip = float(ctx.cfg.get("execution", {}).get("entry_slippage_cap_pct", 0.0) or 0.0)
            limit_px = float(ask) * (1.0 + slip)

            self.last_signal_ts[symbol] = ts_ms
            self._update_state_for_signal(
                ctx=ctx,
                symbol=symbol,
                ts_ms=ts_ms,
                row=row,
                notional=target_notional,
                total_mult=total_mult,
            )

            log.info(
                "ML SIGNAL BUY %s prob=%.4f score=%.4f size_mult=%.4f notional=%.2f",
                symbol,
                float(row.get("pred_prob", row.get("prob", 0.0))),
                float(row.get("entry_score", row.get("score", 0.0))),
                float(total_mult),
                float(target_notional),
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
