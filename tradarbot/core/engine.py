from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from tradarbot.core.events import (
    CandleEvent,
    ListingEvent,
    OrderIntent,
    MLEntryRequestEvent,
    MLExitRequestEvent,
    MLFeatureEvent,
    MLPredictionEvent,
    MLCandidateEvent,
    MLRankingEvent,
    MLSignalEvent,
)

log = logging.getLogger("tradarbot.engine")


class StrategyEngine:
    """Application strategy router.

    Phase 5.0 goals:
    - preserve existing flow for Algo1 / Algo2
    - allow ML strategies to emit richer request/signal objects
    - normalize those objects back into OrderIntent
    - keep risk -> broker execution unchanged
    """

    def __init__(self, strategies, risk, broker, ctx):
        self.strategies = strategies
        self.risk = risk
        self.broker = broker
        self.ctx = ctx

    def on_listing(self, ev: ListingEvent):
        self.ctx.state.current_event_ts_ms = int(ev.ts_ms)
        self.ctx.state.listings.setdefault(ev.symbol, int(ev.ts_ms))
        for strat in self.strategies:
            fn = getattr(strat, "on_listing", None)
            if callable(fn):
                outputs = fn(ev, self.ctx) or []
                self._handle_strategy_outputs(outputs, strat.name, trigger_event=ev)

    def on_candle(self, ev: CandleEvent):
        self.ctx.state.current_event_ts_ms = int(ev.ts_ms)
        self.ctx.store.insert_candle(ev)
        for strat in self.strategies:
            outputs = strat.on_candle(ev, self.ctx) or []
            self._handle_strategy_outputs(outputs, strat.name, trigger_event=ev)

    def _handle_strategy_outputs(self, outputs, strat_name: str, trigger_event=None) -> None:
        normalized = self._normalize_outputs(outputs)
        for item in normalized:
            # informational ML events are published/logged but not executed
            if self._is_observer_event(item):
                self._publish_observer_event(item)
                continue

            intent = self._to_order_intent(item)
            if intent is None:
                log.warning(
                    "UNHANDLED_OUTPUT strat=%s type=%s value=%r",
                    strat_name,
                    type(item).__name__,
                    item,
                )
                continue

            decision = self.risk.check(intent, self.ctx, strat_name)
            if not decision["approved"]:
                log.info(
                    "REJECT strat=%s side=%s sym=%s reason=%s",
                    strat_name,
                    intent.side,
                    intent.symbol,
                    decision.get("reason"),
                )
                state = getattr(self.ctx, "state", None)
                if state is not None:
                    state.order_rejection_counts = int(getattr(state, "order_rejection_counts", 0) or 0) + 1
                self._publish_rejection_signal(item=item, strat_name=strat_name, reason=decision.get("reason"))
                continue

            approved_intent = decision["intent"]
            self._publish_execution_signal(item=item, strat_name=strat_name, approved_intent=approved_intent)
            self.broker.execute_intent(approved_intent, self.ctx)

    def _normalize_outputs(self, outputs) -> List[object]:
        if outputs is None:
            return []
        if isinstance(outputs, list):
            return outputs
        if isinstance(outputs, tuple):
            return list(outputs)
        return [outputs]

    def _is_observer_event(self, item: object) -> bool:
        return isinstance(
            item,
            (
                MLFeatureEvent,
                MLPredictionEvent,
                MLCandidateEvent,
                MLRankingEvent,
                MLSignalEvent,
            ),
        )

    def _to_order_intent(self, item: object) -> Optional[OrderIntent]:
        if isinstance(item, OrderIntent):
            return item

        if isinstance(item, (MLEntryRequestEvent, MLExitRequestEvent)):
            return item.to_order_intent()

        # typed schema objects from tradarbot/ml/signal_schema.py
        to_order_intent = getattr(item, "to_order_intent", None)
        if callable(to_order_intent):
            try:
                intent = to_order_intent()
                if isinstance(intent, OrderIntent):
                    return intent
            except Exception:
                log.exception("FAILED_TO_CONVERT_OUTPUT type=%s", type(item).__name__)
                return None

        # permissive dict support for Phase 5.0 scaffolding / tests
        if isinstance(item, dict):
            kind = str(item.get("type", "")).lower()
            if kind in {"ml_feature", "ml_prediction", "ml_candidate", "ml_ranking", "ml_signal"}:
                self._publish_dict_observer_event(item)
                return None

            required = {"side", "symbol", "qty", "limit_px"}
            if required.issubset(item.keys()):
                return OrderIntent(
                    side=item["side"],
                    symbol=item["symbol"],
                    qty=float(item["qty"]),
                    limit_px=float(item["limit_px"]),
                    tif=item.get("tif", "IOC"),
                )

        return None

    def _publish_observer_event(self, item: object) -> None:
        bus = getattr(self.ctx, "bus", None)
        if bus is not None:
            try:
                bus.publish(item)
            except Exception:
                log.exception("FAILED_TO_PUBLISH_OBSERVER_EVENT type=%s", type(item).__name__)

        # keep a lightweight trace in state for Phase 5.0 introspection
        self._remember_ml_state(item)

    def _publish_dict_observer_event(self, item: dict) -> None:
        state = getattr(self.ctx, "state", None)
        if state is None:
            return
        history = getattr(state, "ml_event_history", None)
        if history is None:
            state.ml_event_history = []
            history = state.ml_event_history
        history.append(item)
        if len(history) > 500:
            del history[:-500]

    def _publish_execution_signal(self, item: object, strat_name: str, approved_intent: OrderIntent) -> None:
        ts_ms = int(getattr(self.ctx.state, "current_event_ts_ms", 0) or 0)
        signal = MLSignalEvent(
            symbol=approved_intent.symbol,
            ts_ms=ts_ms,
            action=f"execute_{approved_intent.side.lower()}",
            prob=getattr(item, "prob", None),
            score=getattr(item, "score", None),
            entry_score=getattr(item, "entry_score", None),
            metadata={
                "strategy": strat_name,
                "qty": approved_intent.qty,
                "limit_px": approved_intent.limit_px,
                "tif": approved_intent.tif,
            },
        )
        self._publish_observer_event(signal)

    def _publish_rejection_signal(self, item: object, strat_name: str, reason: Optional[str]) -> None:
        symbol = getattr(item, "symbol", None)
        if not symbol:
            symbol = getattr(getattr(item, "raw", None), "get", lambda *_: None)("symbol")

        if not symbol:
            return

        ts_ms = int(getattr(self.ctx.state, "current_event_ts_ms", 0) or 0)
        signal = MLSignalEvent(
            symbol=symbol,
            ts_ms=ts_ms,
            action="rejected",
            prob=getattr(item, "prob", None),
            score=getattr(item, "score", None),
            entry_score=getattr(item, "entry_score", None),
            metadata={
                "strategy": strat_name,
                "reason": reason,
            },
        )
        self._publish_observer_event(signal)

    def _remember_ml_state(self, item: object) -> None:
        state = getattr(self.ctx, "state", None)
        if state is None:
            return

        # lazily create fields so this engine can work with the old State class too
        if not hasattr(state, "ml_event_history"):
            state.ml_event_history = []
        if not hasattr(state, "ml_latest_rankings"):
            state.ml_latest_rankings = {}
        if not hasattr(state, "ml_latest_predictions"):
            state.ml_latest_predictions = {}
        if not hasattr(state, "ml_latest_features"):
            state.ml_latest_features = {}

        if isinstance(item, MLFeatureEvent):
            state.ml_latest_features[item.symbol] = item.features
        elif isinstance(item, MLPredictionEvent):
            state.ml_latest_predictions[item.symbol] = {
                "prob": item.prob,
                "score": item.score,
                "entry_score": item.entry_score,
                "prediction_source": item.prediction_source,
                "model_name": item.model_name,
                **dict(item.payload or {}),
            }
        elif isinstance(item, MLCandidateEvent):
            state.ml_latest_rankings[item.symbol] = {
                "prob": item.prob,
                "score": item.score,
                "entry_score": item.entry_score,
                "accepted": item.accepted,
                "reject_reason": item.reject_reason,
                **dict(item.payload or {}),
            }
        elif isinstance(item, MLRankingEvent):
            state.ml_latest_ranking_batch = {
                "top_n": item.top_n,
                "ranking_mode": item.ranking_mode,
                "symbols_ranked": list(item.symbols_ranked),
                "candidates": list(item.candidates),
                "metadata": dict(item.metadata or {}),
            }

        state.ml_event_history.append(item)
        if len(state.ml_event_history) > 500:
            del state.ml_event_history[:-500]
