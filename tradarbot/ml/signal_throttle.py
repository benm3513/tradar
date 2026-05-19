from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

LOGGER = logging.getLogger("tradarbot.signal_throttle")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


@dataclass
class SignalThrottleDecision:
    allowed: bool
    reason: str = "allowed"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolSignalState:
    last_signal_ts_ms: Optional[int] = None
    last_signal_prob: Optional[float] = None
    last_signal_score: Optional[float] = None
    last_signal_side: Optional[str] = None
    recent_signal_ts_ms: Deque[int] = field(default_factory=lambda: deque(maxlen=512))


class SignalThrottle:
    """Stateful live/shadow ML signal suppression layer.

    The object is intentionally small and deterministic. It does not know about
    brokers, risk, or storage; it only decides whether a candidate signal is
    worth emitting based on timestamp, symbol, side, probability, score and YAML
    configuration.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("suppress_duplicate_signals", True))
        self.min_signal_interval_seconds = float(cfg.get("min_signal_interval_seconds", 900) or 0.0)
        self.min_prob_delta = float(cfg.get("min_prob_delta", 0.02) or 0.0)
        self.min_score_delta = float(cfg.get("min_score_delta", 0.15) or 0.0)
        self.max_shadow_signals_per_hour = int(cfg.get("max_shadow_signals_per_hour", 20) or 0)
        self.signal_on_candle_close_only = bool(cfg.get("signal_on_candle_close_only", False))
        self.candle_interval_s = int(cfg.get("candle_interval_s", cfg.get("feature_interval_s", 3600)) or 3600)
        self.candle_close_tolerance_ms = int(cfg.get("candle_close_tolerance_ms", 1500) or 1500)
        self._state: Dict[str, SymbolSignalState] = defaultdict(SymbolSignalState)

    def should_emit_signal(
        self,
        *,
        symbol: str,
        side: str,
        ts_ms: int,
        prob: Optional[float] = None,
        score: Optional[float] = None,
        mode: str = "shadow",
    ) -> SignalThrottleDecision:
        if not self.enabled:
            return SignalThrottleDecision(True, "disabled")

        symbol = str(symbol)
        side = str(side).upper()
        ts_ms = _safe_int(ts_ms)
        prob_f = _safe_float(prob, 0.0)
        score_f = _safe_float(score, prob_f)
        state = self._state[symbol]

        if self.signal_on_candle_close_only and not self._is_candle_close(ts_ms):
            return SignalThrottleDecision(False, "candle_close_only", {"ts_ms": ts_ms})

        if self.max_shadow_signals_per_hour > 0:
            self._prune_recent(state, ts_ms)
            if len(state.recent_signal_ts_ms) >= self.max_shadow_signals_per_hour:
                return SignalThrottleDecision(
                    False,
                    "hourly_cap",
                    {"count": len(state.recent_signal_ts_ms), "cap": self.max_shadow_signals_per_hour},
                )

        if state.last_signal_ts_ms is not None:
            elapsed_s = max(0.0, (ts_ms - int(state.last_signal_ts_ms)) / 1000.0)
            if elapsed_s < self.min_signal_interval_seconds:
                # Permit a materially different signal to break the interval;
                # exact/near duplicates remain suppressed.
                prob_delta = abs(prob_f - _safe_float(state.last_signal_prob, prob_f))
                score_delta = abs(score_f - _safe_float(state.last_signal_score, score_f))
                same_side = side == (state.last_signal_side or side)
                if same_side and prob_delta < self.min_prob_delta and score_delta < self.min_score_delta:
                    return SignalThrottleDecision(
                        False,
                        "cooldown_duplicate",
                        {
                            "elapsed_s": elapsed_s,
                            "min_signal_interval_seconds": self.min_signal_interval_seconds,
                            "prob_delta": prob_delta,
                            "score_delta": score_delta,
                        },
                    )
                if elapsed_s < min(self.min_signal_interval_seconds, 60.0):
                    return SignalThrottleDecision(
                        False,
                        "cooldown",
                        {"elapsed_s": elapsed_s, "min_signal_interval_seconds": self.min_signal_interval_seconds},
                    )

            prob_delta = abs(prob_f - _safe_float(state.last_signal_prob, prob_f))
            score_delta = abs(score_f - _safe_float(state.last_signal_score, score_f))
            if side == (state.last_signal_side or side) and prob_delta < self.min_prob_delta and score_delta < self.min_score_delta:
                return SignalThrottleDecision(
                    False,
                    "duplicate_delta",
                    {"prob_delta": prob_delta, "score_delta": score_delta},
                )

        return SignalThrottleDecision(True, "allowed")

    def update_signal_state(
        self,
        *,
        symbol: str,
        side: str,
        ts_ms: int,
        prob: Optional[float] = None,
        score: Optional[float] = None,
    ) -> None:
        state = self._state[str(symbol)]
        state.last_signal_ts_ms = _safe_int(ts_ms)
        state.last_signal_prob = _safe_float(prob, 0.0)
        state.last_signal_score = _safe_float(score, state.last_signal_prob)
        state.last_signal_side = str(side).upper()
        self._prune_recent(state, state.last_signal_ts_ms)
        state.recent_signal_ts_ms.append(state.last_signal_ts_ms)

    def reset_hourly_counts(self, ts_ms: Optional[int] = None) -> None:
        if ts_ms is None:
            for state in self._state.values():
                state.recent_signal_ts_ms.clear()
            return
        ts_ms = _safe_int(ts_ms)
        for state in self._state.values():
            self._prune_recent(state, ts_ms)

    def _prune_recent(self, state: SymbolSignalState, ts_ms: int) -> None:
        cutoff = int(ts_ms) - 3600 * 1000
        while state.recent_signal_ts_ms and state.recent_signal_ts_ms[0] < cutoff:
            state.recent_signal_ts_ms.popleft()

    def _is_candle_close(self, ts_ms: int) -> bool:
        interval_ms = max(1, int(self.candle_interval_s)) * 1000
        remainder = int(ts_ms) % interval_ms
        return remainder <= self.candle_close_tolerance_ms or (interval_ms - remainder) <= self.candle_close_tolerance_ms
