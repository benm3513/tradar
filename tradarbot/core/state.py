from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class MarketState:
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    last_ts_ms: Optional[int] = None


@dataclass
class MLStrategySymbolState:
    """Per-symbol runtime state for Phase 5.0 ML strategy."""
    symbol: str
    last_feature_ts_ms: Optional[int] = None
    last_prediction_ts_ms: Optional[int] = None
    last_candidate_ts_ms: Optional[int] = None
    last_signal_ts_ms: Optional[int] = None
    last_entry_ts_ms: Optional[int] = None
    last_exit_ts_ms: Optional[int] = None

    cooldown_until_ts_ms: Optional[int] = None

    last_prob: Optional[float] = None
    last_score: Optional[float] = None
    last_entry_score: Optional[float] = None
    last_prediction_source: Optional[str] = None
    last_model_name: Optional[str] = None

    last_prob_percentile_rank: Optional[float] = None
    last_rolling_volatility_24h: Optional[float] = None
    last_predicted_time_to_peak_hours: Optional[float] = None

    last_market_risk_off_score: Optional[float] = None
    last_market_dispersion_24h: Optional[float] = None
    last_market_trend_strength_24h: Optional[float] = None
    last_market_volume_regime_24h: Optional[float] = None

    open_position_managed_by_ml: bool = False
    trailing_stop_price: Optional[float] = None
    peak_price: Optional[float] = None
    partial_exit_taken: bool = False

    latest_features: Dict[str, Any] = field(default_factory=dict)
    latest_prediction: Dict[str, Any] = field(default_factory=dict)
    latest_candidate: Dict[str, Any] = field(default_factory=dict)
    latest_signal: Dict[str, Any] = field(default_factory=dict)


class State:
    def __init__(self):
        # -------------------------
        # Existing app state
        # -------------------------
        self.market: Dict[str, MarketState] = {}
        self.listings: Dict[str, int] = {}
        self.current_event_ts_ms: Optional[int] = None

        # main.py currently attaches this dynamically, so keep it first-class
        self.active_symbols: Set[str] = set()

        # -------------------------
        # Phase 5.0 ML runtime state
        # -------------------------
        # per-symbol ML runtime memory
        self.ml_symbol_state: Dict[str, MLStrategySymbolState] = {}

        # latest cross-sectional outputs
        self.ml_latest_features: Dict[str, Dict[str, Any]] = {}
        self.ml_latest_predictions: Dict[str, Dict[str, Any]] = {}
        self.ml_latest_rankings: Dict[str, Dict[str, Any]] = {}

        # latest ranked batch / selection info
        self.ml_latest_ranking_batch: Dict[str, Any] = {}
        self.ml_current_ranked_symbols: List[str] = []
        self.ml_current_top_n_symbols: List[str] = []

        # cooldown / timing controls
        self.ml_cooldowns_by_symbol: Dict[str, int] = {}
        self.ml_last_signal_ts_by_symbol: Dict[str, int] = {}
        self.ml_last_entry_ts_by_symbol: Dict[str, int] = {}
        self.ml_last_exit_ts_by_symbol: Dict[str, int] = {}

        # generic ML event history for inspection/debugging
        self.ml_event_history: List[Any] = []

        # optional strategy-level aggregates
        self.ml_last_refresh_ts_ms: Optional[int] = None
        self.ml_last_ranking_ts_ms: Optional[int] = None
        self.ml_last_selection_ts_ms: Optional[int] = None

        # -------------------------
        # Phase 5.2 live market-data/feature state
        # -------------------------
        self.feature_state = None
        self.feature_state_health: Dict[str, Any] = {}
        self.market_data_health: Dict[str, Any] = {}
        self.live_regime_snapshot: Dict[str, Any] = {}
        self.latest_context_snapshot_metadata: Dict[str, Any] = {}
        self.rolling_ready_symbols: List[str] = []
        self.ws_health: Dict[str, Any] = {}
        self.rest_health: Dict[str, Any] = {}

    @staticmethod
    def market_state_factory() -> MarketState:
        return MarketState()

    # -----------------------------------------------------------------
    # Phase 5.0 helpers
    # -----------------------------------------------------------------

    def get_ml_symbol_state(self, symbol: str) -> MLStrategySymbolState:
        state = self.ml_symbol_state.get(symbol)
        if state is None:
            state = MLStrategySymbolState(symbol=symbol)
            self.ml_symbol_state[symbol] = state
        return state

    def set_ml_feature_snapshot(self, symbol: str, ts_ms: Optional[int], payload: Dict[str, Any]) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.last_feature_ts_ms = ts_ms
        sym_state.latest_features = dict(payload or {})
        self.ml_latest_features[symbol] = dict(payload or {})
        if ts_ms is not None:
            self.ml_last_refresh_ts_ms = int(ts_ms)

    def set_ml_prediction_snapshot(self, symbol: str, ts_ms: Optional[int], payload: Dict[str, Any]) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.last_prediction_ts_ms = ts_ms
        sym_state.latest_prediction = dict(payload or {})
        sym_state.last_prob = payload.get("prob") if payload else None
        sym_state.last_score = payload.get("score") if payload else None
        sym_state.last_entry_score = payload.get("entry_score") if payload else None
        sym_state.last_prediction_source = payload.get("prediction_source") if payload else None
        sym_state.last_model_name = payload.get("model_name") if payload else None
        sym_state.last_prob_percentile_rank = payload.get("prob_percentile_rank") if payload else None
        sym_state.last_rolling_volatility_24h = payload.get("rolling_volatility_24h") if payload else None
        sym_state.last_predicted_time_to_peak_hours = payload.get("predicted_time_to_peak_hours") if payload else None
        sym_state.last_market_risk_off_score = payload.get("market_risk_off_score") if payload else None
        sym_state.last_market_dispersion_24h = payload.get("market_dispersion_24h") if payload else None
        sym_state.last_market_trend_strength_24h = payload.get("market_trend_strength_24h") if payload else None
        sym_state.last_market_volume_regime_24h = payload.get("market_volume_regime_24h") if payload else None
        self.ml_latest_predictions[symbol] = dict(payload or {})

    def set_ml_candidate_snapshot(self, symbol: str, ts_ms: Optional[int], payload: Dict[str, Any]) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.last_candidate_ts_ms = ts_ms
        sym_state.latest_candidate = dict(payload or {})
        self.ml_latest_rankings[symbol] = dict(payload or {})

    def set_ml_signal_snapshot(self, symbol: str, ts_ms: Optional[int], payload: Dict[str, Any]) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.last_signal_ts_ms = ts_ms
        sym_state.latest_signal = dict(payload or {})
        if ts_ms is not None:
            self.ml_last_signal_ts_by_symbol[symbol] = int(ts_ms)

    def set_ml_entry(self, symbol: str, ts_ms: Optional[int], cooldown_until_ts_ms: Optional[int] = None) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.last_entry_ts_ms = ts_ms
        sym_state.open_position_managed_by_ml = True
        sym_state.partial_exit_taken = False
        sym_state.trailing_stop_price = None
        sym_state.peak_price = None
        if ts_ms is not None:
            self.ml_last_entry_ts_by_symbol[symbol] = int(ts_ms)
        if cooldown_until_ts_ms is not None:
            sym_state.cooldown_until_ts_ms = int(cooldown_until_ts_ms)
            self.ml_cooldowns_by_symbol[symbol] = int(cooldown_until_ts_ms)

    def set_ml_exit(self, symbol: str, ts_ms: Optional[int], cooldown_until_ts_ms: Optional[int] = None) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.last_exit_ts_ms = ts_ms
        sym_state.open_position_managed_by_ml = False
        sym_state.partial_exit_taken = False
        sym_state.trailing_stop_price = None
        sym_state.peak_price = None
        if ts_ms is not None:
            self.ml_last_exit_ts_by_symbol[symbol] = int(ts_ms)
        if cooldown_until_ts_ms is not None:
            sym_state.cooldown_until_ts_ms = int(cooldown_until_ts_ms)
            self.ml_cooldowns_by_symbol[symbol] = int(cooldown_until_ts_ms)

    def set_ml_ranking_batch(
        self,
        *,
        ts_ms: Optional[int],
        ranked_symbols: List[str],
        top_n_symbols: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ml_last_ranking_ts_ms = ts_ms
        self.ml_current_ranked_symbols = list(ranked_symbols or [])
        self.ml_current_top_n_symbols = list(top_n_symbols or ranked_symbols or [])
        self.ml_latest_ranking_batch = {
            "ts_ms": ts_ms,
            "ranked_symbols": list(ranked_symbols or []),
            "top_n_symbols": list(top_n_symbols or ranked_symbols or []),
            "metadata": dict(metadata or {}),
        }
        self.ml_last_selection_ts_ms = ts_ms

    def set_ml_cooldown(self, symbol: str, cooldown_until_ts_ms: Optional[int]) -> None:
        sym_state = self.get_ml_symbol_state(symbol)
        sym_state.cooldown_until_ts_ms = cooldown_until_ts_ms
        if cooldown_until_ts_ms is None:
            self.ml_cooldowns_by_symbol.pop(symbol, None)
        else:
            self.ml_cooldowns_by_symbol[symbol] = int(cooldown_until_ts_ms)

    def is_ml_symbol_on_cooldown(self, symbol: str, ts_ms: Optional[int]) -> bool:
        if ts_ms is None:
            return False
        cooldown_until = self.ml_cooldowns_by_symbol.get(symbol)
        if cooldown_until is None:
            cooldown_until = self.get_ml_symbol_state(symbol).cooldown_until_ts_ms
        return cooldown_until is not None and int(ts_ms) < int(cooldown_until)

    def trim_ml_event_history(self, max_items: int = 500) -> None:
        if len(self.ml_event_history) > max_items:
            del self.ml_event_history[:-max_items]

    # -----------------------------------------------------------------
    # Phase 5.2 helpers
    # -----------------------------------------------------------------

    def set_feature_state_health(self, payload: Dict[str, Any]) -> None:
        self.feature_state_health = dict(payload or {})
        self.rolling_ready_symbols = list((payload or {}).get("ready_symbol_list", self.rolling_ready_symbols))

    def set_market_data_health(self, payload: Dict[str, Any]) -> None:
        self.market_data_health = dict(payload or {})
        if "ws" in self.market_data_health:
            self.ws_health = dict(self.market_data_health.get("ws") or {})
        if "rest" in self.market_data_health:
            self.rest_health = dict(self.market_data_health.get("rest") or {})

    def set_live_regime_snapshot(self, payload: Dict[str, Any]) -> None:
        self.live_regime_snapshot = dict(payload or {})

    def set_live_context_snapshot_metadata(self, payload: Dict[str, Any]) -> None:
        self.latest_context_snapshot_metadata = dict(payload or {})
        ready = self.latest_context_snapshot_metadata.get("ready_symbols")
        if isinstance(ready, list):
            self.rolling_ready_symbols = list(ready)

# ---------------------------------------------------------------------
# Phase 5.4 portfolio-state helpers attached without disturbing prior ML state.
# ---------------------------------------------------------------------
try:
    from tradarbot.portfolio.positions import LivePositionState, PortfolioSnapshot
except Exception:  # import-safe for partial deployments
    LivePositionState = None
    PortfolioSnapshot = None


_original_state_init = State.__init__


def _phase54_state_init(self):
    _original_state_init(self)
    self.live_positions = {}
    self.portfolio_snapshot = {}
    self.last_reconciliation = {}
    self.exit_state_by_symbol = {}
    self.portfolio_fail_closed = False


def _set_live_position(self, symbol, position):
    if position is None:
        self.remove_live_position(symbol)
        return
    sym = str(symbol or getattr(position, "symbol", ""))
    self.live_positions[sym] = position
    try:
        ml_state = self.get_ml_symbol_state(sym)
        ml_state.open_position_managed_by_ml = True
        ml_state.trailing_stop_price = getattr(position, "trailing_stop_price", None)
        ml_state.peak_price = getattr(position, "peak_price", None)
        ml_state.partial_exit_taken = bool(getattr(position, "partial_exit_taken", False))
    except Exception:
        pass


def _remove_live_position(self, symbol):
    sym = str(symbol)
    self.live_positions.pop(sym, None)
    try:
        ml_state = self.get_ml_symbol_state(sym)
        ml_state.open_position_managed_by_ml = False
        ml_state.trailing_stop_price = None
        ml_state.peak_price = None
        ml_state.partial_exit_taken = False
    except Exception:
        pass


def _get_live_position(self, symbol):
    return self.live_positions.get(str(symbol))


def _set_portfolio_snapshot(self, snapshot):
    if hasattr(snapshot, "to_dict"):
        self.portfolio_snapshot = snapshot.to_dict()
        self.live_positions = dict(getattr(snapshot, "positions", {}) or {})
    else:
        self.portfolio_snapshot = dict(snapshot or {})


def _set_reconciliation_result(self, result):
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    self.last_reconciliation = data
    self.portfolio_fail_closed = bool(data.get("fail_closed_active", False))


State.__init__ = _phase54_state_init
State.set_live_position = _set_live_position
State.remove_live_position = _remove_live_position
State.get_live_position = _get_live_position
State.set_portfolio_snapshot = _set_portfolio_snapshot
State.set_reconciliation_result = _set_reconciliation_result
