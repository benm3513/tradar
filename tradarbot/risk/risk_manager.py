from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from tradarbot.core.events import OrderIntent
from tradarbot.safety.kill_switch import KillSwitchManager, KillSwitchReason
from tradarbot.safety.safe_mode import SafeModeManager
from tradarbot.safety.health_rules import STATUS_KILL_SWITCH, STATUS_SAFE_MODE

# StaleDataGuard and HealthMonitor are intentionally optional here: app/main.py
# can inject fully configured instances. RiskManager creates only the two
# entry-facing managers by default so standalone smoke tests can exercise
# kill-switch and safe-mode behavior without accidentally fail-closing on
# missing market-data timestamps.

log = logging.getLogger("tradarbot.risk")


@dataclass
class RiskSnapshot:
    timestamp: Optional[pd.Timestamp]
    current_equity: float
    peak_equity: float
    realized_pnl_today: float
    unrealized_pnl_total: float
    total_exposure: float
    drawdown_pct: float
    safe_mode: bool
    kill_switch_triggered: bool


class RiskManager:
    """
    Hybrid replay/live risk layer.

    Replay-compatible API:
    - set_timestamp
    - update_realized_pnl
    - update_equity
    - update_positions
    - current_drawdown_pct
    - current_daily_pnl
    - register_exit / mark_position_exit
    - activate_kill_switch
    - get_position_size_multiplier
    - can_enter_trade
    - should_force_exit
    - snapshot
    - summary_metrics

    Live-compatible API:
    - check(intent, ctx, strat_name)
    """

    def __init__(self, config: Dict[str, Any]):
        raw = dict(config or {})
        self.root_config = raw

        # Support either full app config or flat risk config.
        risk_block = dict(raw.get("risk", {})) if isinstance(raw.get("risk"), dict) else {}
        ml_live_block = dict(raw.get("ml_live", {})) if isinstance(raw.get("ml_live"), dict) else {}
        ml_replay_block = dict(raw.get("ml_replay", {})) if isinstance(raw.get("ml_replay"), dict) else {}

        merged: Dict[str, Any] = {}
        merged.update(ml_replay_block)
        merged.update(ml_live_block)
        merged.update(risk_block)
        if not merged:
            merged = raw

        self.config = merged
        self.enabled = bool(self.config.get("enabled", True))

        # Phase 5.6 safety managers.
        #
        # These are created by default so standalone risk smoke tests and replay/live
        # helpers can exercise the same public surface without requiring app/main.py
        # to inject them first. app/main.py may still replace them via attach_safety().
        #
        # Compatibility aliases:
        # - self.kill_switch exposes KillSwitchManager for rm.kill_switch.activate(...)
        # - self.safe_mode exposes SafeModeManager for rm.safe_mode.activate(...)
        #
        # The legacy boolean safe-mode state is stored separately in
        # self._legacy_safe_mode so old replay counters/snapshots remain boolean and
        # we do not accidentally treat a manager object as truthy forever.
        self.kill_switch_manager = self._patch_safety_manager_compat(KillSwitchManager(raw))
        self.safe_mode_manager = self._patch_safety_manager_compat(SafeModeManager(raw))
        self.health_monitor = None
        self.stale_data_guard = None
        self.kill_switch = self.kill_switch_manager
        self.safe_mode = self.safe_mode_manager

        self.max_daily_loss_usd = self._opt_float("max_daily_loss_usd")
        self.max_total_exposure_usd = self._opt_float("max_total_exposure_usd")
        self.max_total_exposure_pct = self._opt_float("max_total_exposure_pct")
        self.max_exposure_per_symbol_usd = self._opt_float("max_exposure_per_symbol_usd")
        self.max_drawdown_pct = self._opt_float("max_drawdown_pct")
        self.cooldown_minutes_per_symbol = float(self.config.get("cooldown_minutes_per_symbol", 0.0) or 0.0)
        self.cooldown_until_by_symbol: Dict[str, pd.Timestamp] = {}
        self.enable_drawdown_scaling = bool(self.config.get("enable_drawdown_scaling", False))
        self.drawdown_full_size_pct = float(self.config.get("drawdown_full_size_pct", 0.04) or 0.04)
        self.drawdown_half_size_pct = float(self.config.get("drawdown_half_size_pct", 0.06) or 0.06)
        self.drawdown_quarter_size_pct = float(self.config.get("drawdown_quarter_size_pct", 0.08) or 0.08)
        self.drawdown_half_size_multiplier = float(
            self.config.get("drawdown_half_size_multiplier", 0.50) or 0.50
        )
        self.drawdown_quarter_size_multiplier = float(
            self.config.get("drawdown_quarter_size_multiplier", 0.25) or 0.25
        )

        self.close_positions_on_kill_switch = bool(
            self.config.get("close_positions_on_kill_switch", False)
        )
        self.close_positions_on_daily_loss = bool(
            self.config.get("close_positions_on_daily_loss", False)
        )
        self.close_positions_on_drawdown = bool(
            self.config.get("close_positions_on_drawdown", False)
        )

        self.max_positions = self._opt_int("max_positions")
        self.min_notional_per_trade = float(self.config.get("min_notional_per_trade", 0.0) or 0.0)

        self.current_timestamp: Optional[pd.Timestamp] = None
        self.current_day = None

        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.realized_pnl_today = 0.0
        self.unrealized_pnl_total = 0.0

        self.exposure_by_symbol: Dict[str, float] = {}
        self.total_exposure = 0.0
        self.last_exit_timestamp_by_symbol: Dict[str, pd.Timestamp] = {}

        self._legacy_safe_mode = False
        self.kill_switch_triggered = False

        self.daily_loss_triggered_count = 0
        self.exposure_violations_count = 0
        self.trades_blocked_by_risk_count = 0
        self.forced_exits_count = 0
        self.kill_switch_activations_count = 0
        self.drawdown_breach_count = 0
        self.cooldown_blocks_count = 0
        self.daily_loss_blocks_count = 0

        self.drawdown_scaling_half_count = 0
        self.drawdown_scaling_quarter_count = 0
        self.drawdown_scaling_stop_count = 0

        self._daily_loss_active = False
        self._drawdown_breach_active = False
        self.last_portfolio_snapshot: Dict[str, Any] = {}


    def _patch_safety_manager_compat(self, manager):
        """Allow Phase 5.6 smoke-test style calls against older manager APIs.

        The safety managers generated for this phase accept activate(reason,
        message=None, metadata=None). Some tests and future call sites pass a
        source=... kwarg. Rather than requiring every safety module to have the
        exact same signature, normalize source/extra kwargs into metadata here.
        """
        if manager is None or not hasattr(manager, "activate"):
            return manager
        if getattr(manager, "_risk_compat_activate_patched", False):
            return manager

        original_activate = manager.activate

        def activate_compat(reason, message=None, metadata=None, source=None, **kwargs):
            merged_metadata = dict(metadata or {})
            if source is not None:
                merged_metadata.setdefault("source", source)
            if kwargs:
                merged_metadata.update(kwargs)
            try:
                return original_activate(reason, message=message, metadata=merged_metadata)
            except TypeError:
                try:
                    return original_activate(reason, message, merged_metadata)
                except TypeError:
                    try:
                        return original_activate(reason)
                    except TypeError:
                        return original_activate(str(reason))

        manager.activate = activate_compat
        manager._risk_compat_activate_patched = True
        return manager

    def _opt_float(self, key: str) -> Optional[float]:
        value = self.config.get(key)
        return None if value is None else float(value)

    def _opt_int(self, key: str) -> Optional[int]:
        value = self.config.get(key)
        return None if value is None else int(value)

    # ------------------------------------------------------------------
    # Replay-compatible API
    # ------------------------------------------------------------------

    def set_timestamp(self, timestamp: Optional[pd.Timestamp]) -> None:
        if timestamp is None:
            self.current_timestamp = None
            return

        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        self.current_timestamp = ts
        day = ts.date()

        if self.current_day is None:
            self.current_day = day
        elif day != self.current_day:
            self.current_day = day
            self.realized_pnl_today = 0.0
            self._daily_loss_active = False

    def update_realized_pnl(self, pnl_delta: float, timestamp: Optional[pd.Timestamp] = None) -> None:
        if timestamp is not None:
            self.set_timestamp(timestamp)
        self.realized_pnl_today += float(pnl_delta)
        self._refresh_triggers()

    def update_equity(self, equity: float) -> None:
        self.current_equity = float(equity)
        if self.peak_equity <= 0.0:
            self.peak_equity = float(equity)
        else:
            self.peak_equity = max(self.peak_equity, float(equity))
        self._refresh_triggers()

    def update_positions(
        self,
        positions: Optional[Dict[str, Any]] = None,
        mark_prices: Optional[Dict[str, float]] = None,
        *,
        exposure_by_symbol: Optional[Dict[str, float]] = None,
        total_exposure: Optional[float] = None,
        unrealized_pnl_total: Optional[float] = None,
    ) -> None:
        # Replay-style positional API:
        if positions is not None:
            mark_prices = mark_prices or {}
            computed_exposure_by_symbol: Dict[str, float] = {}

            for symbol, pos in dict(positions).items():
                current_price = mark_prices.get(symbol)

                if isinstance(pos, dict):
                    qty = float(pos.get("quantity", pos.get("qty", 0.0)) or 0.0)
                    entry_price = pos.get("entry_price", pos.get("avg_px"))
                    fallback_notional = float(pos.get("notional_usd", 0.0) or 0.0)
                else:
                    qty = float(getattr(pos, "quantity", getattr(pos, "qty", 0.0)) or 0.0)
                    entry_price = getattr(pos, "entry_price", getattr(pos, "avg_px", None))
                    fallback_notional = float(getattr(pos, "notional_usd", 0.0) or 0.0)

                if current_price is None:
                    current_price = entry_price

                exposure = 0.0
                if current_price is not None and qty > 0.0:
                    exposure = qty * float(current_price)
                else:
                    exposure = fallback_notional

                computed_exposure_by_symbol[str(symbol)] = max(0.0, float(exposure))

            self.exposure_by_symbol = computed_exposure_by_symbol
            self.total_exposure = float(sum(computed_exposure_by_symbol.values()))

            unrealized_total = 0.0
            for symbol, pos in dict(positions).items():
                current_price = mark_prices.get(symbol)
                if isinstance(pos, dict):
                    qty = float(pos.get("quantity", pos.get("qty", 0.0)) or 0.0)
                    entry_price = pos.get("entry_price", pos.get("avg_px"))
                else:
                    qty = float(getattr(pos, "quantity", getattr(pos, "qty", 0.0)) or 0.0)
                    entry_price = getattr(pos, "entry_price", getattr(pos, "avg_px", None))

                if qty <= 0.0 or current_price is None or entry_price is None:
                    continue
                unrealized_total += qty * (float(current_price) - float(entry_price))

            self.unrealized_pnl_total = float(unrealized_total)
            self._refresh_triggers()
            return

        # Live-style keyword API:
        if exposure_by_symbol is not None:
            self.exposure_by_symbol = {
                str(symbol): float(value) for symbol, value in exposure_by_symbol.items()
            }
        if total_exposure is not None:
            self.total_exposure = float(total_exposure)
        else:
            self.total_exposure = float(sum(self.exposure_by_symbol.values()))
        if unrealized_pnl_total is not None:
            self.unrealized_pnl_total = float(unrealized_pnl_total)
        self._refresh_triggers()

    def current_drawdown_pct(self) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

    def get_drawdown_pct(self) -> float:
        return self.current_drawdown_pct()

    def current_daily_pnl(self) -> float:
        return float(self.realized_pnl_today + self.unrealized_pnl_total)

    def register_exit(
        self,
        symbol: str,
        timestamp: Optional[pd.Timestamp] = None,
        forced: bool = False,
    ) -> None:
        if timestamp is not None:
            self.set_timestamp(timestamp)
        if self.current_timestamp is not None:
            self.last_exit_timestamp_by_symbol[str(symbol)] = self.current_timestamp
        if forced:
            self.forced_exits_count += 1

    def mark_position_exit(self, symbol: str, timestamp: Optional[pd.Timestamp] = None) -> None:
        self.register_exit(symbol=symbol, timestamp=timestamp, forced=False)

    def activate_kill_switch(self, reason: str = "manual") -> None:
        if not self.kill_switch_triggered:
            self.kill_switch_activations_count += 1
        self.kill_switch_triggered = True
        self._legacy_safe_mode = True
        if self.kill_switch_manager is not None:
            try:
                self.kill_switch_manager.activate(reason)
            except TypeError:
                self.kill_switch_manager.activate(str(reason))
        if self.safe_mode_manager is not None:
            try:
                self.safe_mode_manager.activate(str(reason), message="kill_switch_active")
            except TypeError:
                self.safe_mode_manager.activate(str(reason))

    def get_position_size_multiplier(self) -> float:
        if not self.enabled:
            return 1.0

        dd = self.current_drawdown_pct()
        hard_stop = self.max_drawdown_pct
        if hard_stop is None:
            hard_stop = self.drawdown_quarter_size_pct

        if not self.enable_drawdown_scaling:
            if hard_stop is not None and dd >= hard_stop:
                self.drawdown_scaling_stop_count += 1
                return 0.0
            return 1.0

        if hard_stop is not None and dd >= hard_stop:
            self.drawdown_scaling_stop_count += 1
            return 0.0
        if dd >= self.drawdown_half_size_pct:
            self.drawdown_scaling_quarter_count += 1
            return float(self.drawdown_quarter_size_multiplier)
        if dd >= self.drawdown_full_size_pct:
            self.drawdown_scaling_half_count += 1
            return float(self.drawdown_half_size_multiplier)
        return 1.0

    def can_enter_trade(self, symbol: str, proposed_notional: float) -> Tuple[bool, str]:
        if not self.enabled:
            return True, "risk_manager_disabled"

        symbol = str(symbol)
        proposed_notional = float(proposed_notional)

        if self.kill_switch_triggered:
            self.trades_blocked_by_risk_count += 1
            return False, "kill_switch_active"

        if self._daily_loss_breached():
            self.trades_blocked_by_risk_count += 1
            self.daily_loss_blocks_count += 1
            return False, "daily_loss_limit"

        if self._drawdown_breached():
            self.trades_blocked_by_risk_count += 1
            return False, "drawdown_limit"

        if self._cooldown_breached(symbol):
            self.trades_blocked_by_risk_count += 1
            self.cooldown_blocks_count += 1
            return False, "symbol_cooldown"

        current_symbol_exposure = float(self.exposure_by_symbol.get(symbol, 0.0))
        if self.max_exposure_per_symbol_usd is not None:
            if (current_symbol_exposure + proposed_notional) > self.max_exposure_per_symbol_usd:
                self.exposure_violations_count += 1
                self.trades_blocked_by_risk_count += 1
                return False, "max_exposure_per_symbol_usd"

        if self.max_total_exposure_usd is not None:
            if (self.total_exposure + proposed_notional) > self.max_total_exposure_usd:
                self.exposure_violations_count += 1
                self.trades_blocked_by_risk_count += 1
                return False, "max_total_exposure_usd"

        if self.max_total_exposure_pct is not None and self.current_equity > 0.0:
            next_exposure_pct = (self.total_exposure + proposed_notional) / self.current_equity
            if next_exposure_pct > self.max_total_exposure_pct:
                self.exposure_violations_count += 1
                self.trades_blocked_by_risk_count += 1
                return False, "max_total_exposure_pct"

        return True, "approved"

    def should_force_exit(self) -> bool:
        if not self.enabled:
            return False
        if self.kill_switch_triggered and self.close_positions_on_kill_switch:
            return True
        if self._daily_loss_breached() and self.close_positions_on_daily_loss:
            return True
        if self._drawdown_breached() and self.close_positions_on_drawdown:
            return True
        return False

    def snapshot(self) -> RiskSnapshot:
        return RiskSnapshot(
            timestamp=self.current_timestamp,
            current_equity=float(self.current_equity),
            peak_equity=float(self.peak_equity),
            realized_pnl_today=float(self.realized_pnl_today),
            unrealized_pnl_total=float(self.unrealized_pnl_total),
            total_exposure=float(self.total_exposure),
            drawdown_pct=float(self.current_drawdown_pct()),
            safe_mode=bool(self._safe_mode_active()),
            kill_switch_triggered=bool(self.kill_switch_triggered),
        )

    def summary_metrics(self) -> Dict[str, float]:
        return {
            "daily_loss_triggered": float(self.daily_loss_triggered_count),
            "exposure_violations": float(self.exposure_violations_count),
            "trades_blocked_by_risk": float(self.trades_blocked_by_risk_count),
            "forced_exits": float(self.forced_exits_count),
            "kill_switch_activations": float(self.kill_switch_activations_count),
            "drawdown_breach_events": float(self.drawdown_breach_count),
            "cooldown_blocks": float(self.cooldown_blocks_count),
            "daily_loss_blocks": float(self.daily_loss_blocks_count),
            "ending_total_exposure_usd": float(self.total_exposure),
            "ending_daily_pnl_usd": float(self.current_daily_pnl()),
            "ending_drawdown_pct": float(self.current_drawdown_pct()),
            "safe_mode_active": float(int(self._safe_mode_active())),
            "kill_switch_active": float(int(self.kill_switch_triggered)),
            "drawdown_scaling_half_count": float(self.drawdown_scaling_half_count),
            "drawdown_scaling_quarter_count": float(self.drawdown_scaling_quarter_count),
            "drawdown_scaling_stop_count": float(self.drawdown_scaling_stop_count),
        }

    # ------------------------------------------------------------------
    # Live-engine API
    # ------------------------------------------------------------------

    def update_from_portfolio_snapshot(self, snapshot) -> None:
        data = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot or {})
        self.last_portfolio_snapshot = data
        positions = dict(data.get("positions", {}) or {})
        exposure_by_symbol = {}
        unrealized_total = 0.0
        for symbol, pos in positions.items():
            if hasattr(pos, "to_dict"):
                pos = pos.to_dict()
            qty = float(pos.get("qty", 0.0) or 0.0)
            px = pos.get("current_price") or pos.get("avg_px") or 0.0
            exposure_by_symbol[str(symbol)] = max(0.0, qty * float(px or 0.0))
            unrealized_total += float(pos.get("unrealized_pnl", 0.0) or 0.0)
        self.update_positions(exposure_by_symbol=exposure_by_symbol, total_exposure=float(data.get("total_exposure", sum(exposure_by_symbol.values())) or 0.0), unrealized_pnl_total=unrealized_total)
        if data.get("equity") is not None:
            self.update_equity(float(data.get("equity") or 0.0))

    def attach_safety(self, kill_switch_manager=None, safe_mode_manager=None, health_monitor=None, stale_data_guard=None) -> None:
        if kill_switch_manager is not None:
            self.kill_switch_manager = self._patch_safety_manager_compat(kill_switch_manager)
            self.kill_switch = self.kill_switch_manager
        if safe_mode_manager is not None:
            self.safe_mode_manager = self._patch_safety_manager_compat(safe_mode_manager)
            self.safe_mode = self.safe_mode_manager
        if health_monitor is not None:
            self.health_monitor = health_monitor
        if stale_data_guard is not None:
            self.stale_data_guard = stale_data_guard

    def _safe_mode_active(self) -> bool:
        return bool(self._legacy_safe_mode) or bool(
            self.safe_mode_manager is not None and self.safe_mode_manager.is_active()
        )

    def _sync_safety_state(self, ctx) -> None:
        state = getattr(ctx, "state", None)
        if state is None:
            return
        kill_active = bool(self.kill_switch_manager and self.kill_switch_manager.is_active()) or bool(self.kill_switch_triggered)
        safe_active = self._safe_mode_active()
        self.kill_switch_triggered = kill_active
        state.runtime_kill_switch = kill_active
        state.runtime_safe_mode = safe_active

    def _record_safety_rejection(self, ctx, intent, reason: str, category: str = "safety", strat_name: Optional[str] = None) -> Dict[str, Any]:
        self.trades_blocked_by_risk_count += 1
        state = getattr(ctx, "state", None)
        if state is not None:
            state.order_rejection_counts = int(getattr(state, "order_rejection_counts", 0) or 0) + 1
        store = getattr(ctx, "store", None)
        if store is not None and hasattr(store, "insert_safety_event"):
            try:
                store.insert_safety_event(
                    event_type="entry_blocked_safety",
                    severity="WARN",
                    source="risk_manager",
                    symbol=getattr(intent, "symbol", None),
                    message=reason,
                    details={"reason_category": category, "strategy": strat_name, "side": getattr(intent, "side", None)},
                )
            except Exception:
                log.exception("FAILED_TO_PERSIST_ENTRY_BLOCKED_SAFETY")
        log.info("ENTRY_BLOCKED_SAFETY strat=%s side=%s sym=%s reason=%s", strat_name, getattr(intent, "side", None), getattr(intent, "symbol", None), reason)
        return {"approved": False, "intent": intent, "reason": reason, "reason_category": category, "strat_name": strat_name}

    def check(self, intent, ctx, strat_name: Optional[str] = None) -> Dict[str, Any]:
        if intent is None:
            return {"approved": False, "intent": intent, "reason": "missing_intent", "reason_category": "validation", "strat_name": strat_name}

        self._refresh_from_ctx(ctx)

        side = str(getattr(intent, "side", "")).upper()
        symbol = str(getattr(intent, "symbol", ""))
        qty = float(getattr(intent, "qty", 0.0) or 0.0)
        limit_px = float(getattr(intent, "limit_px", 0.0) or 0.0)
        tif = str(getattr(intent, "tif", "IOC") or "IOC").upper()

        if not symbol:
            return {"approved": False, "intent": intent, "reason": "missing_symbol", "reason_category": "validation", "strat_name": strat_name}
        if qty <= 0.0:
            return {"approved": False, "intent": intent, "reason": "non_positive_qty", "reason_category": "validation", "strat_name": strat_name}
        if limit_px <= 0.0:
            return {"approved": False, "intent": intent, "reason": "non_positive_limit_px", "reason_category": "validation", "strat_name": strat_name}
        if tif not in {"IOC", "GTC", "FOK", "MARKET"}:
            return {"approved": False, "intent": intent, "reason": "unsupported_tif", "reason_category": "validation", "strat_name": strat_name}

        self._sync_safety_state(ctx)

        # Always allow exits through, including safe mode / kill switch / stale data.
        if side == "SELL":
            log.debug("EXIT_ALLOWED_SAFETY strat=%s sym=%s qty=%s", strat_name, symbol, qty)
            return {"approved": True, "intent": intent, "reason": None, "reason_category": None, "strat_name": strat_name}

        if self.kill_switch_manager is not None and self.kill_switch_manager.should_block_entries():
            return self._record_safety_rejection(ctx, intent, "kill_switch_active", "safety", strat_name)

        if self.stale_data_guard is not None:
            violations = self.stale_data_guard.entry_violations(symbol)
            if violations:
                severe = any(v.severity == "KILL" for v in violations)
                if severe and self.kill_switch_manager is not None:
                    self.kill_switch_manager.activate(KillSwitchReason.STALE_DATA, metadata={"violations": [v.to_dict() for v in violations]})
                state = getattr(ctx, "state", None)
                if state is not None:
                    state.stale_symbols = sorted(set(list(getattr(state, "stale_symbols", []) or []) + [symbol]))
                    state.stale_global = any(v.symbol is None for v in violations)
                return self._record_safety_rejection(ctx, intent, "stale_data", "safety", strat_name)

        if self.health_monitor is not None:
            results = self.health_monitor.evaluate(ctx)
            worst = self.health_monitor.worst_status(results)
            if worst == STATUS_KILL_SWITCH and self.kill_switch_manager is not None:
                self.kill_switch_manager.activate(KillSwitchReason.HEALTH_RULE, metadata={"results": [r.to_dict() for r in results]})
                return self._record_safety_rejection(ctx, intent, "health_kill_switch", "safety", strat_name)
            if worst == STATUS_SAFE_MODE and self.safe_mode_manager is not None:
                self.safe_mode_manager.activate("health_rule", metadata={"results": [r.to_dict() for r in results]})

        if self.safe_mode_manager is not None and self.safe_mode_manager.should_block_entry(strat_name):
            return self._record_safety_rejection(ctx, intent, "safe_mode_blocks_entries", "safety", strat_name)

        if self.safe_mode_manager is not None:
            mult = self.safe_mode_manager.entry_size_multiplier(strat_name)
            if 0.0 < mult < 1.0:
                new_qty = qty * mult
                intent = OrderIntent(side=side, symbol=symbol, qty=new_qty, limit_px=limit_px, tif=tif)
                qty = new_qty
                log.info("SAFE_MODE_SIZE_REDUCED strat=%s sym=%s multiplier=%.4f qty=%.8f", strat_name, symbol, mult, qty)

        last_recon = getattr(getattr(ctx, "state", None), "last_reconciliation", {}) or {}
        if bool(getattr(getattr(ctx, "state", None), "portfolio_fail_closed", False)) or bool(last_recon.get("fail_closed_active", False)):
            if self.kill_switch_manager is not None:
                self.kill_switch_manager.activate(KillSwitchReason.RECONCILIATION_FAIL_CLOSED, metadata={"last_reconciliation": last_recon})
            return self._record_safety_rejection(ctx, intent, "portfolio_reconciliation_fail_closed", "risk", strat_name)

        live_positions = getattr(getattr(ctx, "state", None), "live_positions", {}) or {}

        if self.max_positions is not None:
            broker = getattr(ctx, "broker", None)
            positions = live_positions or (getattr(broker, "positions", {}) if broker is not None else {})
            open_count = 0
            for _, pos in dict(positions or {}).items():
                qty_val = float(getattr(pos, "qty", getattr(pos, "quantity", 0.0)) or 0.0)
                if qty_val > 0.0:
                    open_count += 1

            already_open = False
            pos = None
            if live_positions:
                pos = live_positions.get(symbol)
            if pos is None and broker is not None and hasattr(broker, "positions"):
                pos = broker.positions.get(symbol)
            if pos is not None:
                already_open = bool(float(getattr(pos, "qty", getattr(pos, "quantity", 0.0)) or 0.0) > 0.0)

            if open_count >= int(self.max_positions) and not already_open:
                self.trades_blocked_by_risk_count += 1
                return {"approved": False, "intent": intent, "reason": "max_positions", "reason_category": "exposure", "strat_name": strat_name}

        proposed_notional = qty * limit_px
        if proposed_notional < self.min_notional_per_trade:
            self.trades_blocked_by_risk_count += 1
            return {"approved": False, "intent": intent, "reason": "min_notional_per_trade", "reason_category": "validation", "strat_name": strat_name}

        broker = getattr(ctx, "broker", None)
        cash = float(getattr(broker, "cash", 0.0) or 0.0) if broker is not None else 0.0
        if cash > 0.0 and proposed_notional > cash:
            self.trades_blocked_by_risk_count += 1
            return {"approved": False, "intent": intent, "reason": "insufficient_cash", "reason_category": "capital", "strat_name": strat_name}

        if live_positions:
            exposure_by_symbol = {}
            for psym, pos in dict(live_positions).items():
                qty_val = float(getattr(pos, "qty", 0.0) or 0.0)
                px_val = getattr(pos, "current_price", None) or getattr(pos, "avg_px", 0.0)
                exposure_by_symbol[str(psym)] = max(0.0, qty_val * float(px_val or 0.0))
            self.update_positions(exposure_by_symbol=exposure_by_symbol, total_exposure=sum(exposure_by_symbol.values()))

        allowed, reason = self.can_enter_trade(symbol, proposed_notional)
        if not allowed:
            return {"approved": False, "intent": intent, "reason": reason, "reason_category": self._reason_category(reason), "strat_name": strat_name}

        return {"approved": True, "intent": intent, "reason": None, "reason_category": None, "strat_name": strat_name}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cooldown_breached(self, symbol: str) -> bool:
        symbol = str(symbol)

        if self.current_timestamp is not None:
            direct_until = self.cooldown_until_by_symbol.get(symbol)
            if direct_until is not None and self.current_timestamp < direct_until:
                return True

        if self.cooldown_minutes_per_symbol <= 0.0:
            return False
        if self.current_timestamp is None:
            return False

        last_exit = self.last_exit_timestamp_by_symbol.get(symbol)
        if last_exit is None:
            return False

        elapsed_minutes = (self.current_timestamp - last_exit).total_seconds() / 60.0
        return elapsed_minutes < self.cooldown_minutes_per_symbol
     
    def set_cooldown(self, symbol: str, cooldown_s: int) -> None:
        if cooldown_s is None or int(cooldown_s) <= 0:
            return

        now = self.current_timestamp
        if now is None:
            now = pd.Timestamp.utcnow()
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            else:
                now = now.tz_convert("UTC")
            self.current_timestamp = now

        self.cooldown_until_by_symbol[str(symbol)] = now + pd.Timedelta(seconds=int(cooldown_s))


    def clear_cooldown(self, symbol: str) -> None:
        self.cooldown_until_by_symbol.pop(str(symbol), None)

    def _daily_loss_breached(self) -> bool:
        if self.max_daily_loss_usd is None:
            return False
        return self.current_daily_pnl() <= -self.max_daily_loss_usd

    def _drawdown_breached(self) -> bool:
        if self.max_drawdown_pct is None:
            return False
        return self.current_drawdown_pct() >= self.max_drawdown_pct


    @staticmethod
    def _reason_category(reason: Optional[str]) -> Optional[str]:
        if reason is None:
            return None
        if reason in {"max_exposure_per_symbol_usd", "max_total_exposure_usd", "max_total_exposure_pct", "max_positions"}:
            return "exposure"
        if reason in {"daily_loss_limit", "drawdown_limit", "kill_switch_active", "symbol_cooldown", "portfolio_reconciliation_fail_closed"}:
            return "risk"
        if reason in {"min_notional_per_trade", "missing_symbol", "non_positive_qty", "non_positive_limit_px", "unsupported_tif"}:
            return "validation"
        if reason in {"insufficient_cash"}:
            return "capital"
        return "other"
    def _refresh_triggers(self) -> None:
        if not self.enabled:
            return

        if self._daily_loss_breached():
            if not self._daily_loss_active:
                self._daily_loss_active = True
                self.daily_loss_triggered_count += 1
            self._legacy_safe_mode = True
        else:
            self._daily_loss_active = False

        if self._drawdown_breached():
            if not self._drawdown_breach_active:
                self._drawdown_breach_active = True
                self.drawdown_breach_count += 1
                self.activate_kill_switch("drawdown_limit")
        else:
            self._drawdown_breach_active = False
        self.last_portfolio_snapshot: Dict[str, Any] = {}

    def _refresh_from_ctx(self, ctx) -> None:
        if ctx is None:
            return

        state = getattr(ctx, "state", None)
        broker = getattr(ctx, "broker", None)

        ts_ms = getattr(state, "current_event_ts_ms", None) if state is not None else None
        if ts_ms is not None:
            try:
                self.set_timestamp(pd.to_datetime(int(ts_ms), unit="ms", utc=True))
            except Exception:
                pass

        if broker is not None:
            equity = self._infer_current_equity(broker)
            if equity is not None:
                self.update_equity(equity)

            inferred_exposure = self._infer_exposure_by_symbol(broker)
            inferred_total_exposure = sum(inferred_exposure.values())
            inferred_unrealized = self._infer_unrealized_pnl(broker)
            self.update_positions(
                exposure_by_symbol=inferred_exposure,
                total_exposure=inferred_total_exposure,
                unrealized_pnl_total=inferred_unrealized,
            )

    def _infer_current_equity(self, broker) -> Optional[float]:
        for attr in ("equity", "account_equity", "current_equity"):
            value = getattr(broker, attr, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

        cash = float(getattr(broker, "cash", 0.0) or 0.0)
        mtm_value = 0.0
        positions = getattr(broker, "positions", {}) or {}
        for _, pos in dict(positions).items():
            qty = float(getattr(pos, "qty", getattr(pos, "quantity", 0.0)) or 0.0)
            if qty <= 0.0:
                continue

            price = None
            for attr in ("current_price", "avg_px", "entry_price", "mark_price", "last_price"):
                val = getattr(pos, attr, None)
                if val is not None:
                    try:
                        price = float(val)
                        break
                    except Exception:
                        pass
            if price is None:
                continue
            mtm_value += qty * price

        return cash + mtm_value

    def _infer_exposure_by_symbol(self, broker) -> Dict[str, float]:
        exposures: Dict[str, float] = {}
        positions = getattr(broker, "positions", {}) or {}
        for symbol, pos in dict(positions).items():
            qty = float(getattr(pos, "qty", getattr(pos, "quantity", 0.0)) or 0.0)
            if qty <= 0.0:
                continue

            price = None
            for attr in ("current_price", "avg_px", "entry_price", "mark_price", "last_price"):
                val = getattr(pos, attr, None)
                if val is not None:
                    try:
                        price = float(val)
                        break
                    except Exception:
                        pass
            if price is None:
                continue

            exposures[str(symbol)] = qty * price
        return exposures

    def _infer_unrealized_pnl(self, broker) -> float:
        total = 0.0
        positions = getattr(broker, "positions", {}) or {}
        for _, pos in dict(positions).items():
            for attr in ("unrealized_pnl", "unrealized_pnl_usd"):
                val = getattr(pos, attr, None)
                if val is not None:
                    try:
                        total += float(val)
                        break
                    except Exception:
                        pass
        return total
