from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


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
    Deterministic replay/live risk layer.

    Phase 4.6 / 4.6.1 features:
    - daily loss cap
    - total exposure cap
    - per-symbol exposure cap
    - cooldown by symbol
    - hard drawdown stop / kill switch
    - optional drawdown-based position scaling

    Drawdown scaling regime:
    - below full-size threshold: 100%
    - full-size threshold to half-size threshold: 100%
    - half-size threshold to quarter-size threshold: configurable reduced size
    - quarter-size threshold to hard-stop threshold: configurable reduced size
    - at or above hard-stop threshold: 0% / block entries
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))

        self.max_daily_loss_usd = self._opt_float("max_daily_loss_usd")
        self.max_total_exposure_usd = self._opt_float("max_total_exposure_usd")
        self.max_total_exposure_pct = self._opt_float("max_total_exposure_pct")
        self.max_exposure_per_symbol_usd = self._opt_float("max_exposure_per_symbol_usd")
        self.max_drawdown_pct = self._opt_float("max_drawdown_pct")
        self.cooldown_minutes_per_symbol = float(
            self.config.get("cooldown_minutes_per_symbol", 0.0) or 0.0
        )

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

        self.current_timestamp: Optional[pd.Timestamp] = None
        self.current_day = None

        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.realized_pnl_today = 0.0
        self.unrealized_pnl_total = 0.0

        self.exposure_by_symbol: Dict[str, float] = {}
        self.total_exposure = 0.0
        self.last_exit_timestamp_by_symbol: Dict[str, pd.Timestamp] = {}

        self.safe_mode = False
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

    def _opt_float(self, key: str) -> Optional[float]:
        value = self.config.get(key)
        return None if value is None else float(value)

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
        positions: Dict[str, Any],
        mark_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        mark_prices = mark_prices or {}
        exposure_by_symbol: Dict[str, float] = {}

        for symbol, pos in positions.items():
            mark_price = float(mark_prices.get(symbol, getattr(pos, "entry_price", 0.0) or 0.0))
            quantity = float(getattr(pos, "quantity", 0.0) or 0.0)
            fallback_notional = float(getattr(pos, "notional_usd", 0.0) or 0.0)

            exposure = quantity * mark_price if mark_price > 0.0 and quantity > 0.0 else fallback_notional
            exposure_by_symbol[str(symbol)] = max(0.0, float(exposure))

        self.exposure_by_symbol = exposure_by_symbol
        self.total_exposure = float(sum(exposure_by_symbol.values()))
        self.unrealized_pnl_total = float(
            sum(
                float(getattr(pos, "quantity", 0.0) or 0.0)
                * (
                    float(mark_prices.get(symbol, getattr(pos, "entry_price", 0.0) or 0.0))
                    - float(getattr(pos, "entry_price", 0.0) or 0.0)
                )
                for symbol, pos in positions.items()
            )
        )
        self._refresh_triggers()

    def current_drawdown_pct(self) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

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

    def activate_kill_switch(self, reason: str = "manual") -> None:
        _ = reason
        if not self.kill_switch_triggered:
            self.kill_switch_activations_count += 1
        self.kill_switch_triggered = True
        self.safe_mode = True

    def get_position_size_multiplier(self) -> float:
        """
        Returns the drawdown throttle multiplier to apply to a proposed entry size.

        Example default regime:
        0–4% drawdown   -> 1.00
        4–6% drawdown   -> 0.50
        6–8% drawdown   -> 0.25
        >=8% drawdown   -> 0.00
        """
        if not self.enabled or not self.enable_drawdown_scaling:
            return 1.0

        dd = self.current_drawdown_pct()
        hard_stop = self.max_drawdown_pct
        if hard_stop is None:
            hard_stop = self.drawdown_quarter_size_pct

        if dd >= hard_stop:
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

        if self.kill_switch_triggered or self.safe_mode:
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
                return False, "per_symbol_exposure_limit"

        if self.max_total_exposure_usd is not None:
            if (self.total_exposure + proposed_notional) > self.max_total_exposure_usd:
                self.exposure_violations_count += 1
                self.trades_blocked_by_risk_count += 1
                return False, "total_exposure_limit_usd"

        if self.max_total_exposure_pct is not None and self.current_equity > 0.0:
            next_exposure_pct = (self.total_exposure + proposed_notional) / self.current_equity
            if next_exposure_pct > self.max_total_exposure_pct:
                self.exposure_violations_count += 1
                self.trades_blocked_by_risk_count += 1
                return False, "total_exposure_limit_pct"

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
            safe_mode=bool(self.safe_mode),
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
            "safe_mode_active": float(int(self.safe_mode)),
            "kill_switch_active": float(int(self.kill_switch_triggered)),
            "drawdown_scaling_half_count": float(self.drawdown_scaling_half_count),
            "drawdown_scaling_quarter_count": float(self.drawdown_scaling_quarter_count),
            "drawdown_scaling_stop_count": float(self.drawdown_scaling_stop_count),
        }

    def _cooldown_breached(self, symbol: str) -> bool:
        if self.cooldown_minutes_per_symbol <= 0.0:
            return False
        if self.current_timestamp is None:
            return False

        last_exit = self.last_exit_timestamp_by_symbol.get(symbol)
        if last_exit is None:
            return False

        elapsed_minutes = (self.current_timestamp - last_exit).total_seconds() / 60.0
        return elapsed_minutes < self.cooldown_minutes_per_symbol

    def _daily_loss_breached(self) -> bool:
        if self.max_daily_loss_usd is None:
            return False
        return self.current_daily_pnl() <= -self.max_daily_loss_usd

    def _drawdown_breached(self) -> bool:
        if self.max_drawdown_pct is None:
            return False
        return self.current_drawdown_pct() >= self.max_drawdown_pct

    def _refresh_triggers(self) -> None:
        if not self.enabled:
            return

        if self._daily_loss_breached() and not self._daily_loss_active:
            self._daily_loss_active = True
            self.daily_loss_triggered_count += 1
            self.safe_mode = True

        if self._drawdown_breached() and not self._drawdown_breach_active:
            self._drawdown_breach_active = True
            self.drawdown_breach_count += 1
            self.activate_kill_switch("drawdown_limit")
