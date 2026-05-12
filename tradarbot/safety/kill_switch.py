from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("tradarbot.safety.kill_switch")


class KillSwitchReason(str, Enum):
    MANUAL = "manual"
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss"
    API_ERRORS = "api_errors"
    ORDER_REJECTIONS = "order_rejections"
    STALE_DATA = "stale_data"
    RECONCILIATION_FAIL_CLOSED = "reconciliation_fail_closed"
    PREDICTION_CORRUPTION = "prediction_corruption"
    HEALTH_RULE = "health_rule"
    STARTUP_FAIL_CLOSED = "startup_fail_closed"


@dataclass
class KillSwitchState:
    active: bool = False
    activated_ts_ms: Optional[int] = None
    released_ts_ms: Optional[int] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KillSwitchManager:
    """Central hard-stop gate for entries. SELL/exit intents are intentionally allowed."""

    def __init__(self, cfg: Dict[str, Any], store=None):
        root = dict(cfg or {})
        safety = dict(root.get("safety", {}) or {})
        self.cfg = dict(safety.get("kill_switch", {}) or root.get("kill_switch", {}) or {})
        self.enabled = bool(self.cfg.get("enabled", safety.get("enabled", True)))
        self.allow_exit_orders = bool(self.cfg.get("allow_exit_orders", True))
        self.flatten_positions_on_trigger = bool(self.cfg.get("flatten_positions_on_trigger", False))
        self.store = store
        self.state = KillSwitchState()

    def activate(self, reason: Any, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        reason_value = getattr(reason, "value", str(reason))
        now_ms = int(time.time() * 1000)
        first_activation = not self.state.active
        self.state.active = True
        if self.state.activated_ts_ms is None:
            self.state.activated_ts_ms = now_ms
        if reason_value not in self.state.reasons:
            self.state.reasons.append(reason_value)
        if metadata:
            self.state.metadata.update(metadata)
        self.state.released_ts_ms = None
        log.warning("KILL_SWITCH_ACTIVATED reason=%s message=%s metadata=%s", reason_value, message, metadata or {})
        self._persist("kill_switch_activated", "KILL", reason_value, message, metadata)
        return first_activation

    def deactivate(self, reason: Any = "manual_release", message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        reason_value = getattr(reason, "value", str(reason))
        if not self.state.active:
            return False
        self.state.active = False
        self.state.released_ts_ms = int(time.time() * 1000)
        self.state.reasons = []
        self.state.metadata.update(metadata or {})
        log.warning("KILL_SWITCH_RELEASED reason=%s message=%s", reason_value, message)
        self._persist("kill_switch_released", "INFO", reason_value, message, metadata)
        return True

    def is_active(self) -> bool:
        return bool(self.enabled and self.state.active)

    def active_reasons(self) -> List[str]:
        return list(self.state.reasons)

    def should_block_entries(self) -> bool:
        return self.is_active()

    def should_allow_side(self, side: str) -> bool:
        side = str(side or "").upper()
        if not self.is_active():
            return True
        if side == "SELL" and self.allow_exit_orders:
            return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        out = self.state.to_dict()
        out["enabled"] = self.enabled
        out["allow_exit_orders"] = self.allow_exit_orders
        return out

    def _persist(self, event_type: str, severity: str, reason: str, message: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        if self.store is None:
            return
        try:
            self.store.insert_safety_event(
                event_type=event_type,
                severity=severity,
                source="kill_switch",
                symbol=None,
                message=message or reason,
                details={"reason": reason, "state": self.snapshot(), **(metadata or {})},
            )
            self.store.insert_safety_snapshot(False, self.is_active(), self.active_reasons(), {"source": "kill_switch"})
        except Exception:
            log.exception("FAILED_TO_PERSIST_KILL_SWITCH_EVENT")
