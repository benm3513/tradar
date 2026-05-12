from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("tradarbot.safety.safe_mode")


@dataclass
class SafeModePolicy:
    enabled: bool = True
    size_multiplier: float = 0.50
    max_positions_multiplier: float = 0.50
    cooldown_minutes_multiplier: float = 2.0
    disable_ml_entries: bool = False
    disable_new_entries: bool = False
    auto_recover: bool = True
    recover_after_minutes: float = 30.0
    escalate_after_minutes: Optional[float] = None


@dataclass
class SafeModeState:
    active: bool = False
    activated_ts_ms: Optional[int] = None
    released_ts_ms: Optional[int] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafeModeManager:
    """Degraded mode: entries may be reduced/muted, exits remain available."""

    def __init__(self, cfg: Dict[str, Any], store=None):
        root = dict(cfg or {})
        safety = dict(root.get("safety", {}) or {})
        block = dict(safety.get("safe_mode", {}) or root.get("safe_mode", {}) or {})
        self.policy = SafeModePolicy(
            enabled=bool(block.get("enabled", safety.get("enabled", True))),
            size_multiplier=float(block.get("size_multiplier", 0.50) or 0.50),
            max_positions_multiplier=float(block.get("max_positions_multiplier", 0.50) or 0.50),
            cooldown_minutes_multiplier=float(block.get("cooldown_minutes_multiplier", 2.0) or 2.0),
            disable_ml_entries=bool(block.get("disable_ml_entries", False)),
            disable_new_entries=bool(block.get("disable_new_entries", False)),
            auto_recover=bool(block.get("auto_recover", True)),
            recover_after_minutes=float(block.get("recover_after_minutes", 30.0) or 30.0),
            escalate_after_minutes=block.get("escalate_after_minutes"),
        )
        self.store = store
        self.state = SafeModeState()

    def activate(self, reason: str, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.policy.enabled:
            return False
        now_ms = int(time.time() * 1000)
        first = not self.state.active
        self.state.active = True
        if self.state.activated_ts_ms is None:
            self.state.activated_ts_ms = now_ms
        if reason not in self.state.reasons:
            self.state.reasons.append(str(reason))
        if metadata:
            self.state.metadata.update(metadata)
        self.state.released_ts_ms = None
        log.warning("SAFE_MODE_ENABLED reason=%s message=%s metadata=%s", reason, message, metadata or {})
        self._persist("safe_mode_enabled", "SAFE_MODE", reason, message, metadata)
        return first

    def deactivate(self, reason: str = "auto_recover", message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.state.active:
            return False
        self.state.active = False
        self.state.released_ts_ms = int(time.time() * 1000)
        self.state.reasons = []
        self.state.metadata.update(metadata or {})
        log.warning("SAFE_MODE_DISABLED reason=%s message=%s", reason, message)
        self._persist("safe_mode_disabled", "INFO", reason, message, metadata)
        return True

    def maybe_auto_recover(self, now_ms: Optional[int] = None) -> bool:
        if not (self.policy.auto_recover and self.state.active and self.state.activated_ts_ms):
            return False
        now_ms = int(now_ms or time.time() * 1000)
        age_min = (now_ms - int(self.state.activated_ts_ms)) / 60000.0
        if age_min >= self.policy.recover_after_minutes:
            return self.deactivate("auto_recover", metadata={"age_min": age_min})
        return False

    def is_active(self) -> bool:
        return bool(self.policy.enabled and self.state.active)

    def should_block_entry(self, strat_name: Optional[str] = None) -> bool:
        if not self.is_active():
            return False
        if self.policy.disable_new_entries:
            return True
        if self.policy.disable_ml_entries and str(strat_name or "").lower() in {"ml_strategy", "ml"}:
            return True
        return False

    def entry_size_multiplier(self, strat_name: Optional[str] = None) -> float:
        if self.should_block_entry(strat_name):
            return 0.0
        if self.is_active():
            return max(0.0, min(1.0, float(self.policy.size_multiplier)))
        return 1.0

    def snapshot(self) -> Dict[str, Any]:
        return {"state": self.state.to_dict(), "policy": asdict(self.policy)}

    def _persist(self, event_type: str, severity: str, reason: str, message: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        if self.store is None:
            return
        try:
            self.store.insert_safety_event(event_type=event_type, severity=severity, source="safe_mode", symbol=None, message=message or reason, details={"reason": reason, "state": self.snapshot(), **(metadata or {})})
            self.store.insert_safety_snapshot(self.is_active(), False, list(self.state.reasons), {"source": "safe_mode"})
        except Exception:
            log.exception("FAILED_TO_PERSIST_SAFE_MODE_EVENT")
