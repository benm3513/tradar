from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class StaleDataViolation:
    code: str
    severity: str
    symbol: Optional[str] = None
    age_seconds: Optional[float] = None
    threshold_seconds: Optional[float] = None
    message: str = ""
    source: str = "stale_data_guard"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StaleDataSnapshot:
    ts_ms: int
    enabled: bool
    stale_global: bool
    stale_symbols: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    last_candle_ts_by_symbol: Dict[str, int] = field(default_factory=dict)
    last_feature_ts_by_symbol: Dict[str, int] = field(default_factory=dict)
    last_prediction_ts_by_symbol: Dict[str, int] = field(default_factory=dict)
    last_book_ts_by_symbol: Dict[str, int] = field(default_factory=dict)
    last_ws_heartbeat_ts_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StaleDataGuard:
    def __init__(self, config: Optional[Dict[str, Any]] = None, store: Any = None, **kwargs: Any):
        self.store = store
        self.extra_kwargs = dict(kwargs or {})
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.max_candle_age_seconds = self._float("max_candle_age_seconds", 120.0)
        self.max_feature_age_seconds = self._float("max_feature_age_seconds", 300.0)
        self.max_prediction_age_seconds = self._float("max_prediction_age_seconds", 300.0)
        self.max_book_age_seconds = self._float("max_book_age_seconds", 30.0)
        self.max_ws_heartbeat_age_seconds = self._float("max_ws_heartbeat_age_seconds", 60.0)
        self.kill_severity_after_seconds = self._float(
            "kill_severity_after_seconds",
            self.config.get("stale_data_kill_seconds", 300.0),
        )
        self.last_candle_ts_by_symbol: Dict[str, int] = {}
        self.last_feature_ts_by_symbol: Dict[str, int] = {}
        self.last_prediction_ts_by_symbol: Dict[str, int] = {}
        self.last_book_ts_by_symbol: Dict[str, int] = {}
        self.last_ws_heartbeat_ts_ms: Optional[int] = None
        self.last_violations: List[StaleDataViolation] = []
        self.stale_symbols: set[str] = set()
        self.stale_global: bool = False

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _normalize_ts_ms(ts_ms: Optional[int]) -> int:
        if ts_ms is None:
            return StaleDataGuard.now_ms()
        ts = int(ts_ms)
        if ts < 10_000_000_000:
            ts *= 1000
        return ts

    def update_candle(self, symbol: str, ts_ms: Optional[int] = None) -> None:
        self.last_candle_ts_by_symbol[str(symbol)] = self._normalize_ts_ms(ts_ms)

    def update_feature(self, symbol: str, ts_ms: Optional[int] = None) -> None:
        self.last_feature_ts_by_symbol[str(symbol)] = self._normalize_ts_ms(ts_ms)

    def update_prediction(self, symbol: str, ts_ms: Optional[int] = None) -> None:
        self.last_prediction_ts_by_symbol[str(symbol)] = self._normalize_ts_ms(ts_ms)

    def update_book(self, symbol: str, ts_ms: Optional[int] = None) -> None:
        self.last_book_ts_by_symbol[str(symbol)] = self._normalize_ts_ms(ts_ms)

    def update_ws_heartbeat(self, ts_ms: Optional[int] = None) -> None:
        self.last_ws_heartbeat_ts_ms = self._normalize_ts_ms(ts_ms)

    mark_candle = update_candle
    mark_feature = update_feature
    mark_prediction = update_prediction
    mark_book = update_book
    mark_ws_heartbeat = update_ws_heartbeat

    def _age_seconds(self, ts_ms: Optional[int], now_ms: Optional[int] = None) -> Optional[float]:
        if ts_ms is None:
            return None
        now = self.now_ms() if now_ms is None else int(now_ms)
        return max(0.0, (now - int(ts_ms)) / 1000.0)

    def _severity_for_age(self, age: Optional[float], threshold: float) -> str:
        if age is None:
            return "SAFE_MODE"
        if age >= max(float(self.kill_severity_after_seconds), threshold * 2.0):
            return "KILL_SWITCH"
        return "SAFE_MODE"

    def _check_one(
        self,
        *,
        code: str,
        symbol: Optional[str],
        ts_ms: Optional[int],
        threshold_seconds: float,
        now_ms: int,
        required: bool = False,
    ) -> Optional[StaleDataViolation]:
        if ts_ms is None:
            if not required:
                return None
            return StaleDataViolation(
                code=code,
                severity="SAFE_MODE",
                symbol=symbol,
                age_seconds=None,
                threshold_seconds=float(threshold_seconds),
                message=f"{code} missing timestamp",
            )
        age = self._age_seconds(ts_ms, now_ms)
        if age is not None and age > float(threshold_seconds):
            return StaleDataViolation(
                code=code,
                severity=self._severity_for_age(age, threshold_seconds),
                symbol=symbol,
                age_seconds=age,
                threshold_seconds=float(threshold_seconds),
                message=f"{code} stale age={age:.2f}s threshold={float(threshold_seconds):.2f}s",
            )
        return None

    def check_symbol(
        self,
        symbol: str,
        *,
        now_ms: Optional[int] = None,
        require_feature: bool = False,
        require_prediction: bool = False,
        require_book: bool = False,
    ) -> List[StaleDataViolation]:
        if not self.enabled:
            return []
        sym = str(symbol)
        now = self.now_ms() if now_ms is None else int(now_ms)
        checks = [
            self._check_one(
                code="stale_candle",
                symbol=sym,
                ts_ms=self.last_candle_ts_by_symbol.get(sym),
                threshold_seconds=self.max_candle_age_seconds,
                now_ms=now,
                required=True,
            ),
            self._check_one(
                code="stale_feature",
                symbol=sym,
                ts_ms=self.last_feature_ts_by_symbol.get(sym),
                threshold_seconds=self.max_feature_age_seconds,
                now_ms=now,
                required=require_feature,
            ),
            self._check_one(
                code="stale_prediction",
                symbol=sym,
                ts_ms=self.last_prediction_ts_by_symbol.get(sym),
                threshold_seconds=self.max_prediction_age_seconds,
                now_ms=now,
                required=require_prediction,
            ),
            self._check_one(
                code="stale_book",
                symbol=sym,
                ts_ms=self.last_book_ts_by_symbol.get(sym),
                threshold_seconds=self.max_book_age_seconds,
                now_ms=now,
                required=require_book,
            ),
        ]
        violations = [v for v in checks if v is not None]
        if violations:
            self.stale_symbols.add(sym)
        else:
            self.stale_symbols.discard(sym)
        self.last_violations = violations
        return violations

    def check_global(self, *, now_ms: Optional[int] = None, require_ws_heartbeat: bool = False) -> List[StaleDataViolation]:
        if not self.enabled:
            return []
        now = self.now_ms() if now_ms is None else int(now_ms)
        violation = self._check_one(
            code="stale_ws_heartbeat",
            symbol=None,
            ts_ms=self.last_ws_heartbeat_ts_ms,
            threshold_seconds=self.max_ws_heartbeat_age_seconds,
            now_ms=now,
            required=require_ws_heartbeat,
        )
        violations = [violation] if violation is not None else []
        self.stale_global = bool(violations)
        self.last_violations = violations
        return violations

    def check(
        self,
        symbols: Optional[Iterable[str]] = None,
        *,
        now_ms: Optional[int] = None,
        require_feature: bool = False,
        require_prediction: bool = False,
        require_book: bool = False,
        require_ws_heartbeat: bool = False,
    ) -> List[StaleDataViolation]:
        violations: List[StaleDataViolation] = []
        violations.extend(self.check_global(now_ms=now_ms, require_ws_heartbeat=require_ws_heartbeat))
        for sym in symbols or []:
            violations.extend(
                self.check_symbol(
                    str(sym),
                    now_ms=now_ms,
                    require_feature=require_feature,
                    require_prediction=require_prediction,
                    require_book=require_book,
                )
            )
        self.last_violations = violations
        return violations

    def has_entry_blocking_violation(self, symbol: Optional[str] = None) -> bool:
        violations = self.last_violations
        if symbol is not None:
            violations = [v for v in violations if v.symbol in {None, symbol}]
        return any(v.severity in {"SAFE_MODE", "KILL_SWITCH"} for v in violations)
    
    def entry_violations(self, symbol: str):
        """
        Runtime compatibility helper for app/main.py.
        Returns active violations relevant to a symbol.
        """
        try:
            violations = self.check_symbol(symbol)
        except Exception:
            violations = []

        return list(violations or [])

    def snapshot(self, symbols = None) -> StaleDataSnapshot:
        return StaleDataSnapshot(
            ts_ms=self.now_ms(),
            enabled=self.enabled,
            stale_global=self.stale_global,
            stale_symbols=sorted(self.stale_symbols),
            violations=[v.to_dict() for v in self.last_violations],
            last_candle_ts_by_symbol=dict(self.last_candle_ts_by_symbol),
            last_feature_ts_by_symbol=dict(self.last_feature_ts_by_symbol),
            last_prediction_ts_by_symbol=dict(self.last_prediction_ts_by_symbol),
            last_book_ts_by_symbol=dict(self.last_book_ts_by_symbol),
            last_ws_heartbeat_ts_ms=self.last_ws_heartbeat_ts_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.snapshot().to_dict()
