from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional


STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_SAFE_MODE = "SAFE_MODE"
STATUS_KILL_SWITCH = "KILL_SWITCH"
STATUS_DISABLED = "DISABLED"

SEVERITY_OK = STATUS_OK
SEVERITY_WARN = STATUS_WARN
SEVERITY_SAFE_MODE = STATUS_SAFE_MODE
SEVERITY_KILL_SWITCH = STATUS_KILL_SWITCH


@dataclass
class HealthRuleResult:
    code: str
    status: str
    severity: str
    message: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> str:
        return self.status

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthEvaluation:
    ts_ms: int
    status: str
    results: List[HealthRuleResult] = field(default_factory=list)
    safe_mode: bool = False
    kill_switch: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> str:
        return self.status

    def __iter__(self) -> Iterator[HealthRuleResult]:
        # main.py treats evaluate(ctx) as iterable during escalation:
        # [r.to_dict() for r in results]
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __bool__(self) -> bool:
        return bool(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "status": self.status,
            "safe_mode": self.safe_mode,
            "kill_switch": self.kill_switch,
            "details": dict(self.details or {}),
            "results": [r.to_dict() for r in self.results],
        }


class HealthRule:
    def evaluate(self, monitor: "HealthMonitor") -> Optional[HealthRuleResult]:
        return None


class HealthMonitor:
    """
    Phase 5.6 runtime health monitor.

    Runtime compatibility with main.py:
    - __init__(cfg, store=..., stale_guard=...)
    - evaluate(ctx)
    - worst_status(results)
    - messages(results)
    - record_* counters
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        store: Any = None,
        stale_guard: Any = None,
        **kwargs: Any,
    ):
        self.store = store
        self.stale_guard = stale_guard
        self.extra_kwargs = dict(kwargs or {})

        root = dict(config or {})
        health_cfg = {}
        if isinstance(root.get("safety"), dict) and isinstance(root["safety"].get("health"), dict):
            health_cfg.update(root["safety"]["health"])
        health_cfg.update(root)

        self.config = health_cfg
        self.enabled = bool(self.config.get("enabled", True))
        self.window_seconds = float(self.config.get("window_seconds", 60.0) or 60.0)

        self.api_error_warn_threshold = self._float("api_error_warn_threshold", 10.0)
        self.api_error_kill_threshold = self._float("api_error_kill_threshold", 25.0)
        self.order_rejection_warn_threshold = self._float("order_rejection_warn_threshold", 10.0)
        self.order_rejection_kill_threshold = self._float("order_rejection_kill_threshold", 20.0)
        self.ws_disconnect_warn_threshold = self._float("ws_disconnect_warn_threshold", 5.0)
        self.ws_disconnect_kill_threshold = self._float("ws_disconnect_kill_threshold", 20.0)
        self.prediction_error_warn_threshold = self._float("prediction_error_warn_threshold", 10.0)
        self.prediction_error_kill_threshold = self._float("prediction_error_kill_threshold", 25.0)
        self.fallback_warn_threshold = self._float("fallback_warn_threshold", 10.0)
        self.fallback_kill_threshold = self._float("fallback_kill_threshold", 25.0)

        self.api_errors: Deque[int] = deque()
        self.order_rejections: Deque[int] = deque()
        self.ws_disconnects: Deque[int] = deque()
        self.prediction_errors: Deque[int] = deque()
        self.fallback_predictions: Deque[int] = deque()
        self.custom_events: Dict[str, Deque[int]] = {}

        self.last_evaluation = HealthEvaluation(ts_ms=self.now_ms(), status=STATUS_OK)

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _record(self, bucket: Deque[int], ts_ms: Optional[int] = None) -> None:
        bucket.append(self.now_ms() if ts_ms is None else int(ts_ms))
        self._prune(bucket)

    def _prune(self, bucket: Deque[int], now_ms: Optional[int] = None) -> None:
        now = self.now_ms() if now_ms is None else int(now_ms)
        cutoff = now - int(self.window_seconds * 1000)
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def _count(self, bucket: Deque[int], now_ms: Optional[int] = None) -> int:
        self._prune(bucket, now_ms=now_ms)
        return len(bucket)

    def record_api_error(self, ts_ms: Optional[int] = None, **metadata: Any) -> None:
        self._record(self.api_errors, ts_ms)

    def record_order_rejection(self, ts_ms: Optional[int] = None, **metadata: Any) -> None:
        self._record(self.order_rejections, ts_ms)

    def record_ws_disconnect(self, ts_ms: Optional[int] = None, **metadata: Any) -> None:
        self._record(self.ws_disconnects, ts_ms)

    def record_prediction_error(self, ts_ms: Optional[int] = None, **metadata: Any) -> None:
        self._record(self.prediction_errors, ts_ms)

    def record_fallback_prediction(self, ts_ms: Optional[int] = None, **metadata: Any) -> None:
        self._record(self.fallback_predictions, ts_ms)

    def record_event(self, name: str, ts_ms: Optional[int] = None, **metadata: Any) -> None:
        bucket = self.custom_events.setdefault(str(name), deque())
        self._record(bucket, ts_ms)

    def _threshold_result(
        self,
        *,
        code: str,
        value: int,
        warn_threshold: float,
        kill_threshold: float,
        message: str,
    ) -> Optional[HealthRuleResult]:
        if value >= kill_threshold:
            return HealthRuleResult(
                code=code,
                status=STATUS_KILL_SWITCH,
                severity=SEVERITY_KILL_SWITCH,
                message=message,
                value=float(value),
                threshold=float(kill_threshold),
            )
        if value >= warn_threshold:
            return HealthRuleResult(
                code=code,
                status=STATUS_SAFE_MODE,
                severity=SEVERITY_SAFE_MODE,
                message=message,
                value=float(value),
                threshold=float(warn_threshold),
            )
        return None

    def _ctx_counter(self, ctx: Any, name: str, default: int = 0) -> int:
        try:
            state = getattr(ctx, "state", None)
            return int(getattr(state, name, default) or default)
        except Exception:
            return int(default)

    def evaluate(
        self,
        ctx: Any = None,
        *,
        now_ms: Optional[int] = None,
        stale_violations: Optional[Iterable[Any]] = None,
        **kwargs: Any,
    ) -> HealthEvaluation:
        now = self.now_ms() if now_ms is None else int(now_ms)

        if not self.enabled:
            self.last_evaluation = HealthEvaluation(ts_ms=now, status=STATUS_DISABLED)
            return self.last_evaluation

        api_count = self._count(self.api_errors, now)
        rejection_count = self._count(self.order_rejections, now)
        ws_count = self._count(self.ws_disconnects, now)
        prediction_count = self._count(self.prediction_errors, now)
        fallback_count = self._count(self.fallback_predictions, now)

        # Pull soft runtime counters from ctx.state when available.
        if ctx is not None:
            api_count += self._ctx_counter(ctx, "api_error_counts", 0)
            ws_count += self._ctx_counter(ctx, "ws_disconnect_counts", 0)
            prediction_count += self._ctx_counter(ctx, "prediction_error_counts", 0)
            fallback_count += self._ctx_counter(ctx, "fallback_prediction_counts", 0)

        results: List[HealthRuleResult] = []
        candidates = [
            self._threshold_result(
                code="api_error_rate_high",
                value=api_count,
                warn_threshold=self.api_error_warn_threshold,
                kill_threshold=self.api_error_kill_threshold,
                message="API error rate exceeded configured threshold",
            ),
            self._threshold_result(
                code="order_rejection_rate_high",
                value=rejection_count,
                warn_threshold=self.order_rejection_warn_threshold,
                kill_threshold=self.order_rejection_kill_threshold,
                message="Order rejection rate exceeded configured threshold",
            ),
            self._threshold_result(
                code="ws_disconnect_rate_high",
                value=ws_count,
                warn_threshold=self.ws_disconnect_warn_threshold,
                kill_threshold=self.ws_disconnect_kill_threshold,
                message="Websocket disconnect rate exceeded configured threshold",
            ),
            self._threshold_result(
                code="prediction_error_rate_high",
                value=prediction_count,
                warn_threshold=self.prediction_error_warn_threshold,
                kill_threshold=self.prediction_error_kill_threshold,
                message="Prediction error rate exceeded configured threshold",
            ),
            self._threshold_result(
                code="prediction_fallback_active",
                value=fallback_count,
                warn_threshold=self.fallback_warn_threshold,
                kill_threshold=self.fallback_kill_threshold,
                message="Prediction fallback usage exceeded configured threshold",
            ),
        ]
        results.extend([r for r in candidates if r is not None])

        if stale_violations is None and self.stale_guard is not None:
            stale_violations = getattr(self.stale_guard, "last_violations", []) or []

        for violation in stale_violations or []:
            severity = getattr(violation, "severity", None) or getattr(violation, "status", None) or STATUS_SAFE_MODE
            code = getattr(violation, "code", "stale_data")
            message = getattr(violation, "message", "Stale data violation")
            results.append(
                HealthRuleResult(
                    code=str(code),
                    status=str(severity),
                    severity=str(severity),
                    message=str(message),
                    metadata={"source": "stale_data_guard"},
                )
            )

        kill = any(r.status == STATUS_KILL_SWITCH or r.severity == STATUS_KILL_SWITCH for r in results)
        safe = any(r.status == STATUS_SAFE_MODE or r.severity == STATUS_SAFE_MODE for r in results)
        warn = any(r.status == STATUS_WARN or r.severity == STATUS_WARN for r in results)

        status = STATUS_KILL_SWITCH if kill else STATUS_SAFE_MODE if safe else STATUS_WARN if warn else STATUS_OK
        self.last_evaluation = HealthEvaluation(
            ts_ms=now,
            status=status,
            results=results,
            safe_mode=safe,
            kill_switch=kill,
            details={
                "api_errors": api_count,
                "order_rejections": rejection_count,
                "ws_disconnects": ws_count,
                "prediction_errors": prediction_count,
                "fallback_predictions": fallback_count,
                "window_seconds": self.window_seconds,
            },
        )
        return self.last_evaluation

    def _coerce_results(self, results: Any = None) -> List[HealthRuleResult]:
        if results is None:
            results = self.last_evaluation
        if isinstance(results, HealthEvaluation):
            return list(results.results)
        if isinstance(results, HealthRuleResult):
            return [results]
        try:
            out = []
            for item in results or []:
                if isinstance(item, HealthRuleResult):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(
                        HealthRuleResult(
                            code=str(item.get("code", "unknown")),
                            status=str(item.get("status", item.get("severity", STATUS_OK))),
                            severity=str(item.get("severity", item.get("status", STATUS_OK))),
                            message=str(item.get("message", "")),
                            value=item.get("value"),
                            threshold=item.get("threshold"),
                            metadata=dict(item.get("metadata", {}) or {}),
                        )
                    )
            return out
        except TypeError:
            status = getattr(results, "status", None) or getattr(results, "severity", None) or STATUS_OK
            return [HealthRuleResult(code="unknown", status=str(status), severity=str(status))]

    def worst_status(self, results: Any = None) -> str:
        if results is None:
            results = self.last_evaluation
        if isinstance(results, HealthEvaluation):
            return results.status
        if isinstance(results, str):
            return results

        statuses = []
        for item in self._coerce_results(results):
            statuses.append(str(item.status or item.severity or STATUS_OK))

        if STATUS_KILL_SWITCH in statuses:
            return STATUS_KILL_SWITCH
        if STATUS_SAFE_MODE in statuses:
            return STATUS_SAFE_MODE
        if STATUS_WARN in statuses:
            return STATUS_WARN
        return STATUS_OK

    def messages(self, results: Any = None) -> List[str]:
        msgs: List[str] = []
        for item in self._coerce_results(results):
            text = item.message or item.code
            if text:
                if item.value is not None and item.threshold is not None:
                    text = f"{text} value={item.value} threshold={item.threshold}"
                msgs.append(str(text))
        return msgs

    def should_enter_safe_mode(self, results: Any = None) -> bool:
        return self.worst_status(results) == STATUS_SAFE_MODE

    def should_trigger_kill_switch(self, results: Any = None) -> bool:
        return self.worst_status(results) == STATUS_KILL_SWITCH

    def snapshot(self) -> Dict[str, Any]:
        return self.last_evaluation.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return self.snapshot()
