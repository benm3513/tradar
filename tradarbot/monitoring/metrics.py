from __future__ import annotations

import json
import os
import resource
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


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


def _len(value: Any) -> int:
    try:
        return len(value or [])
    except Exception:
        return 0


def _rss_memory_mb() -> float:
    """Return resident memory in MB using stdlib-only APIs.

    Linux reports ru_maxrss in KiB; macOS reports bytes. The VPS target is
    Linux, but keeping the macOS case helps local development.
    """
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss > 1024 * 1024 * 10:  # likely bytes
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


def _sqlite_size_mb(store: Any) -> float:
    try:
        path = getattr(store, "path", None)
        if not path:
            db = store.conn.execute("PRAGMA database_list").fetchone()
            path = db[2] if db and len(db) >= 3 else None
        if not path:
            return 0.0
        p = Path(path)
        total = p.stat().st_size if p.exists() else 0
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(p) + suffix)
            if sidecar.exists():
                total += sidecar.stat().st_size
        return total / (1024.0 * 1024.0)
    except Exception:
        return 0.0


class MetricsCollector:
    """Collects lightweight JSON-serializable runtime metrics.

    Collection is deliberately defensive: any individual metric failure falls
    back to a zero/empty value so monitoring never crashes the trading runtime.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, process_start_ts: Optional[float] = None):
        root = dict(cfg or {})
        self.cfg = dict(root.get("monitoring", {}) or {})
        self.process_start_ts = float(process_start_ts or time.time())

    def collect(self, ctx: Any, broker: Any = None, profile: Optional[str] = None) -> Dict[str, Any]:
        ts = now_ms()
        state = getattr(ctx, "state", None)
        store = getattr(ctx, "store", None)
        cfg = getattr(ctx, "cfg", {}) or {}
        runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
        profile = profile or runtime.get("profile") or os.environ.get("TRADAR_PROFILE", "paper")
        broker = broker or getattr(ctx, "broker", None)

        metrics: Dict[str, Any] = {
            "ts_ms": ts,
            "runtime": self._runtime_metrics(state, profile),
            "market_data": self._market_data_metrics(state),
            "ml": self._ml_metrics(state),
            "execution": self._execution_metrics(ctx, broker),
            "safety": self._safety_metrics(state, ctx),
            "system": self._system_metrics(store),
        }
        return metrics

    def flatten(self, snapshot: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        ts = int(snapshot.get("ts_ms") or now_ms())
        for group, payload in snapshot.items():
            if group == "ts_ms" or not isinstance(payload, dict):
                continue
            for name, value in payload.items():
                labels = None
                metric_value: Any = value
                if isinstance(value, (dict, list, tuple, set)):
                    labels = {"value_json": json.dumps(value, sort_keys=True, default=str)}
                    metric_value = None
                yield {
                    "ts_ms": ts,
                    "metric_group": str(group),
                    "metric_name": str(name),
                    "metric_value": metric_value,
                    "labels": labels,
                }

    def _runtime_metrics(self, state: Any, profile: str) -> Dict[str, Any]:
        uptime = max(0.0, time.time() - self.process_start_ts)
        last_heartbeat_ts = getattr(state, "last_heartbeat_ts_ms", None)
        heartbeat_age = None
        if last_heartbeat_ts:
            heartbeat_age = max(0.0, (now_ms() - int(last_heartbeat_ts)) / 1000.0)
        return {
            "uptime_seconds": round(uptime, 3),
            "runtime_profile": str(profile),
            "process_start_ts": int(self.process_start_ts),
            "process_start_ts_ms": int(self.process_start_ts * 1000),
            "pid": os.getpid(),
            "heartbeat_age_seconds": heartbeat_age,
        }

    def _market_data_metrics(self, state: Any) -> Dict[str, Any]:
        ws_health = dict(getattr(state, "ws_health", {}) or {})
        rest_health = dict(getattr(state, "rest_health", {}) or {})
        market_health = dict(getattr(state, "market_data_health", {}) or {})
        ws_connected = ws_health.get("connected", market_health.get("ws_connected", False))
        return {
            "ws_connected": bool(ws_connected),
            "ws_disconnect_count": _safe_int(getattr(state, "ws_disconnect_counts", ws_health.get("disconnects", 0))),
            "poll_ok": _safe_int(getattr(state, "_poll_ok", rest_health.get("poll_ok", 0))),
            "poll_err": _safe_int(getattr(state, "_poll_err", rest_health.get("poll_err", 0))),
            "poll_backoff_s": _safe_float(getattr(state, "_poll_backoff_s", rest_health.get("backoff_s", 0.0))),
            "active_symbols": _len(getattr(state, "active_symbols", [])),
            "ready_symbols": _len(getattr(state, "rolling_ready_symbols", [])),
            "ready_symbol_list": list(getattr(state, "rolling_ready_symbols", []) or [])[:50],
            "stale_symbols": list(getattr(state, "stale_symbols", []) or []),
            "stale_global": bool(getattr(state, "stale_global", False)),
        }

    def _ml_metrics(self, state: Any) -> Dict[str, Any]:
        feature_state = getattr(state, "feature_state", None)
        feature_updates = _safe_int(getattr(feature_state, "update_count", 0))
        return {
            "feature_updates": feature_updates,
            "prediction_count": _safe_int(getattr(state, "ml_prediction_count", 0) or _len(getattr(state, "ml_latest_predictions", {}))),
            "candidate_count": _safe_int(getattr(state, "ml_candidate_count", 0) or _len(getattr(state, "ml_latest_rankings", {}))),
            "ranking_count": _safe_int(getattr(state, "ml_ranking_count", 0) or (1 if getattr(state, "ml_latest_ranking_batch", {}) else 0)),
            "signal_count": _safe_int(getattr(state, "ml_signal_count", 0) or _len(getattr(state, "ml_last_signal_ts_by_symbol", {}))),
            "inference_failures": _safe_int(getattr(state, "prediction_error_counts", 0)),
            "fallback_predictions": _safe_int(getattr(state, "fallback_prediction_counts", 0)),
            "last_refresh_ts_ms": getattr(state, "ml_last_refresh_ts_ms", None),
            "last_ranking_ts_ms": getattr(state, "ml_last_ranking_ts_ms", None),
            "top_symbols": list(getattr(state, "ml_current_top_n_symbols", []) or [])[:20],
            "shadow_prediction_count": _safe_int(getattr(state, "ml_shadow_prediction_count", 0)),
            "shadow_candidate_count": _safe_int(getattr(state, "ml_shadow_candidate_count", 0)),
            "shadow_signal_count": _safe_int(getattr(state, "ml_shadow_signal_count", 0)),
            "shadow_would_trade_count": _safe_int(getattr(state, "ml_shadow_would_trade_count", 0)),
            "shadow_blocked_execution_count": _safe_int(getattr(state, "ml_shadow_blocked_execution_count", 0)),
            "parity_check_count": _safe_int(getattr(state, "ml_parity_check_count", 0)),
            "parity_failure_count": _safe_int(getattr(state, "ml_parity_failure_count", 0)),
        }

    def _execution_metrics(self, ctx: Any, broker: Any) -> Dict[str, Any]:
        state = getattr(ctx, "state", None)
        out: Dict[str, Any] = {
            "fills": 0,
            "open_positions": 0,
            "exposure": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": 0.0,
            "cash": 0.0,
            "rejected_orders": _safe_int(getattr(state, "order_rejection_counts", 0)),
            "open_orders": 0,
        }
        if broker is None:
            return out
        try:
            positions = getattr(broker, "positions", {}) or {}
            out["open_positions"] = len([p for p in positions.values() if _safe_float(getattr(p, "qty", 0.0)) > 0])
            exposure = 0.0
            for sym, pos in positions.items():
                qty = _safe_float(getattr(pos, "qty", 0.0))
                px = _safe_float(getattr(pos, "current_price", None), 0.0) or _safe_float(getattr(pos, "avg_px", 0.0), 0.0)
                exposure += abs(qty * px)
            out["exposure"] = round(exposure, 8)
            out["cash"] = _safe_float(getattr(broker, "cash", 0.0))
            out["realized_pnl"] = _safe_float(getattr(broker, "realized_pnl", 0.0))
            try:
                out["unrealized_pnl"] = _safe_float(broker.unrealized_pnl(state))
            except Exception:
                pass
            try:
                out["equity"] = _safe_float(broker.equity(state))
            except Exception:
                out["equity"] = _safe_float(getattr(broker, "account_equity", out["cash"]))
            out["open_orders"] = len(getattr(broker, "open_orders", {}) or {})
            try:
                m = broker.metrics_snapshot()
                out["fills"] = _safe_int(m.get("trades", 0))
                out.update({f"broker_{k}": v for k, v in m.items() if isinstance(v, (int, float, str, bool))})
            except Exception:
                pass
        except Exception:
            return out
        return out

    def _safety_metrics(self, state: Any, ctx: Any) -> Dict[str, Any]:
        kill_switch = getattr(ctx, "kill_switch", None)
        safe_mode = getattr(ctx, "safe_mode", None)
        return {
            "safe_mode": bool(getattr(state, "runtime_safe_mode", False) or (safe_mode.is_active() if safe_mode and hasattr(safe_mode, "is_active") else False)),
            "kill_switch": bool(getattr(state, "runtime_kill_switch", False) or (kill_switch.is_active() if kill_switch and hasattr(kill_switch, "is_active") else False)),
            "runtime_health_status": str(getattr(state, "runtime_health_status", "UNKNOWN")),
            "api_errors": _safe_int(getattr(state, "api_error_counts", 0)),
            "order_rejections": _safe_int(getattr(state, "order_rejection_counts", 0)),
            "stale_data_violations": _len(getattr(state, "stale_symbols", [])) + (1 if getattr(state, "stale_global", False) else 0),
            "health_messages": list(getattr(state, "runtime_health_messages", []) or [])[:20],
        }

    def _system_metrics(self, store: Any) -> Dict[str, Any]:
        return {
            "rss_memory_mb": round(_rss_memory_mb(), 3),
            "thread_count": threading.active_count(),
            "sqlite_size_mb": round(_sqlite_size_mb(store), 3),
        }
